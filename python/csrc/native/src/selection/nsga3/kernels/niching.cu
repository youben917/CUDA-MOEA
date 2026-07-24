#include <thrust/sequence.h>

#include "cuda_moea/core/cuda/cuda_random.cuh"
#include "cuda_moea/selection/nsga3/kernels/non_dominated_sort.cuh"
#include "cuda_moea/core/cuda/cuda_ref_segment_ops.cuh"

//=================================================================================================================================================
//                                         PHASE 1: HISTOGRAM FOR PRIOR FRONTS (SIMPLIFIED)
//=================================================================================================================================================

/**
 * @brief Computes histogram of reference point associations for prior fronts.
 * 
 * ASSUMPTION: Contiguous layout from upstream
 * - Prior fronts occupy indices [0, N_prior) in d_asrpts_idx
 * - No gather needed: d_asrpts_idx[tidx] directly gives rpts_idx
 * 
 * @note Thread assignment: 1 thread per individual
 * @note Memory pattern: Coalesced reads, atomic histogram writes
 */
__global__ void prior_histogram_kernel(
    const int* __restrict__ d_asrpts_idx,  // input: (N_nds,) - ref point index for ALL NDS individuals
    int* __restrict__       d_rho,         // output: (N_ref,) - histogram counts (atomically updated)
    int                     N_prior        // input: scalar - number of individuals in prior fronts
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N_prior) return;
    
    // Direct access - no gather needed with contiguous layout
    const int rpts_idx = d_asrpts_idx[tidx];
    atomicAdd(&d_rho[rpts_idx], 1);
}

//=================================================================================================================================================
//                                         PHASE 2: BUILD SORT KEY FOR LAST FRONT (SIMPLIFIED)
//=================================================================================================================================================
/**
 * @brief Builds packed sorting keys for individuals in the **last non-dominated front**.
 *
 * This kernel constructs 64-bit composite keys for all individuals belonging to the
 * last front, encoding their associated reference point index and perpendicular distance.
 * The resulting keys establish a total order that groups individuals by reference point
 * and orders them by increasing distance within each group, forming the basis for
 * subsequent niche segmentation and ranking.
 *
 * ASSUMPTION (Upstream Contract):
 * - The last front occupies indices [N_prior, N_prior + N_last) in both d_asrpts_idx
 *     and d_asdist arrays.
 * - Data layout is contiguous in N_nds space; no gather or indirection is required.
 * - Distance values are non-negative angles in radians [0, π/2], and may include INF
 *     for degenerate cases.
 *
 * Key layout (64 bits):
 *   [63:32] Associated reference point index (asrpts_idx)
 *   [31:0]  Associated distance bits (reinterpretation of non-negative float as uint32,
 *           preserving monotonic ordering)
 *
 * Ordering semantics:
 * - Primary key: reference point index (groups individuals into niches)
 * - Secondary key: distance to reference line (orders individuals within each niche)
 *
 * NOTE:
 * - IEEE 754 non-negative float reinterpretation guarantees correct ordering.
 * - INF values naturally sort to the end of each reference-point segment, representing
 *     the worst association distance within that niche.
 */
__global__ void build_lf_asrpts_asdist_key_kernel(
    const int*   __restrict__ d_asrpts_idx,         // input: (N_nds,) - The indice of reference points associated with individuals after non-dominated sorting process
    const float* __restrict__ d_asdist,             // input: (N_nds,) - distance for ALL NDS individuals
    ull*         __restrict__ d_sorted_keys_ping,   // output: (N_last,) - packed sort keys for primary buffer
    int                       N_prior,              // input: scalar - offset to last front data
    int                       N_last                // input: scalar - number in last front
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N_last) return;
    
    // Direct access with offset - contiguous layout assumed
    const int   data_idx = N_prior + tidx;
    const int   rpts_idx = d_asrpts_idx[data_idx];
    const float dist     = d_asdist[data_idx];
    
    // Build packed key: (rpts_idx << 32) | dist_bits
    // Non-negative IEEE 754 floats (including INF) are monotonic when interpreted as uint32:
    //   0.0f -> 0x00000000; 1.0f -> 0x3F800000; INF -> 0x7F800000 (sorts last, which is correct for "worst distance")  
    // Place the associated reference point index into the high 32 bits of the 64-bit key
    const ull ref_part  = static_cast<ull>(static_cast<uint>(rpts_idx)) << 32; 
    // Reinterpret the non-negative IEEE 754 float distance as an unsigned 32-bit integer
    const ull dist_part = __float_as_uint(dist); // for floats >=0.0f, the bitwise representation preserves numerical order, enabling correct asdist radix sorting within each rfpt segment
    d_sorted_keys_ping[tidx] = ref_part | dist_part;
}
//================================================================================================================================================= //
//                                         PHASE 3: SEGMENT DETECTION AND LOCAL INDEX (MERGED)
//================================================================================================================================================= //

//=================================================================================================================================================
//                                         PHASE 4: RANK KEY COMPUTATION
//=================================================================================================================================================

/**
 * @brief Computes rank keys for intra-segment sorting.
 * 
 * Key layout (64 bits):
 *   [63:32] rpts_idx (MUST be here for segment detection reuse)
 *   [31:0]  rank_tiebreak:
 * - 0 for deterministic case (ρ_prior=0 AND local_idx=0, i.e., closest to empty niche)
 * - (rand | 1) for others (ensures non-zero, so deterministic always wins)
 * 
 * NSGA-III Selection Rule:
 * - ρ_prior[ref] == 0 AND local_idx == 0: deterministic selection (closest to least-crowded niche)
 * - All other cases: random tie-breaking
 */
__global__ void compute_rank_keys_kernel(
    const int* __restrict__          d_sorted_rpts_idx,        // input: (N,) - ref IDs in sorted order
    const int* __restrict__          d_niche_lf_rank,          // input: (N,) - local index within segment
    const int* __restrict__          d_rho_prior,              // input: (N_ref,) - prior niche counts
    const uint* __restrict__         d_rand_bits,              // input: (N,) - random bits for tie-breaking
    ull* __restrict__                d_sorted_keys,            // output: (N,) - keys for rank sorting
    int                              N                         // input: scalar - population size
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;
    
    const int rpts_idx  = d_sorted_rpts_idx[tidx];
    const int local_idx = d_niche_lf_rank[tidx];
    const int rho       = d_rho_prior[rpts_idx];
    
    // Deterministic case: ρ_prior = 0 AND closest (local_idx = 0)
    const int is_deterministic = (rho == 0) & (local_idx == 0);
    
    // Non-deterministic keys use (rand | 1) to guarantee they're never 0
    // This ensures deterministic case (key=0) ALWAYS wins in tie-breaking
    const uint low_bits = is_deterministic ? 0u : (d_rand_bits[tidx] | 1u);
    
    // Key layout: (rpts_idx << 32) | tiebreak
    d_sorted_keys[tidx] = (static_cast<ull>(static_cast<uint>(rpts_idx)) << 32) | static_cast<ull>(low_bits);
}

//=================================================================================================================================================
//                                         PHASE 5: FINAL NICHING KEY COMPUTATION
//=================================================================================================================================================

/**
 * @brief Computes final niching sort keys for selection ordering.
 * 
 * virtual_round = ρ_prior[ref] + rank
 * 
 * Key layout (64 bits):
 *   [63:32] virtual_round (determines selection priority - lower is better)
 *   [31:0]  secondary random (full 32 bits for minimal collision)
 * 
 * Semantics: Candidates with same virtual_round compete via random secondary key,
 * matching NSGA-III's round-robin style selection with random truncation.
 */
__global__ void compute_niching_keys_kernel(
    const int* __restrict__          d_sorted_rpts_idx,        // input: (N,) - ref IDs in current sorted order
    const int* __restrict__          d_niche_lf_rank,          // input: (N,) - rank within each niche
    const int* __restrict__          d_rho_prior,              // input: (N_ref,) - prior niche counts
    const uint* __restrict__         d_uint32_niching,         // input: (N,) - random for tie-breaking
    ull* __restrict__                d_sorted_keys,            // output: (N,) - final sort keys
    int                              N                         // input: scalar - population size
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;
    
    const int rpts_idx = d_sorted_rpts_idx[tidx];
    const int rank     = d_niche_lf_rank[tidx];
    const int rho      = d_rho_prior[rpts_idx];
    
    // Virtual round: determines selection priority (lower = earlier selection)
    const uint virtual_round = static_cast<uint>(rho + rank);
    
    // Key: (virtual_round << 32) | full_32bit_random
    d_sorted_keys[tidx] = (static_cast<ull>(virtual_round) << 32) |static_cast<ull>(d_uint32_niching[tidx]);
}

//=================================================================================================================================================
//                                         PHASE 6: FINAL GATHER (SIMPLIFIED)
//=================================================================================================================================================

/**
 * @brief Gathers selected individuals to output array with index space mapping.
 * 
 * Maps: sorted_position -> original_last_front_position -> N_nds_index -> N_mix_index
 * 
 * The d_nds_gidx array provides the mapping from N_nds space to N_mix space:
 *   d_nds_gidx[N_prior + orig_idx] gives the N_mix index for last front individual
 */
__global__ void gather_selected_kernel(
    const int* __restrict__ d_selected_orig_idx,  // input: (N_rem,) - original positions in last front (0 to N_last-1)
    const int* __restrict__ d_nds_gidx,           // input: (N_nds,) - mapping from N_nds space to N_mix space
    int* __restrict__       d_output,             // output: (N_rem,) - selected global indices in N_mix space
    int                     N_prior,              // input: scalar - offset in d_nds_gidx for last front
    int                     N_rem                 // input: scalar - number to select
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < N_rem) {
        // Map: orig_idx in [0, N_last) -> N_nds index -> N_mix index
        const int nds_idx = N_prior + d_selected_orig_idx[tidx];
        d_output[tidx] = d_nds_gidx[nds_idx];
    }
}


//=================================================================================================================================================
//                                         HELPER: RANDOM NUMBER GENERATION
//=================================================================================================================================================

//=================================================================================================================================================
//                                         MAIN NICHING PROCEDURE
//=================================================================================================================================================

/**
 * @brief Sort-based NSGA-III niching optimized with contiguous layout
 * 
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                              UPSTREAM CONTRACT (MUST BE SATISFIED)                               ║
 * ╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
 * ║ 1. d_asrpts_idx and d_asdist have contiguous layout in N_nds space:                              ║
 * ║ - Indices [0, N_prior): prior fronts data                                                     ║
 * ║ - Indices [N_prior, N_prior + N_last): last front data                                        ║
 * ║                                                                                                  ║
 * ║ 2. d_nds_gidx provides mapping from N_nds space to N_mix space:                                  ║
 * ║ - d_nds_gidx[i] for i < N_prior: N_mix index of prior front individual                        ║
 * ║ - d_nds_gidx[N_prior + j]: N_mix index of last front individual j                             ║
 * ║                                                                                                  ║
 * ║ 3. d_asdist values are non-negative (angle in radians [0, π/2]), may contain INF for edge cases  ║
 * ║                                                                                                  ║
 * ║ 4. Output d_newpop_gidx contains N_mix space indices for selected individuals                    ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
 * 
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                              ALGORITHM PHASES                                                    ║
 * ╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
 * ║ PHASE 1: Compute ρ_prior histogram (direct access in N_nds space)                                ║
 * ║ PHASE 2: Build (ref, dist) keys for last front, radix sort                                       ║
 * ║ PHASE 3: Segment detection, compute local_idx within each niche                                  ║
 * ║ PHASE 4: Build rank keys, radix sort, extract ranks                                              ║
 * ║ PHASE 5: Build final niching keys (virtual_round + random), radix sort                           ║
 * ║ PHASE 6: Gather top N_rem to output with N_nds -> N_mix index mapping                            ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
 * 
 * COMPLEXITY: O(N_last log N_last) - dominated by 3 radix sorts
 * MEMORY: ~8 * N_last * INT_SIZE working space
 */
void execute_niching(
    ull               rnd_seed,                 // input:  scalar - RNG seed for all niching random draws
    ull&              glb_rnd_offset,           // update: scalar - global RNG offset (reference for proper update propagation)

    const int*        d_nds_gidx,               // input: (N_nds,) - mapping from N_nds to N_mix space
    const float*      d_asdist,                 // input: (N_nds,) - perpendicular distance per individual (N_nds space)
    const int*        d_asrpts_idx,             // input: (N_nds,) - associated ref point per individual (N_nds space)

    int*              d_rho_prior,              // buffer: (N_ref,) - niche counts from prior fronts (Phase 1 output, Phase 4/5 input for virtual_round computation)
    
    ull*              d_sorted_keys_ping,       // buffer: (N_last,) - ping buffer for radix sort keys (alternates across 3 sorts)
                                                //                         Phase 2: (ref_idx << 32) | dist_bits
                                                //                         Phase 4: (ref_idx << 32) | rank_tiebreak  
                                                //                         Phase 5: (virtual_round << 32) | random_secondary
    
    ull*              d_sorted_keys_pong,       // buffer: (N_last,) - pong buffer for radix sort keys (CUB DoubleBuffer alternate)
    
    int*              d_sorted_lfidx_ping,      // buffer: (N_last,) - ping buffer for sorted indices in last front local space [0, N_last)
                                                //                         paired with keys through all 3 sorts, final order = selection order
    
    int*              d_sorted_lfidx_pong,      // buffer: (N_last,) - pong buffer for sorted indices (CUB DoubleBuffer alternate)
    
    int*              d_seg_head_idx,           // buffer: (N_last,) - segment boundary detection workspace (reused in Phase 3 & Phase 4)
                                                //                         Pre-scan:  -1 (non-boundary) or tidx (boundary marker)
                                                //                         Post-scan: propagated segment head index for each position
    
    int*              d_sorted_rpts_idx,        // buffer: (N_last,) - reference point IDs extracted from current sorted keys (high 32 bits)
                                                //                         updated after each sort to maintain ref_idx alignment with sorted order
    
    int*              d_niche_lf_rank,          // buffer: (N_last,) - multi-purpose rank storage (semantic changes across phases)
                                                //                         Phase 3: local_idx within segment (0 = closest to ref line)
                                                //                         Phase 4+: niching_rank after rank sort (intra-niche selection order)
                                                //                         used in Phase 5 for virtual_round = ρ_prior + rank computation
    
    uint*             d_uint32_niching,         // buffer: (N_last,) - 32-bit random number workspace (regenerated twice)
                                                //                         Phase 4: random bits for rank tiebreak (rand | 1 to avoid 0)
                                                //                         Phase 5: fresh random for final sort secondary key (full 32 bits)
    
    int*              d_newpop_gidx,            // output: (N,) - selected individual indices in N_mix space
                                                //                         [0, N_prior): prior fronts (direct copy)
                                                //                         [N_prior, N): selected from last front (Phase 6 gather)

    const NumNDSData& num_ndsdata,              // input: struct - {N_nds, N_prior, N_last, N_rem} 
    int               N_ref,                    // input: scalar - number of reference points
    int               N,                        // input: scalar - target population size
    cudaMemPool_t     exec_pool,                // input: scalar - memory pool
    cudaStream_t      exec_stream,              // input: scalar - execution stream
    bool              enable_h_save
)
{
    int N_prior = num_ndsdata.N_prior;
    int N_last  = num_ndsdata.N_last;
    int N_rem   = num_ndsdata.N_rem;

    constexpr int BLK_SIZE = 256;
    // ==================== PHASE 1: COMPUTE ρ_prior HISTOGRAM ====================
    
    CUDA_CHECK(cudaMemsetAsync(d_rho_prior, 0, N_ref * INT_SIZE, exec_stream));
    
    if (N_prior > 0) {
        // Copy prior front indices from d_nds_gidx (first N_prior elements are prior fronts)
        CUDA_CHECK(cudaMemcpyAsync(d_newpop_gidx, d_nds_gidx, N_prior * INT_SIZE, cudaMemcpyDeviceToDevice, exec_stream));
        
        // Compute histogram - direct access in N_nds space
        const int grid_prior = (N_prior + BLK_SIZE - 1) / BLK_SIZE;
        prior_histogram_kernel<<<grid_prior, BLK_SIZE, 0, exec_stream>>>(d_asrpts_idx, d_rho_prior, N_prior);
        CUDA_CHECK(cudaGetLastError());
    }
    
    if (N_rem <= 0 || N_last <= 0) {return;}
    
    // ==================== PHASE 2: BUILD KEYS AND SORT BY (ref, dist) ====================
    // Working pointers (will be updated by sort)
    ull*  d_sorted_keys = d_sorted_keys_ping;
    int*  d_sorted_idx  = d_sorted_lfidx_ping;
    const int grid_last = (N_last + BLK_SIZE - 1) / BLK_SIZE;
    // Build sort keys
    build_lf_asrpts_asdist_key_kernel<<<grid_last, BLK_SIZE, 0, exec_stream>>>(d_asrpts_idx, d_asdist, d_sorted_keys, N_prior, N_last);
    CUDA_CHECK(cudaGetLastError());
    
    // Initialize value indices: [0, 1, 2, ..., N_last)
    thrust::sequence(thrust::cuda::par_nosync.on(exec_stream), d_sorted_idx, d_sorted_idx + N_last, 0);
    
    // Radix sort by (rpts_idx, distance)
    ref_segment_ops::run_cub_radix_sorting(
        d_sorted_keys, d_sorted_idx,
        (d_sorted_keys == d_sorted_keys_ping)  ? d_sorted_keys_pong  : d_sorted_keys_ping,
        (d_sorted_idx  == d_sorted_lfidx_ping) ? d_sorted_lfidx_pong : d_sorted_lfidx_ping,
        N_last, exec_pool, exec_stream
    );
    
    // ==================== PHASE 3: SEGMENT DETECTION AND LOCAL INDEX ====================
    ref_segment_ops::mark_boundaries_extract_ref(
        d_sorted_keys, d_seg_head_idx, d_sorted_rpts_idx, N_last, exec_stream
    );
    
    ref_segment_ops::run_segment_head_scan(d_seg_head_idx, N_last, exec_pool, exec_stream);
    
    // Compute local_idx (reuse d_niche_lf_rank temporarily, will be overwritten in Phase 4)
    ref_segment_ops::compute_niche_rank(d_seg_head_idx, d_niche_lf_rank, N_last, exec_stream);
    
    // ==================== PHASE 4: RANK COMPUTATION ====================
    // Generate random bits using unified RNG API
    launch_random_uint32_kernel(d_uint32_niching, N_last, rnd_seed, glb_rnd_offset, BLK_SIZE, exec_stream);
    
    // Build rank keys (d_niche_lf_rank currently holds local_idx from Phase 3)
    compute_rank_keys_kernel<<<grid_last, BLK_SIZE, 0, exec_stream>>>(d_sorted_rpts_idx, d_niche_lf_rank, d_rho_prior, d_uint32_niching, d_sorted_keys, N_last);
    CUDA_CHECK(cudaGetLastError());
    
    // Radix sort to determine ranks
    ref_segment_ops::run_cub_radix_sorting(
        d_sorted_keys, d_sorted_idx,
        (d_sorted_keys == d_sorted_keys_ping)  ? d_sorted_keys_pong  : d_sorted_keys_ping,
        (d_sorted_idx  == d_sorted_lfidx_ping) ? d_sorted_lfidx_pong : d_sorted_lfidx_ping,
        N_last, exec_pool, exec_stream
    );
    
    // Recompute segment info after rank sort
    ref_segment_ops::mark_boundaries_extract_ref(
        d_sorted_keys, d_seg_head_idx, d_sorted_rpts_idx, N_last, exec_stream
    );
    
    ref_segment_ops::run_segment_head_scan(d_seg_head_idx, N_last, exec_pool, exec_stream);
    
    // Extract actual ranks
    ref_segment_ops::compute_niche_rank(d_seg_head_idx, d_niche_lf_rank, N_last, exec_stream);
    
    // ==================== PHASE 5: FINAL NICHING KEY COMPUTATION ====================
    
    // Generate secondary random bits using unified RNG API
    launch_random_uint32_kernel(d_uint32_niching, N_last, rnd_seed, glb_rnd_offset, BLK_SIZE, exec_stream);
    
    // Build final niching keys
    compute_niching_keys_kernel<<<grid_last, BLK_SIZE, 0, exec_stream>>>(d_sorted_rpts_idx, d_niche_lf_rank, d_rho_prior, d_uint32_niching, d_sorted_keys, N_last
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Final radix sort
    ref_segment_ops::run_cub_radix_sorting(
        d_sorted_keys, d_sorted_idx,
        (d_sorted_keys == d_sorted_keys_ping)  ? d_sorted_keys_pong  : d_sorted_keys_ping,
        (d_sorted_idx  == d_sorted_lfidx_ping) ? d_sorted_lfidx_pong : d_sorted_lfidx_ping,
        N_last, exec_pool, exec_stream
    );
    
    // ==================== PHASE 6: GATHER SELECTED INDIVIDUALS ====================
    
    const int grid_rem = (N_rem + BLK_SIZE - 1) / BLK_SIZE;
    gather_selected_kernel<<<grid_rem, BLK_SIZE, 0, exec_stream>>>(d_sorted_idx, d_nds_gidx, d_newpop_gidx + N_prior, N_prior, N_rem);
    CUDA_CHECK(cudaGetLastError());
}
