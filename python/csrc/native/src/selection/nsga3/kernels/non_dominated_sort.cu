
#include <math_constants.h>

#include "cuda_moea/core/cuda/time_utils.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <climits>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cub/cub.cuh>

#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"
#include "cuda_moea/selection/nsga3/kernels/non_dominated_sort.cuh"
// ============================================================================================================= //
namespace {

struct CudaStageEventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
};

/**
 * MODULE: Create stage timing events.
 * PURPOSE: Initialize a start/stop CUDA event pair for optional timing breakdown.
 */
inline void create_stage_events(
    CudaStageEventPair& ev  // update: struct - CUDA event pair to initialize
) {
    CUDA_CHECK(cudaEventCreate(&ev.start));
    CUDA_CHECK(cudaEventCreate(&ev.stop));
}

/**
 * MODULE: Destroy stage timing events.
 * PURPOSE: Release a CUDA event pair and reset handles to nullptr.
 */
inline void destroy_stage_events(
    CudaStageEventPair& ev  // update: struct - CUDA event pair to destroy/reset
) {
    if (ev.stop)  cudaEventDestroy(ev.stop);
    if (ev.start) cudaEventDestroy(ev.start);
    ev.start = nullptr;
    ev.stop  = nullptr;
}

/**
 * MODULE: Record stage start event.
 * PURPOSE: Record the start marker for a timed stage on the given stream.
 */
inline void stage_record_start(
    const CudaStageEventPair& ev,  // input: struct - CUDA event pair containing start event
    cudaStream_t stream            // input: handle - CUDA execution stream
) {
    CUDA_CHECK(cudaEventRecord(ev.start, stream));
}

/**
 * MODULE: Record stage stop event.
 * PURPOSE: Record the stop marker for a timed stage on the given stream.
 */
inline void stage_record_stop(
    const CudaStageEventPair& ev,  // input: struct - CUDA event pair containing stop event
    cudaStream_t stream            // input: handle - CUDA execution stream
) {
    CUDA_CHECK(cudaEventRecord(ev.stop, stream));
}

/**
 * MODULE: Measure stage duration.
 * PURPOSE: Return elapsed milliseconds between a recorded start/stop event pair.
 */
inline float stage_elapsed_ms(
    const CudaStageEventPair& ev  // input: struct - CUDA event pair with recorded timestamps
) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev.start, ev.stop));
    return ms;
}

/**
 * MODULE: Read boolean env flag.
 * PURPOSE: Treat non-empty and non-'0' environment values as enabled.
 */
inline bool env_flag_enabled(
    const char* name  // input: cstr - environment variable name
) {
    const char* value = std::getenv(name);
    return (value && value[0] != '\0' && value[0] != '0');
}

/**
 * MODULE: Cache front-propagation timing switch.
 * PURPOSE: Lazily read `ASAP_NDSORT_FRONT_PROP_TIMING` once and reuse the result.
 */
inline bool front_prop_timing_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = env_flag_enabled("ASAP_NDSORT_FRONT_PROP_TIMING") ? 1 : 0;
    }
    return (enabled != 0);
}

/**
 * MODULE: Front-propagation path heuristic state.
 * PURPOSE: Share the latest propagation `iter_count` between init sorting and iterative sorting.
 */
constexpr int FRONT_PROP_COO_QUEUE_ACTIVATE_ITER = 25;
constexpr int FRONT_PROP_CSR_QUEUE_ACTIVATE_ITER = 30;
static bool g_use_csr_queue_front_prop = true;
static int  g_prev_front_prop_iter_count = -1;

inline void reset_front_prop_path_heuristic_state() {
    g_use_csr_queue_front_prop = true;
    g_prev_front_prop_iter_count = -1;
}

inline void apply_front_prop_path_hysteresis() {
    if (g_prev_front_prop_iter_count >= FRONT_PROP_CSR_QUEUE_ACTIVATE_ITER) {
        g_use_csr_queue_front_prop = true;
    } else if (g_prev_front_prop_iter_count >= 0 &&
               g_prev_front_prop_iter_count <= FRONT_PROP_COO_QUEUE_ACTIVATE_ITER) {
        g_use_csr_queue_front_prop = false;
    }
}

inline void update_front_prop_path_heuristic_from_iter_count(
    int iter_count  // input: scalar - observed front propagation iteration count
) {
    g_prev_front_prop_iter_count = iter_count;
    apply_front_prop_path_hysteresis();
}

} // namespace
// ============================================================================================================= //
/**
 * MODULE: Shared-memory objective padding helper.
 * PURPOSE: Pad even objective counts by +1 to reduce bank conflicts.
 */
template<int M>
constexpr int padded_M() {
    // Only pad if M is even; odd M is already optimal
    return (M % 2 == 0) ? (M + 1) : M;
}
// ============================================================================================================= //
/**
 * MODULE: Constraint-dominance pair generation kernel (COO).
 * PURPOSE: Build `(p, q)` dominance pairs from fitness/CV data using warp-tiled comparison.
 * NOTES: Uses padded shared-memory tiles, branch-light dominance checks, and ballot-based pair compaction.
 */
// ============================================================================================================= //
template<int M = 3, int WPB = 6>
__global__ void constraint_domination_pairs_kernel(
    const float* __restrict__ d_mixcv,   // input: (N_mix,) - constraint violations
    const float* __restrict__ d_mixfv,   // input: (M, N_mix) - objective values
    int* __restrict__ d_dompairs,        // output: (2 * PAIRS_CAP,) - COO pairs written as `int2(p, q)`
    long long* __restrict__ d_dom_cnt,   // output: (1,) - total pair count (64-bit)
    int N_mix,                           // input: scalar - population size
    long long PAIRS_CAP)                 // input: scalar - pair buffer capacity
{
    constexpr uint32_t FMASK = 0xFFFFFFFFu;
    constexpr int M_PAD = padded_M<M>();
    
    const int N_tiles = (N_mix + WS - 1) / WS;
    const int TOT_TILES = N_tiles * N_tiles;
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;
    
    __shared__ float s_tiles_p[WPB][WS][M_PAD];
    __shared__ float s_tiles_q[WPB][WS][M_PAD];
    __shared__ uint32_t s_mask[WPB][WS];
    
    const int tile_base = blockIdx.x * WPB + warp_idx;
    const int stride = gridDim.x * WPB;
    
    for (int tile_idx = tile_base; tile_idx < TOT_TILES; tile_idx += stride) {
        const int tile_row = tile_idx / N_tiles;
        const int tile_col = tile_idx % N_tiles;
        const int tile_p_base = tile_row * WS;
        const int tile_q_base = tile_col * WS;
        
        const int glb_p = tile_p_base + lane_idx;
        const int glb_q = tile_q_base + lane_idx;
        
        const float cv_q = (glb_q < N_mix) ? __ldg(&d_mixcv[glb_q]) : CUDART_INF_F;
        const float cv_p_reg = (glb_p < N_mix) ? __ldg(&d_mixcv[glb_p]) : CUDART_INF_F;
        
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            s_tiles_p[warp_idx][lane_idx][m] = (glb_p < N_mix) ? __ldg(&d_mixfv[m * N_mix + glb_p]) : CUDART_INF_F;
            s_tiles_q[warp_idx][lane_idx][m] = (glb_q < N_mix) ? __ldg(&d_mixfv[m * N_mix + glb_q]) : CUDART_INF_F;
        }
        __syncwarp();
        
        const bool q_feas = (cv_q == FEASIBLE_CV);
        int dom_cnt = 0;
        
        #pragma unroll
        for (int bmsk_idx = 0; bmsk_idx < WS; ++bmsk_idx) {
            const int glb_row = tile_p_base + bmsk_idx;
            const int glb_col = tile_q_base + lane_idx;
            const bool lane_valid = (glb_row < N_mix) && (glb_col < N_mix) && (glb_row != glb_col);
            
            const float cv_p = __shfl_sync(FMASK, cv_p_reg, bmsk_idx);
            // Feasibility remains strict-zero. Tiny cv_fpb numerical residue
            // should be absorbed in the evaluator/kernel path, not here.
            const bool p_feas = (cv_p == FEASIBLE_CV);
            
            // Case 1: P feasible, Q infeasible → P dominates
            const bool fs_dom = p_feas & !q_feas;
            
            // Case 2: Both infeasible → smaller CV wins
            const bool cv_dom = !p_feas & !q_feas & (cv_p < cv_q);
            
            // Case 3: Both feasible → Pareto dominance
            const bool need_fv = p_feas & q_feas;
            
            bool lane_nw = lane_valid;
            bool lane_sb = false;
            
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float pv = s_tiles_p[warp_idx][bmsk_idx][m];
                const float qv = s_tiles_q[warp_idx][lane_idx][m];
                const bool nw_step = (pv <= qv);
                const bool sb_step = (pv < qv);
                
                lane_nw &= (nw_step | !need_fv);
                lane_sb |= (sb_step & need_fv);
            }
            
            const bool fv_dom = need_fv & lane_nw & lane_sb;
            
            // Final decision: P dominates Q if any case is true
            const bool lane_dom = lane_valid & (fs_dom | cv_dom | fv_dom);
            const uint32_t p_dom_q = __ballot_sync(FMASK, lane_dom);
            
            if (lane_idx == 0) {
                s_mask[warp_idx][bmsk_idx] = p_dom_q;
            }
            dom_cnt += __popc(p_dom_q);
        }
        
        if (dom_cnt == 0) continue;
        
        ull output_base = 0ULL;
        if (lane_idx == 0) {
            output_base = atomicAdd(reinterpret_cast<ull*>(d_dom_cnt), static_cast<ull>(dom_cnt));
        }
        output_base = __shfl_sync(FMASK, output_base, 0);
        
        int bitmask_offset = 0;
        for (int bmsk_idx = 0; bmsk_idx < WS; ++bmsk_idx) {
            const uint32_t bitmask = s_mask[warp_idx][bmsk_idx];
            const int bits_cnt = __popc(bitmask);
            if (bits_cnt == 0) continue;
            
            const int glb_row = tile_p_base + bmsk_idx;
            const int glb_col = tile_q_base + lane_idx;
            
            const uint32_t lane_bitmask = 1u << lane_idx;
            if (bitmask & lane_bitmask) {
                const uint32_t ps_mask = bitmask & (lane_bitmask - 1u);
                const int lane_offset = __popc(ps_mask);
                const int output_offset = bitmask_offset + lane_offset;
                const ull dompairs_idx = output_base + static_cast<ull>(output_offset);
                
                if (dompairs_idx < static_cast<ull>(PAIRS_CAP)) {
                    reinterpret_cast<int2*>(d_dompairs)[static_cast<size_t>(dompairs_idx)] = make_int2(glb_row, glb_col);
                }
            }
            bitmask_offset += bits_cnt;
        }
    }
}
// ============================================================================================================= //
/**
 * MODULE: Constraint-dominance pair launcher (COO).
 * PURPOSE: Reset pair counter and dispatch the template-specialized pair kernel for runtime `M`.
 * NOTES: Keeps async execution; callers clamp `d_dom_cnt` by `PAIRS_CAP` if overflow occurs.
 */
// ================================================================================================== //
void compute_dompairs(
    const float* d_mixcv,    // input: (N_mix,) - constraint violations
    const float* d_mixfv,    // input: (M, N_mix) - objective values
    int*         d_dompairs, // output: (2 * PAIRS_CAP,) - COO dominance pairs
    ll*          d_dom_cnt,  // output: (1,) - total pair count (64-bit)
    int          N_mix,      // input: scalar - population size
    int          M,          // input: scalar - objective count
    ll           PAIRS_CAP,  // input: scalar - pair buffer capacity
    cudaStream_t exec_stream // input: handle - CUDA execution stream
)
{
    // ===== Step 1: Initialize global counter =====
    CUDA_CHECK(cudaMemsetAsync(d_dom_cnt, 0, LL_SIZE, exec_stream));
    
    // ===== Step 2: Configure launch parameters =====
    constexpr int WPB   = 6;                     // Warps per block (tuned value for constraint version)
    const int BLK_SIZE  = WPB * WS;              // Total threads per block (256)
    const int N_tiles   = (N_mix + WS - 1) / WS; // Tiles per dimension (ceiling division)
    const int TOT_TILES = N_tiles * N_tiles;     // Total tiles in N_mix×N_mix matrix
    
    // Compute grid size: each block handles WPB tiles concurrently via warps
    const int GRID_SIZE         = (TOT_TILES + WPB - 1) / WPB;
    const int APPLIED_GRID_SIZE = min(GRID_SIZE, SM_COUNT * BLK_SIZE); // Cap for load balancing

    // ===== Step 3: Launch kernel specialized by M =====
    // Each template specialization automatically applies:
    // - Padding optimization for even M (bank conflict elimination)
    // - Computational masking for branch-free constraint-dominance evaluation
    #define LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(M) \
        constraint_domination_pairs_kernel<M, WPB> \
            <<<APPLIED_GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(d_mixcv, d_mixfv, d_dompairs, d_dom_cnt, N_mix, PAIRS_CAP)
    
    switch (M) {
        case 1: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(1); break;
        case 2: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(2); break;
        case 3: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(3); break;
        case 4: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(4); break;
        case 5: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(5); break;
        case 6: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(6); break;
        case 7: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(7); break;
        case 8: LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL(8); break;
        default:
            std::cout << "ERROR: Unsupported objective count M = " << M
                      << " (max supported: 8)" << std::endl;
            return;
    }
    #undef LAUNCH_CONSTRAINT_DOMINATION_PAIRS_KERNEL
    
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());  // Uncomment for synchronous execution
}
// ============================================================================================================= //
/**
 * MODULE: Dominance bitmask kernel (CSR bitset form).
 * PURPOSE: Compute `q dominates p` flags tile-by-tile and store them as row-wise packed bitmasks.
 * NOTES: Reuses the same constraint-dominance rules as COO pair generation, with reversed comparison direction.
 */
// ================================================================================================================== //

// ===== Bank Conflict Elimination via Compile-Time Padding =====
// For even M values, padding +1 element eliminates bank conflicts by converting
// the stride from even to odd, achieving gcd(M, 32) = 1.
// 
// Analysis:
// - ODD M (3,5,7):  gcd(M, 32) = 1  → No conflicts (optimal)
// - EVEN M (4,6,8): gcd(M, 32) > 1  → Conflicts present
//
// Solution: Pad even M to M+1 (always odd) → conflict-free access
// Memory overhead: +12.5% to +25% (negligible for typical M values)
// Performance gain: Up to 8× faster for M=8 (8-way conflict → conflict-free)
// ====================================================================================================================== //
template<int M = 3, int WPB = 8>
__global__ void constraint_dominatee_bitmask_kernel(
    const float* __restrict__ d_mixcv,          // input: (N_mix,) - constraint violations
    const float* __restrict__ d_mixfv,          // input: (M, N_mix) - objective values
    uint32_t*    __restrict__ d_dominatee_mask, // output: (N_mix, N_tiles) - packed domination bitmask
    int N_mix                                   // input: scalar - population size
)
{
    constexpr uint32_t FMASK = 0xFFFFFFFFu;      // Full-warp ballot bitmask for synchronization
    
    constexpr int M_PAD = padded_M<M>();
    
    const int N_tiles   = (N_mix + WS - 1) / WS; // Tiles per dimension (ceiling division)
    const int TOT_TILES = N_tiles * N_tiles;     // Total number of tiles in the matrix

    const int tidx     = threadIdx.x;            // Thread index within the block
    const int warp_idx = tidx / WS;              // Warp index within the block (0 to WPB-1)
    const int lane_idx = tidx % WS;              // Lane index within the warp (0 to 31)

    __shared__ float    s_tiles_p[WPB][WS][M_PAD];   // P-tile: rows representing dominated candidates
    __shared__ float    s_tiles_q[WPB][WS][M_PAD];   // Q-tile: columns representing potential dominators
    __shared__ uint32_t s_bitmask[WPB][WS];          // Per-row 32-bit bitmask encoding "q dominates p"

    const int tile_base = blockIdx.x * WPB + warp_idx;  // Starting tile for this warp
    const int stride    = gridDim.x  * WPB;             // Tile stride across the grid

    for (int tile_idx = tile_base; tile_idx < TOT_TILES; tile_idx += stride)
    {
        const int tile_row    = tile_idx / N_tiles;   // Tile row index (P dimension)
        const int tile_col    = tile_idx % N_tiles;   // Tile column index (Q dimension)
        const int tile_p_base = tile_row * WS;        // Global row offset for P tile
        const int tile_q_base = tile_col * WS;        // Global column offset for Q tile

        const int glb_p = tile_p_base + lane_idx;     // Global P row for this lane
        const int glb_q = tile_q_base + lane_idx;     // Global Q column for this lane

        const float cv_q     = (glb_q < N_mix) ? __ldg(&d_mixcv[glb_q]) : CUDART_INF_F;  // Register (reused 32× per tile)
        const float cv_p_reg = (glb_p < N_mix) ? __ldg(&d_mixcv[glb_p]) : CUDART_INF_F;  // Register (broadcast 32× via shfl)

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            s_tiles_p[warp_idx][lane_idx][m] = (glb_p < N_mix) ? __ldg(&d_mixfv[m * N_mix + glb_p]) : CUDART_INF_F;
            s_tiles_q[warp_idx][lane_idx][m] = (glb_q < N_mix) ? __ldg(&d_mixfv[m * N_mix + glb_q]) : CUDART_INF_F;
        }       
        __syncwarp();  // Ensure all lanes have completed loading before dominance checks

        // CRITICAL: Now computing "q dominates p" (reversed from pairs kernel)
        // Dominance criterion: (q <= p for ALL objectives) AND (q < p for SOME objective)
        
        const bool q_feas = (cv_q == FEASIBLE_CV);  // Q's feasibility (loop-invariant)
        
        #pragma unroll
        for (int bmsk_idx = 0; bmsk_idx < WS; ++bmsk_idx) {
            const int glb_row = tile_p_base + bmsk_idx;   // Global P row being dominated
            const int glb_col = tile_q_base + lane_idx;   // Global Q column (dominator candidate)

            // Validity check: skip if out of bounds or comparing same individual
            const bool lane_valid = (glb_row < N_mix) && (glb_col < N_mix) && (glb_row != glb_col);

            // Broadcast P's constraint violation from lane bmsk_idx
            const float cv_p  = __shfl_sync(FMASK, cv_p_reg, bmsk_idx);
            const bool p_feas = (cv_p == FEASIBLE_CV);

            // ===== Constraint-Dominance Logic (REVERSED: q dominates p) =====
            // Three mutually exclusive cases determine dominance:
            //   1. Feasible dominates infeasible (regardless of objectives)
            //   2. Both infeasible → less violated dominates
            //   3. Both feasible → Pareto dominance applies
            
            // Case 1: Q feasible, P infeasible → Q dominates P
            const bool fs_dom = q_feas & !p_feas;

            // Case 2: Both infeasible → smaller CV wins
            const bool cv_dom = !q_feas & !p_feas & (cv_q < cv_p);

            // Case 3: Both feasible → Pareto dominance applies
            const bool need_fv = q_feas & p_feas;
            
            bool lane_nw = lane_valid;      // Q not worse than P on all objectives
            bool lane_sb = false;           // Q strictly better than P on some objective

            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float pv = s_tiles_p[warp_idx][bmsk_idx][m];
                const float qv = s_tiles_q[warp_idx][lane_idx][m];
                
                // REVERSED COMPARISON: q dominates p (not p dominates q)
                const bool nw_step = (qv <= pv);
                const bool sb_step = (qv <  pv);
                
                // Accumulate results only when both are feasible (need_fv=true)
                // Bitwise masking: neutral values when need_fv=false
                lane_nw &= (nw_step | !need_fv);  // Force true if not needed
                lane_sb |= (sb_step &  need_fv);  // Force false if not needed
            }

            const bool fv_dom = need_fv & lane_nw & lane_sb;

            // Final decision: Q dominates P if any case is true
            const bool lane_dom = lane_valid & (fs_dom | cv_dom | fv_dom);
            
            // ===== Single ballot operation (optimized from 3 ballots) =====
            const uint32_t q_dom_p = __ballot_sync(FMASK, lane_dom);

            // Store the 32-bit dominance bitmask for this P-row
            if (lane_idx == 0) {
                s_bitmask[warp_idx][bmsk_idx] = q_dom_p;
            }
        }

        // ===== Write compressed bitsets directly to global memory =====
        // Each P-row writes its bitmask to d_dominatee_mask[p * N_tiles + tile_col]
        // Parallel write: each lane handles one row of the tile (WS rows total)
        const int glb_row = tile_p_base + lane_idx;
        if (glb_row < N_mix) {
            const int output_idx = glb_row * N_tiles + tile_col;
            d_dominatee_mask[output_idx] = s_bitmask[warp_idx][lane_idx];
        }
    }
}
// =============================================================================================== //
/**
 * MODULE: Dominance bitmask launcher.
 * PURPOSE: Dispatch the bitmask kernel specialization for runtime `M` and current population size.
 * NOTES: Produces `(N_mix, ceil(N_mix/32))` row-wise packed domination bitsets.
 */
// ========================================================================================================================== //
void compute_dominatee_bitmask(
    const float* __restrict__ d_mixcv,          // input: (N_mix,) - constraint violations
    const float* __restrict__ d_mixfv,          // input: (M, N_mix) - objective values
    uint32_t*    __restrict__ d_dominatee_mask, // output: (N_mix, N_tiles) - packed domination bitmask
    int N_mix,                                  // input: scalar - merged population size
    int M,                                      // input: scalar - objective count
    cudaStream_t exec_stream                    // input: handle - CUDA execution stream
)
{   
    // ===== Step 1: Configure launch parameters =====
    constexpr int WPB   = 6;
    const int BLK_SIZE  = WPB * WS;
    const int N_tiles   = (N_mix + WS - 1) / WS;
    const int TOT_TILES = N_tiles * N_tiles; // tile is squared layout: (TS, TS)

    // Each block contains WPB warps; each warp takes one tile at a time → ceil(TOT_TILES/WPB) blocks
    const int GRID_SIZE         = (TOT_TILES + WPB - 1) / WPB;
    const int APPLIED_GRID_SIZE = min(GRID_SIZE, SM_COUNT * BLK_SIZE); // SM_COUNT * BLK_SIZE

    // ===== Step 2: Launch kernel specialized by M =====
    // Note: Each instantiation automatically applies padding for even M via padded_M<M>()
    #define LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(M) \
        constraint_dominatee_bitmask_kernel<M, WPB> \
            <<<APPLIED_GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(d_mixcv, d_mixfv, d_dominatee_mask, N_mix)
    
    switch (M) {
        case 1: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(1); break;
        case 2: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(2); break;
        case 3: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(3); break;
        case 4: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(4); break;
        case 5: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(5); break;
        case 6: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(6); break;
        case 7: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(7); break;
        case 8: LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(8); break;
        default:
            std::cout << "ERROR (NDSort): Unsupported objective count M = " << M
                      << " (max supported: 8)" << std::endl;
            return;
    }
    #undef LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL
    
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
// ================================================================================================== //
/**
 * MODULE: COO dominatee-count kernel.
 * PURPOSE: Convert COO dominance pairs `(p, q)` into per-node domination counts by atomic accumulation.
 */
__global__ void count_dominatee_from_dompairs_kernel(
    const int* __restrict__ d_dompairs, // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int* __restrict__ d_dominatee,      // update: (N_mix,) - domination count per individual
    ll d_dom_cnt,                       // input: scalar - valid COO pair count
    int N_mix                           // input: scalar - population size (bounds check)
)
{
    // ===== Step 1: Global thread index and range check =====
    ll tidx = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= d_dom_cnt) return;

    // ===== Step 2: Read COO pair and atomically accumulate =====
    const int2 pair = reinterpret_cast<const int2*>(d_dompairs)[tidx];
    const int dominatee = pair.y;   // q: dominated individual
    if (dominatee >= 0 && dominatee < N_mix) {
        atomicAdd(&d_dominatee[dominatee], 1);
    }
}
// ================================================================================================== //
/**
 * MODULE: COO dominatee-count launcher.
 * PURPOSE: Launch the COO counting kernel and populate `d_dominatee` from COO pairs.
 */
// ================================================================================================== //
void count_dominatee_from_dompairs(
    const int*   d_dompairs,  // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int*         d_dominatee, // update: (N_mix,) - domination count per individual
    ll           dom_count,   // input: scalar - valid COO pair count
    int          N_mix,       // input: scalar - population size
    cudaStream_t exec_stream  // input: handle - CUDA execution stream
)
{
    if (dom_count <= 0) {
        return;
    }

    // ===== Step 1: Configure launch bounds =====
    constexpr ll BLK_SIZE = 256;
    const ll GRID_SIZE = (dom_count + BLK_SIZE - 1) / BLK_SIZE;

    // ===== Step 2: Dispatch COO counting kernel =====
    count_dominatee_from_dompairs_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>
                                        (d_dompairs, d_dominatee, dom_count, N_mix);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
// =============================================================================================== //
/**
 * MODULE: Bitmask row popcount kernel.
 * PURPOSE: Sum packed domination bits per row and output row-wise domination counts.
 * NOTES: Uses an aligned `uint2` fast path when row storage permits vectorized loads.
 */
// memory alignment version (so claimed)
template <unsigned int WPB>
__global__ void count_dominatee_from_bitmask_kernel(
    const uint32_t* __restrict__ d_dominatee_mask, // input: (N_mix, N_tiles) - packed domination bitmask
    int* __restrict__ d_dominatee,                 // output: (N_mix,) - domination count per row
    int N_mix,                                     // input: scalar - population size
    int N_tiles)                                   // input: scalar - words per row
{
    constexpr unsigned int BLK_SIZE = WPB * WS;

    const int row = blockIdx.x;
    if (row >= N_mix) return;

    const size_t row_offset = (size_t)row * N_tiles;
    int local_sum = 0;

    // ===== Key Fix: Check alignment before vectorized access =====
    // uint2 requires 8-byte alignment, which means row_offset must be even
    // (since each uint32_t is 4 bytes, offset*4 must be divisible by 8)
    const bool is_aligned = ((row_offset & 1) == 0);
    
    if (is_aligned && N_tiles >= 2) {
        // Vectorized path: row start address is 8-byte aligned
        const uint2* row_ptr_u2 = reinterpret_cast<const uint2*>(d_dominatee_mask + row_offset);
        const int u2_count = N_tiles / 2;
        
        for (int i = threadIdx.x; i < u2_count; i += BLK_SIZE) {
            const uint2 data = row_ptr_u2[i];
            local_sum += __popc(data.x) + __popc(data.y);
        }
        
        // Handle trailing element if N_tiles is odd
        if ((N_tiles & 1) && threadIdx.x == 0) {
            local_sum += __popc(d_dominatee_mask[row_offset + N_tiles - 1]);
        }
    } else {
        // Scalar fallback path: row start address is NOT 8-byte aligned
        // This occurs when N_tiles is odd AND row is odd
        for (int i = threadIdx.x; i < N_tiles; i += BLK_SIZE) {
            local_sum += __popc(d_dominatee_mask[row_offset + i]);
        }
    }

    // ===== Intra-warp reduction =====
    #pragma unroll
    for (int offset = WS >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }

    // ===== Inter-warp reduction via shared memory =====
    __shared__ int warp_sums[WPB];

    const int warp_idx = threadIdx.x / WS;
    const int lane_idx = threadIdx.x % WS;

    if (lane_idx == 0) {
        warp_sums[warp_idx] = local_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int final_sum = 0;
        #pragma unroll
        for (int i = 0; i < WPB; ++i) {
            final_sum += warp_sums[i];
        }
        d_dominatee[row] = final_sum;
    }
}
// ================================================================================================== //
/**
 * MODULE: Bitmask row popcount launcher.
 * PURPOSE: Launch the row-wise popcount kernel and fill `d_dominatee` from packed bitmasks.
 */
void count_dominatee_from_bitmask(
    const uint32_t* d_dominatee_mask, // input: (N_mix, N_tiles) - packed domination bitmask
    int*            d_dominatee,      // output: (N_mix,) - domination count per individual
    int             N_mix,            // input: scalar - population size
    cudaStream_t    exec_stream       // input: handle - CUDA execution stream
)
{
    // ===== Step 1: Derive bitmask geometry =====
    const int N_tiles = (N_mix + WS - 1) / WS;
    // ===== Step 2: Configure launch parameters =====
    constexpr int WPB  = 8;            // Warps per block for counting
    constexpr int BLK_SIZE = WPB * WS;
    // ===== Step 3: Launch the row-wise bitcount kernel =====
    count_dominatee_from_bitmask_kernel<WPB><<<N_mix, BLK_SIZE, 0, exec_stream>>>
                                        (d_dominatee_mask, d_dominatee, N_mix, N_tiles);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
// ================================================================================================== //
/**
 * MODULE: Front-0 assignment kernel.
 * PURPOSE: Assign individuals with zero domination count to front 0 and count assignments.
 */
// =============================================================================================== //
__global__ void front_initialization_kernel(
    const int* __restrict__ d_dominatee, // input: (N_mix,) - domination count per individual
    int* __restrict__ d_fronts,          // update: (N_mix,) - front index per individual
    int* __restrict__ d_asgn_cnt,        // update: (1,) - number of assigned individuals
    int N_mix                            // input: scalar - population size
)
{
    // ===== Step 1: Global thread index and range check =====
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_mix) return;

    // ===== Step 2: Assign front 0 to non-dominated entries =====
    if (d_dominatee[tid] == 0) {
        const int old_front = atomicCAS(&d_fronts[tid], -1, 0);
        if (old_front == -1) {
            atomicAdd(d_asgn_cnt, 1);
        }
    }
}
// =============================================================================================== //
/**
 * MODULE: Front-0 assignment launcher.
 * PURPOSE: Launch `front_initialization_kernel` to initialize the first Pareto front.
 */
// =============================================================================================== //
void initialize_front(
    const int*   d_dominatee, // input: (N_mix,) - domination count per individual
    int*         d_fronts,    // update: (N_mix,) - front index per individual
    int*         d_asgn_cnt,  // update: (1,) - number of assigned individuals
    int          N_mix,       // input: scalar - population size
    cudaStream_t exec_stream  // input: handle - CUDA execution stream
)
{   
    // ===== Step 1: Configure launch bounds =====
    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE = (N_mix + BLK_SIZE - 1) / BLK_SIZE;

    // ===== Step 2: Dispatch front-0 assignment kernel =====
    front_initialization_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(d_dominatee, d_fronts, d_asgn_cnt, N_mix);

    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
// =============================================================================================== //
/**
 * MODULE: COO front propagation kernel.
 * PURPOSE: Process COO pairs for one front level and enqueue newly released dominatees to the next front.
 */
// =============================================================================================== //
__global__ void fronts_propagation_kernel(
    int* __restrict__ d_dompairs,  // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int* __restrict__ d_dominatee, // update: (N_mix,) - remaining domination count
    int* __restrict__ d_fronts,    // update: (N_mix,) - front index per individual
    int* __restrict__ d_new_front, // update: (1,) - next-front discovery flag
    int* __restrict__ d_asgn_cnt,  // update: (1,) - cumulative assignment count
    ll d_dom_cnt,                  // input: scalar - valid COO pair count
    int N_mix,                     // input: scalar - population size
    int max_front)                 // input: scalar - current front index
{
    // ===== Step 1: Thread indexing with 64-bit support =====
    ll tidx   = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    ll stride = (ll)blockDim.x * gridDim.x;
    int lane_idx = threadIdx.x % WS;
    
    int lane_asn = 0;
    int lane_new = 0;
    
    // ===== Step 2: Grid-stride loop over dominance pairs =====
    for (ll s = tidx; s < d_dom_cnt; s += stride) {
        int2 pair = reinterpret_cast<int2*>(d_dompairs)[s];
        int dominator = pair.x;
        int dominatee = pair.y;
        
        if (d_fronts[dominator] != max_front) continue;
        
        // ===== Step 3: Update dominatee count and assign if ready =====
        int prev = atomicSub(&d_dominatee[dominatee], 1);
        
        if (prev == 1) {
            int old = atomicMax(&d_fronts[dominatee], max_front + 1);
            
            if (old == -1) {
                lane_asn += 1;
                lane_new = 1;
            }
        }
    }
    // ===== Step 4: Warp-level reduction for aggregation =====
    lane_asn = warp_reduce_sum<int, WS>(lane_asn);
    lane_new = warp_reduce_or<int, WS>(lane_new);
    // constexpr uint32_t FMASK = 0xFFFFFFFFu;
    // const uint32_t bits      = __ballot_sync(FMASK, lane_new != 0);
    // lane_new = (bits != 0) ? 1 : 0;

    // ===== Step 5: Commit results to global memory =====
    if (lane_idx == 0) {
        if (lane_asn) atomicAdd(d_asgn_cnt, lane_asn);
        if (lane_new) atomicOr(d_new_front, 1);
    }
}
// ==============================================================================
/**
 * MODULE: COO front propagation loop.
 * PURPOSE: Iteratively apply the COO propagation kernel until no new front appears or `N` is reached.
 */
// =============================================================================================== //
static std::pair<int, int> propagate_fronts_coo(
    int*         d_dompairs,   // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int*         d_dominatee,  // update: (N_mix,) - remaining domination count
    int*         d_fronts,     // update: (N_mix,) - front index per individual
    int*         d_new_front,  // update: (1,) - next-front discovery flag
    int*         d_asgn_cnt,   // update: (1,) - cumulative assignment count
    ll           dom_count,    // input: scalar - valid COO pair count
    int          N_mix,        // input: scalar - population size
    int          N,            // input: scalar - target selected count
    bool         use_prop_free_mode,   // input: scalar - early-exit hint for prop-free mode
    cudaStream_t exec_stream   // input: handle - CUDA execution stream
)
{
    // int device;
    // cudaGetDevice(&device);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);
    // ===== Step 1: Configure kernel launch parameters =====
    constexpr ll BLK_SIZE = 256;
    ll GRID_SIZE_DOMCOUNT = (dom_count + BLK_SIZE - 1) / BLK_SIZE;
    const int MAX_GRID_SIZE = (SM_COUNT * MAX_THREADS_PER_SM) / BLK_SIZE;
    // const int MAX_GRID_SIZE = prop.multiProcessorCount * BLK_SIZE;
    // const int MAX_GRID_SIZE = (SM_COUNT * 8);
    int GRID_SIZE = (int)min(GRID_SIZE_DOMCOUNT, (ll)MAX_GRID_SIZE);
    
    // ===== Step 2: Initialize loop variables =====
    int max_front    = 0;
    int iter_count   = 0;
    int is_new_front = 1;
    int h_asgn_count = 0;
    
    // ===== Step 3: Front propagation loop =====
    while (true) {
        // printf("iteration ... \n");
        CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        CUDA_CHECK(cudaMemcpy(&is_new_front, d_new_front, INT_SIZE, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_asgn_count,  d_asgn_cnt, INT_SIZE, cudaMemcpyDeviceToHost));
        
        if (!is_new_front || h_asgn_count >= N || use_prop_free_mode) {
            break;
        }
        
        CUDA_CHECK(cudaMemsetAsync(d_new_front, 0, INT_SIZE, exec_stream));
        fronts_propagation_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>
                            (d_dompairs, d_dominatee, d_fronts, d_new_front, d_asgn_cnt, dom_count, N_mix, max_front);
        CUDA_CHECK(cudaGetLastError());
        max_front++;
        iter_count++;
    }
    // std::cout << "NDS iteration count: " << iter_count << std::endl;
    return std::make_pair(max_front, iter_count);
}
// =============================================================================================== //
// ==================== MODULE: CSR-Based Front Propagation ====================
// =============================================================================================== //

/**
 * MODULE: CSR out-degree histogram kernel.
 * PURPOSE: Count outgoing edges per dominator from COO pairs.
 */
__global__ void histogram_dominator_outdeg_kernel(
    const int* __restrict__ d_dompairs,   // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int* __restrict__ d_out_degree,       // output: (N_mix,) - outgoing edge count per dominator
    ll dom_count_used,                    // input: scalar - valid COO pair count
    int N_mix)                            // input: scalar - population size
{
    ll tidx   = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    ll stride = (ll)blockDim.x * gridDim.x;
    for (ll s = tidx; s < dom_count_used; s += stride) {
        int2 pair = reinterpret_cast<const int2*>(d_dompairs)[s];
        atomicAdd(&d_out_degree[pair.x], 1);
    }
}

/**
 * MODULE: COO-to-CSR scatter kernel.
 * PURPOSE: Scatter COO pairs into CSR `col_idx` using atomic write offsets.
 */
__global__ void scatter_dompairs_to_csr_kernel(
    const int* __restrict__ d_dompairs,    // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int* __restrict__ d_col_idx,           // output: (dom_count_used,) - CSR adjacency list
    int* __restrict__ d_write_offset,      // buffer: (N_mix,) - per-row atomic write cursor
    ll dom_count_used,                     // input: scalar - valid COO pair count
    int N_mix)                             // input: scalar - population size
{
    ll tidx   = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    ll stride = (ll)blockDim.x * gridDim.x;
    for (ll s = tidx; s < dom_count_used; s += stride) {
        int2 pair = reinterpret_cast<const int2*>(d_dompairs)[s];
        int pos = atomicAdd(&d_write_offset[pair.x], 1);
        d_col_idx[pos] = pair.y;
    }
}

/**
 * MODULE: CSR sentinel kernel.
 * PURPOSE: Write `d_row_ptr[N_mix]` sentinel after exclusive-scan row offsets are built.
 */
__global__ void set_csr_sentinel_kernel(
    int* __restrict__ d_row_ptr, // update: (N_mix + 1,) - CSR row pointer
    int N_mix,                   // input: scalar - population size
    int val)                     // input: scalar - sentinel value (`dom_count_used`)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_row_ptr[N_mix] = val;
    }
}

/**
 * MODULE: COO-to-CSR builder.
 * PURPOSE: Build CSR adjacency (`row_ptr`, `col_idx`) from COO dominance pairs using pool scratch memory.
 */
static void build_csr_from_coo(
    const int* d_dompairs,     // input: (2 * PAIRS_CAP,) - COO dominance pairs
    ll dom_count_used,         // input: scalar - valid COO pair count
    int* d_row_ptr,            // output: (N_mix + 1,) - CSR row pointer
    int* d_col_idx,            // output: (dom_count_used,) - CSR adjacency list
    int N_mix,                 // input: scalar - population size
    cudaMemPool_t exec_pool,   // input: handle - CUDA memory pool
    cudaStream_t exec_stream)  // input: handle - CUDA execution stream
{
    constexpr int BLK = 256;
    const ll GRID_EDGE = min((dom_count_used + BLK - 1) / BLK, (ll)(SM_COUNT * 16));

    // 1. Allocate d_out_degree, memset 0
    int* d_out_degree = nullptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_out_degree, (size_t)N_mix * INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_out_degree, 0, (size_t)N_mix * INT_SIZE, exec_stream));

    // 2. Histogram
    histogram_dominator_outdeg_kernel<<<(int)GRID_EDGE, BLK, 0, exec_stream>>>(
        d_dompairs, d_out_degree, dom_count_used, N_mix);
    CUDA_CHECK(cudaGetLastError());

    // 3. ExclusiveSum: d_out_degree → d_row_ptr[0..N_mix-1]
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_out_degree, d_row_ptr, N_mix, exec_stream);
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_temp, temp_bytes, exec_pool, exec_stream));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_out_degree, d_row_ptr, N_mix, exec_stream);
    CUDA_CHECK(cudaFreeAsync(d_temp, exec_stream));

    // 4. Sentinel
    set_csr_sentinel_kernel<<<1, 1, 0, exec_stream>>>(d_row_ptr, N_mix, (int)dom_count_used);
    CUDA_CHECK(cudaGetLastError());

    // 5. Allocate d_write_offset, copy from d_row_ptr
    int* d_write_offset = nullptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_write_offset, (size_t)N_mix * INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_write_offset, d_row_ptr, (size_t)N_mix * INT_SIZE, cudaMemcpyDeviceToDevice, exec_stream));

    // 6. Scatter
    scatter_dompairs_to_csr_kernel<<<(int)GRID_EDGE, BLK, 0, exec_stream>>>(
        d_dompairs, d_col_idx, d_write_offset, dom_count_used, N_mix);
    CUDA_CHECK(cudaGetLastError());

    // 7. Free scratch
    CUDA_CHECK(cudaFreeAsync(d_write_offset, exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_out_degree, exec_stream));
}

/**
 * MODULE: CSR frontier expansion kernel.
 * PURPOSE: Expand one BFS frontier layer over CSR adjacency and build the next frontier queue.
 */
__global__ void expand_frontier_csr_kernel(
    const int* __restrict__ d_row_ptr,        // input: (N_mix + 1,) - CSR row pointer
    const int* __restrict__ d_col_idx,        // input: (dom_count_used,) - CSR adjacency list
    int* __restrict__ d_dominatee,            // update: (N_mix,) - remaining domination count
    int* __restrict__ d_fronts,               // update: (N_mix,) - front index per individual
    const int* __restrict__ d_frontier_in,    // input: (N_mix,) - current frontier queue
    int* __restrict__ d_frontier_out,         // output: (N_mix,) - next frontier queue
    int* __restrict__ d_frontier_out_size,    // update: (1,) - next frontier size
    int* __restrict__ d_frontier_work_idx,    // update: (1,) - dynamic work cursor for frontier_in nodes
    int frontier_in_size,                     // input: scalar - current frontier size
    int current_front)                        // input: scalar - current front index
{
    constexpr uint32_t FMASK = 0xFFFFFFFFu;
    int lane_idx = threadIdx.x % WS;
    int warp_idx_in_blk = threadIdx.x / WS;
    int warps_per_blk = (blockDim.x >= WS) ? (blockDim.x / WS) : 1;
    __shared__ int s_work_base;

    // Warp-cooperative frontier expansion with dynamic warp scheduling:
    // each CTA claims a batch of `warps_per_blk` frontier nodes, then each warp
    // processes at most one node from the batch. This reduces work-index atomics
    // versus per-warp claiming while keeping warp-cooperative edge traversal.
    while (true) {
        if (threadIdx.x == 0) {
            s_work_base = atomicAdd(d_frontier_work_idx, warps_per_blk);
        }
        __syncthreads();

        // Block-wide exit condition to avoid barrier divergence.
        if (s_work_base >= frontier_in_size) break;

        int f = s_work_base + warp_idx_in_blk;
        if (f >= frontier_in_size) continue;

        int node = d_frontier_in[f];
        int row_begin = d_row_ptr[node];
        int row_end   = d_row_ptr[node + 1];

        int degree = row_end - row_begin;
        int iters  = (degree + WS - 1) / WS;

        for (int it = 0; it < iters; ++it) {
            int e = row_begin + it * WS + lane_idx;
            bool valid = (e < row_end);
            bool should_enqueue = false;
            int dominatee_node = -1;

            if (valid) {
                dominatee_node = d_col_idx[e];
                int prev = atomicSub(&d_dominatee[dominatee_node], 1);
                should_enqueue = (prev == 1);
                if (should_enqueue) {
                    d_fronts[dominatee_node] = current_front + 1;
                }
            }

            uint32_t enq_mask = __ballot_sync(FMASK, should_enqueue);
            if (enq_mask) {
                int base_pos = 0;
                if (lane_idx == 0) {
                    base_pos = atomicAdd(d_frontier_out_size, __popc(enq_mask));
                }
                base_pos = __shfl_sync(FMASK, base_pos, 0);

                if (should_enqueue) {
                    uint32_t lane_mask = (1u << lane_idx);
                    int prefix = __popc(enq_mask & (lane_mask - 1u));
                    d_frontier_out[base_pos + prefix] = dominatee_node;
                }
            }
        }
    }
}

/**
 * MODULE: CSR frontier seed kernel.
 * PURPOSE: Initialize the CSR BFS frontier with all zero-domination individuals (front 0).
 */
__global__ void seed_frontier_kernel(
    const int* __restrict__ d_dominatee,   // input: (N_mix,) - domination count per individual
    int* __restrict__ d_fronts,            // update: (N_mix,) - front index per individual
    int* __restrict__ d_frontier,          // output: (N_mix,) - initial frontier queue
    int* __restrict__ d_frontier_size,     // update: (1,) - initial frontier size
    int N_mix)                             // input: scalar - population size
{
    constexpr uint32_t FMASK = 0xFFFFFFFFu;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_idx = threadIdx.x % WS;

    bool should_enqueue = (tid < N_mix) && (d_dominatee[tid] == 0);
    if (should_enqueue) {
        d_fronts[tid] = 0;
    }

    uint32_t enq_mask = __ballot_sync(FMASK, should_enqueue);
    if (enq_mask) {
        int base_pos = 0;
        if (lane_idx == 0) {
            base_pos = atomicAdd(d_frontier_size, __popc(enq_mask));
        }
        base_pos = __shfl_sync(FMASK, base_pos, 0);

        if (should_enqueue) {
            int prefix = __popc(enq_mask & ((1u << lane_idx) - 1u));
            d_frontier[base_pos + prefix] = tid;
        }
    }
}

/**
 * MODULE: CSR queue front propagation loop.
 * PURPOSE: Propagate fronts with a host-driven CSR BFS queue.
 */
static std::pair<int, int> propagate_fronts_csr_queue(
    int* d_dompairs,           // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int* d_dominatee,          // update: (N_mix,) - remaining domination count
    int* d_fronts,             // update: (N_mix,) - front index per individual
    int* d_asgn_cnt,           // update: (1,) - cumulative assignment count
    ll dom_count_used,         // input: scalar - valid COO pair count
    int N_mix,                 // input: scalar - population size
    int N,                     // input: scalar - target selected count
    cudaMemPool_t exec_pool,   // input: handle - CUDA memory pool
    cudaStream_t exec_stream)  // input: handle - CUDA execution stream
{
    // Edge case: no domination edges → all front-0, already initialized
    if (dom_count_used == 0) {
        return {0, 0};
    }
    const bool timing_enabled = front_prop_timing_enabled();

    // Timing breakdown (host + stream sections) to diagnose CSR front propagation regressions.
    double alloc_host_ms      = 0.0;
    double queue_loop_host_ms = 0.0;
    double copyback_host_ms   = 0.0;
    double free_host_ms       = 0.0;
    float  csr_build_gpu_ms   = 0.0f;
    float  seed_gpu_ms        = 0.0f;
    float  bfs_expand_gpu_ms  = 0.0f;
    int    bfs_kernel_launches = 0;
    int    max_frontier_seen   = 0;

    // ----- Allocate temporary buffers from exec_pool -----
    int* d_row_ptr        = nullptr;
    int* d_col_idx        = nullptr;
    int* d_frontier_ping  = nullptr;
    int* d_frontier_pong  = nullptr;
    int* d_frontier_size  = nullptr;
    int* d_frontier_work_idx = nullptr;

    auto try_alloc = [&]() -> cudaError_t {
        cudaError_t err;
        err = cudaMallocFromPoolAsync(&d_row_ptr,       (size_t)(N_mix + 1) * INT_SIZE, exec_pool, exec_stream);
        if (err != cudaSuccess) return err;
        err = cudaMallocFromPoolAsync(&d_col_idx,       (size_t)dom_count_used * INT_SIZE, exec_pool, exec_stream);
        if (err != cudaSuccess) return err;
        err = cudaMallocFromPoolAsync(&d_frontier_ping, (size_t)N_mix * INT_SIZE, exec_pool, exec_stream);
        if (err != cudaSuccess) return err;
        err = cudaMallocFromPoolAsync(&d_frontier_pong, (size_t)N_mix * INT_SIZE, exec_pool, exec_stream);
        if (err != cudaSuccess) return err;
        err = cudaMallocFromPoolAsync(&d_frontier_size, INT_SIZE, exec_pool, exec_stream);
        if (err != cudaSuccess) return err;
        err = cudaMallocFromPoolAsync(&d_frontier_work_idx, INT_SIZE, exec_pool, exec_stream);
        if (err != cudaSuccess) return err;
        return cudaSuccess;
    };

    auto free_all = [&]() {
        if (d_frontier_work_idx) cudaFreeAsync(d_frontier_work_idx, exec_stream);
        if (d_frontier_size)  cudaFreeAsync(d_frontier_size, exec_stream);
        if (d_frontier_pong)  cudaFreeAsync(d_frontier_pong, exec_stream);
        if (d_frontier_ping)  cudaFreeAsync(d_frontier_ping, exec_stream);
        if (d_col_idx)        cudaFreeAsync(d_col_idx, exec_stream);
        if (d_row_ptr)        cudaFreeAsync(d_row_ptr, exec_stream);
    };

    if (timing_enabled) {
        Timer alloc_timer;
        if (cudaError_t alloc_err = try_alloc(); alloc_err != cudaSuccess) {
            alloc_host_ms = alloc_timer.elapsed_ms();
            // No COO fallback here: fail fast on CSR queue allocation errors.
            free_all();
            CUDA_CHECK(alloc_err);
            return {0, 0};
        }
        alloc_host_ms = alloc_timer.elapsed_ms();
    } else {
        if (cudaError_t alloc_err = try_alloc(); alloc_err != cudaSuccess) {
            free_all();
            CUDA_CHECK(alloc_err);
            return {0, 0};
        }
    }

    CudaStageEventPair ev_build;
    CudaStageEventPair ev_seed;
    CudaStageEventPair ev_expand;
    if (timing_enabled) {
        create_stage_events(ev_build);
        create_stage_events(ev_seed);
        create_stage_events(ev_expand);
    }

    // ----- Build CSR -----
    if (timing_enabled) stage_record_start(ev_build, exec_stream);
    build_csr_from_coo(d_dompairs, dom_count_used, d_row_ptr, d_col_idx, N_mix, exec_pool, exec_stream);
    if (timing_enabled) stage_record_stop(ev_build, exec_stream);

    // ----- Seed frontier (front-0) -----
    CUDA_CHECK(cudaMemsetAsync(d_frontier_size, 0, INT_SIZE, exec_stream));

    constexpr int BLK = 256;
    int GRID_NODE = (N_mix + BLK - 1) / BLK;
    if (timing_enabled) stage_record_start(ev_seed, exec_stream);
    seed_frontier_kernel<<<GRID_NODE, BLK, 0, exec_stream>>>(
        d_dominatee, d_fronts, d_frontier_ping, d_frontier_size, N_mix);
    CUDA_CHECK(cudaGetLastError());
    if (timing_enabled) stage_record_stop(ev_seed, exec_stream);

    // ----- Host-loop BFS -----
    int max_front = 0;
    int h_frontier_size = 0;
    int h_total_assigned = 0;
    bool build_timed = false;
    bool seed_timed = false;
    bool expand_pending = false;
    auto queue_loop_body = [&]() {
        while (true) {
            CUDA_CHECK(cudaStreamSynchronize(exec_stream));

            if (timing_enabled) {
                if (!build_timed) {
                    csr_build_gpu_ms = stage_elapsed_ms(ev_build);
                    build_timed = true;
                }
                if (!seed_timed) {
                    seed_gpu_ms = stage_elapsed_ms(ev_seed);
                    seed_timed = true;
                }
                if (expand_pending) {
                    bfs_expand_gpu_ms += stage_elapsed_ms(ev_expand);
                    expand_pending = false;
                }
            }

            CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_frontier_size, INT_SIZE, cudaMemcpyDeviceToHost));
            h_total_assigned += h_frontier_size;

            if (h_frontier_size == 0 || h_total_assigned >= N) break;
            if (timing_enabled && h_frontier_size > max_frontier_seen) max_frontier_seen = h_frontier_size;

            // Reset next frontier counter
            CUDA_CHECK(cudaMemsetAsync(d_frontier_size, 0, INT_SIZE, exec_stream));
            CUDA_CHECK(cudaMemsetAsync(d_frontier_work_idx, 0, INT_SIZE, exec_stream));

            // Ping-pong buffers
            int* d_in  = (max_front & 1) ? d_frontier_pong : d_frontier_ping;
            int* d_out = (max_front & 1) ? d_frontier_ping : d_frontier_pong;

            constexpr int WARPS_PER_BLK = BLK / WS;
            int GRID_FRONTIER = min((h_frontier_size + WARPS_PER_BLK - 1) / WARPS_PER_BLK, SM_COUNT * 16);
            if (timing_enabled) stage_record_start(ev_expand, exec_stream);
            expand_frontier_csr_kernel<<<GRID_FRONTIER, BLK, 0, exec_stream>>>(
                d_row_ptr, d_col_idx, d_dominatee, d_fronts,
                d_in, d_out, d_frontier_size, d_frontier_work_idx,
                h_frontier_size, max_front);
            CUDA_CHECK(cudaGetLastError());
            if (timing_enabled) {
                stage_record_stop(ev_expand, exec_stream);
                expand_pending = true;
                bfs_kernel_launches++;
            }

            max_front++;
        }
    };

    if (timing_enabled) {
        Timer queue_loop_timer;
        queue_loop_body();
        queue_loop_host_ms = queue_loop_timer.elapsed_ms();
    } else {
        queue_loop_body();
    }

    // Update d_asgn_cnt on device
    if (timing_enabled) {
        Timer copyback_timer;
        CUDA_CHECK(cudaMemcpyAsync(d_asgn_cnt, &h_total_assigned, INT_SIZE, cudaMemcpyHostToDevice, exec_stream));
        copyback_host_ms = copyback_timer.elapsed_ms();
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_asgn_cnt, &h_total_assigned, INT_SIZE, cudaMemcpyHostToDevice, exec_stream));
    }

    // Free all temporaries
    if (timing_enabled) {
        Timer free_timer;
        free_all();
        free_host_ms = free_timer.elapsed_ms();
    } else {
        free_all();
    }

    if (timing_enabled) {
        destroy_stage_events(ev_expand);
        destroy_stage_events(ev_seed);
        destroy_stage_events(ev_build);

        std::cout << std::fixed << std::setprecision(3)
                  << "[TIMING][ndsort][front_prop][csr_queue] alloc_host=" << alloc_host_ms
                  << " ms, csr_build_gpu=" << csr_build_gpu_ms
                  << " ms, seed_gpu=" << seed_gpu_ms
                  << " ms, queue_loop_host=" << queue_loop_host_ms
                  << " ms, bfs_expand_gpu=" << bfs_expand_gpu_ms
                  << " ms, copyback_host=" << copyback_host_ms
                  << " ms, free_host=" << free_host_ms
                  << " ms, bfs_kernels=" << bfs_kernel_launches
                  << ", max_frontier=" << max_frontier_seen
                  << ", max_front=" << max_front
                  << std::endl;
    }

    return {max_front, max_front};
}

/**
 * MODULE: COO/CSR_QUEUE front-propagation path selector.
 * PURPOSE: Select and run CSR queue or COO_QUEUE propagation directly.
 * NOTES: `ASAP_NDSORT_FORCE_COO_QUEUE=1` forces COO_QUEUE; `ASAP_NDSORT_FORCE_CSR_QUEUE=1` forces CSR queue.
 */
static std::pair<int, int> propagate_fronts_csr(
    int* d_dompairs,           // input: (2 * PAIRS_CAP,) - COO dominance pairs
    int* d_dominatee,          // update: (N_mix,) - remaining domination count
    int* d_fronts,             // update: (N_mix,) - front index per individual
    int* d_new_front,          // update: (1,) - COO_QUEUE next-front discovery flag
    int* d_asgn_cnt,           // update: (1,) - cumulative assignment count
    ll dom_count_used,         // input: scalar - valid COO pair count
    int N_mix,                 // input: scalar - population size
    int N,                     // input: scalar - target selected count
    bool prefer_csr_queue,     // input: scalar - heuristic preference for CSR queue propagation
    cudaMemPool_t exec_pool,   // input: handle - CUDA memory pool
    cudaStream_t exec_stream)  // input: handle - CUDA execution stream
{
    const bool force_coo_queue = env_flag_enabled("ASAP_NDSORT_FORCE_COO_QUEUE");
    const bool force_csr_queue = env_flag_enabled("ASAP_NDSORT_FORCE_CSR_QUEUE");
    const bool timing_enabled  = front_prop_timing_enabled();
    const bool use_csr_queue = force_csr_queue || (!force_coo_queue && prefer_csr_queue);

    if (use_csr_queue) {
        if (timing_enabled) {
            if (force_csr_queue) {
                std::cout << "[TIMING][ndsort][front_prop] forcing csr_queue path "
                          << "(ASAP_NDSORT_FORCE_CSR_QUEUE=1)" << std::endl;
            } else {
                std::cout << "[TIMING][ndsort][front_prop] selecting csr_queue path" << std::endl;
            }
        }
        return propagate_fronts_csr_queue(d_dompairs, d_dominatee, d_fronts, d_asgn_cnt,
                                          dom_count_used, N_mix, N, exec_pool, exec_stream);
    }

    if (timing_enabled) {
        if (force_coo_queue) {
            std::cout << "[TIMING][ndsort][front_prop] forcing coo_queue path "
                      << "(ASAP_NDSORT_FORCE_COO_QUEUE=1)" << std::endl;
        } else {
            std::cout << "[TIMING][ndsort][front_prop] heuristic selects coo_queue path" << std::endl;
        }
    }

    int is_new_front = 1;
    CUDA_CHECK(cudaMemcpyAsync(d_new_front, &is_new_front, INT_SIZE, cudaMemcpyHostToDevice, exec_stream));
    initialize_front(d_dominatee, d_fronts, d_asgn_cnt, N_mix, exec_stream);
    return propagate_fronts_coo(d_dompairs, d_dominatee, d_fronts, d_new_front, d_asgn_cnt,
                                dom_count_used, N_mix, N, false, exec_stream);
}

/**
 * MODULE: Initial-generation non-dominated sorting.
 * PURPOSE: Run pair generation, dominatee counting, front-0 assignment, and front propagation for the first population.
 */
int run_ndsort_init(
    cudaStreamSync cuda_streams,    // input: struct - CUDA stream/event bundle
    float* d_cv,                    // input: (N,) - constraint violations
    float* d_fv,                    // input: (M, N) - objective values
    int* d_fronts,                  // output: (N,) - front index per individual
    int* d_dompairs_init,           // buffer: (2 * PAIRS_CAP_INIT,) - COO dominance pairs
    int* d_dominatee_init,          // buffer: (N,) - domination count per individual
    ll* d_dom_cnt,                  // buffer: (1,) - total pair count (64-bit)
    int* d_asgn_cnt,                // buffer: (1,) - cumulative assignment count
    int* d_new_front,               // buffer: (1,) - next-front discovery flag
    ll PAIRS_CAP_INIT,              // input: scalar - pair buffer capacity
    int M,                          // input: scalar - objective count
    int N,                          // input: scalar - population size
    cudaMemPool_t exec_pool         // input: handle - CUDA memory pool
)
{   
    // Start a new front-propagation routing epoch and seed it with the init-stage observation below.
    reset_front_prop_path_heuristic_state();

    cudaStream_t& exec_stream = cuda_streams.exec_stream;
    // Use async memset on exec_stream to maintain async execution
    CUDA_CHECK(cudaMemsetAsync(d_fronts,         -1,     N * INT_SIZE, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_dominatee_init,  0,     N * INT_SIZE, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_dom_cnt,         0,          LL_SIZE, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_asgn_cnt,        0,         INT_SIZE, exec_stream));
     
    int is_new_front = 1;
    CUDA_CHECK(cudaMemcpyAsync(d_new_front, &is_new_front, INT_SIZE, cudaMemcpyHostToDevice, exec_stream));

    // ===== Step 2: Compute dominance relationships =====
    compute_dompairs(d_cv, d_fv, d_dompairs_init, d_dom_cnt, N, M, PAIRS_CAP_INIT, exec_stream);

    // ===== Step 3: Count dominatees =====
    ll h_dom_count_init = 0;
    CUDA_CHECK(cudaStreamSynchronize(exec_stream));
    CUDA_CHECK(cudaMemcpy(&h_dom_count_init, d_dom_cnt, LL_SIZE, cudaMemcpyDeviceToHost));

    if (h_dom_count_init > PAIRS_CAP_INIT) {
        std::cout << "[WARNING] Init dominance pairs overflow: " << h_dom_count_init
                  << " > " << PAIRS_CAP_INIT << ". Results may be approximate." << std::endl;
    }
    const ll dom_count_used = min(h_dom_count_init, PAIRS_CAP_INIT);
    if (dom_count_used > 0) {
        constexpr int BLK_SIZE = 256;
        const int GRID_SIZE    = (dom_count_used + BLK_SIZE - 1) / BLK_SIZE;
        count_dominatee_from_dompairs_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(d_dompairs_init, d_dominatee_init, dom_count_used, N);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Zero dominance pairs is a valid state (all mutually non-dominated); avoid <<<0,...>>> launch.
        std::cout << "[INFO][ndsort_init] dom_count_used=0 (N=" << N
                  << ", M=" << M
                  << ", h_dom_count_init=" << h_dom_count_init
                  << "); skipping COO dominatee-count kernel." << std::endl;
    }

    // ===== Step 4+5: Front-0 assignment + front propagation (CSR_QUEUE / COO_QUEUE selector) =====
    auto [max_front, iter_count] = propagate_fronts_csr(
        d_dompairs_init, d_dominatee_init, d_fronts, d_new_front, d_asgn_cnt,
        dom_count_used, N, N, true, exec_pool, exec_stream);

    update_front_prop_path_heuristic_from_iter_count(iter_count);
    std::cout << "intial NDS iter_count is " << iter_count << std::endl;
    return max_front;
}
// ==================================================================================================================== //
/**
 * MODULE: Main non-dominated sorting routine.
 * PURPOSE: Run adaptive COO/CSR non-dominated sorting for the mixed population during iterative evolution.
 */
// ================================================================================================================================
int run_ndsort(
    float* d_mixcv,                 // input: (N_mix,) - constraint violations
    float* d_mixfv,                 // input: (M, N_mix) - objective values
    int* d_mixfronts,               // output: (N_mix,) - front index per individual
    int* d_dompairs,                // buffer: (2 * PAIRS_CAP,) - COO dominance pairs
    uint32_t* d_dominatee_mask,     // buffer: (N_mix, N_tiles) - CSR domination bitmask
    int* d_dominatee,               // buffer: (N_mix,) - domination count per individual
    ll* d_dom_cnt,                  // buffer: (1,) - total pair count (64-bit)
    int* d_asgn_cnt,                // buffer: (1,) - cumulative assignment count
    int* d_new_front,               // buffer: (1,) - next-front discovery flag
    ll PAIRS_CAP,                   // input: scalar - pair buffer capacity
    bool& use_prop_free_mode,       // update: scalar - toggle prop-free (dominatee-bitmask) mode
    int n_iter,                     // input: scalar - current iteration index
    int N_iter,                     // input: scalar - total iteration count
    int M,                          // input: scalar - objective count
    int N,                          // input: scalar - target selected count
    cudaStream_t exec_stream,       // input: handle - CUDA execution stream
    cudaMemPool_t exec_pool         // input: handle - CUDA memory pool
)
{   
    const int N_mix = 2 * N;
    // ===== Step 1: Configure switching mechanism parameters =====
    constexpr int HISTORY_WINDOW       = 5;
    constexpr float PROGRESS_THRESHOLD = 5.0f;
    const int threshold_iter = max(50, static_cast<int>((N_iter * PROGRESS_THRESHOLD) / 100.0f));
    static int iter_count_window[HISTORY_WINDOW] = {0};
    if (n_iter == 0) {
        memset(iter_count_window, 0, HISTORY_WINDOW * INT_SIZE);
    }
    apply_front_prop_path_hysteresis();

    // ===== Step 2: Initialize arrays (async on exec_stream) =====
    CUDA_CHECK(cudaMemsetAsync(d_mixfronts,    -1,     N_mix * INT_SIZE, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_dominatee,     0,     N_mix * INT_SIZE, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_asgn_cnt,      0,             INT_SIZE, exec_stream));

    int is_new_front = 1;
    CUDA_CHECK(cudaMemcpyAsync(d_new_front, &is_new_front, INT_SIZE, cudaMemcpyHostToDevice, exec_stream));

    // ===== Step 3: Compute dominance relationships (COO or CSR mode) =====
    ll  h_dom_count = 0;
    ll  dom_count_used = 0;
    if (!use_prop_free_mode) {
        compute_dompairs(d_mixcv, d_mixfv, d_dompairs, d_dom_cnt, N_mix, M, PAIRS_CAP, exec_stream);

        // Synchronize before D2H copy to ensure d_dom_cnt is ready
        CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        CUDA_CHECK(cudaMemcpy(&h_dom_count, d_dom_cnt, LL_SIZE, cudaMemcpyDeviceToHost));

        if (h_dom_count > PAIRS_CAP) {
            std::cout << "[WARNING] Dominance pairs overflow: " << h_dom_count
                      << " > " << PAIRS_CAP << ". Results may be approximate." << std::endl;
        }
        dom_count_used = min(h_dom_count, PAIRS_CAP);

        count_dominatee_from_dompairs(d_dompairs, d_dominatee, dom_count_used, N_mix, exec_stream);
    } else {
        // printf("branch CSR");
        compute_dominatee_bitmask(d_mixcv, d_mixfv, d_dominatee_mask, N_mix, M, exec_stream);
        count_dominatee_from_bitmask(d_dominatee_mask, d_dominatee, N_mix, exec_stream);
        h_dom_count = thrust::reduce(thrust::cuda::par.on(exec_stream), 
                                    d_dominatee, d_dominatee + N_mix,  0LL, thrust::plus<ll>());
    }
    
    // ===== Step 4+5+6: Front assignment + propagation =====
    int max_front = 0;
    int iter_count = 0;
    bool ran_front_propagation = false;
    const bool prefer_csr_queue = g_use_csr_queue_front_prop;
    const bool timing_enabled = front_prop_timing_enabled();

    if (timing_enabled) {
        const char* reason = "hold";
        if (g_prev_front_prop_iter_count >= FRONT_PROP_CSR_QUEUE_ACTIVATE_ITER) {
            reason = "iter_count>=30 -> CSR_QUEUE";
        } else if (g_prev_front_prop_iter_count >= 0 &&
                   g_prev_front_prop_iter_count <= FRONT_PROP_COO_QUEUE_ACTIVATE_ITER) {
            reason = "iter_count<=25 -> COO_QUEUE";
        } else if (g_prev_front_prop_iter_count < 0) {
            reason = "init default -> CSR_QUEUE";
        }
        std::cout << "[TIMING][ndsort][front_prop] heuristic prev_iter_count="
                  << g_prev_front_prop_iter_count
                  << " choose=" << (prefer_csr_queue ? "csr_queue" : "coo_queue")
                  << " (" << reason << ")" << std::endl;
    }

    if (!use_prop_free_mode) {
        // COO pair path: choose CSR_QUEUE or COO_QUEUE propagation.
        ran_front_propagation = true;
        time_function("propagate_fronts", [&]() {
            auto res = propagate_fronts_csr(
                d_dompairs, d_dominatee, d_mixfronts, d_new_front, d_asgn_cnt,
                dom_count_used, N_mix, N, prefer_csr_queue, exec_pool, exec_stream);
            max_front  = res.first;
            iter_count = res.second;
        }, false);

    } else {
        // Bitmask path: check if front-0 covers N
        int h_asgn_count = 0;
        initialize_front(d_dominatee, d_mixfronts, d_asgn_cnt, N_mix, exec_stream);
        CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        CUDA_CHECK(cudaMemcpy(&h_asgn_count, d_asgn_cnt, INT_SIZE, cudaMemcpyDeviceToHost));

        if (h_asgn_count >= N) {
            // Prop-free fast path: front-0 covers the target population.
            max_front = 0;
            iter_count = 0;
        } else {
            // Prop-free fallback: recompute COO pairs and run front propagation.
            std::cout << "prop-free mode fallback is triggered "
                      << "(iteration round = " << n_iter << ")" << std::endl;
            CUDA_CHECK(cudaMemsetAsync(d_mixfronts, -1, N_mix * INT_SIZE, exec_stream));
            CUDA_CHECK(cudaMemsetAsync(d_dominatee,  0, N_mix * INT_SIZE, exec_stream));
            CUDA_CHECK(cudaMemsetAsync(d_asgn_cnt,   0,       INT_SIZE, exec_stream));

            compute_dompairs(d_mixcv, d_mixfv, d_dompairs, d_dom_cnt, N_mix, M, PAIRS_CAP, exec_stream);

            CUDA_CHECK(cudaStreamSynchronize(exec_stream));
            CUDA_CHECK(cudaMemcpy(&h_dom_count, d_dom_cnt, LL_SIZE, cudaMemcpyDeviceToHost));
            dom_count_used = min(h_dom_count, PAIRS_CAP);

            count_dominatee_from_dompairs(d_dompairs, d_dominatee, dom_count_used, N_mix, exec_stream);

            ran_front_propagation = true;
            time_function("propagate_fronts", [&]() {
                auto res = propagate_fronts_csr(
                    d_dompairs, d_dominatee, d_mixfronts, d_new_front, d_asgn_cnt,
                    dom_count_used, N_mix, N, prefer_csr_queue, exec_pool, exec_stream);
                max_front  = res.first;
                iter_count = res.second;
            }, false);

            use_prop_free_mode = false;
        }
    }

    // ===== Step 7: Update iteration history and check for switching =====
    memmove(&iter_count_window[1], &iter_count_window[0], (HISTORY_WINDOW - 1) * INT_SIZE);
    iter_count_window[0] = iter_count;

    if (!use_prop_free_mode) {
        if (n_iter >= threshold_iter) {
            if (iter_count_window[0] == 0) {
                if (iter_count_window[HISTORY_WINDOW - 1] > 0) {
                    bool all_zeros = true;
                    #pragma unroll
                    for (int i = 1; i < HISTORY_WINDOW - 1; i++) {
                        if (iter_count_window[i] != 0) {
                            all_zeros = false;
                            break;
                        }
                    }
                    if (all_zeros) {
                        use_prop_free_mode = true;
                    }
                }
            }
        }
    }

    if (ran_front_propagation) {
        update_front_prop_path_heuristic_from_iter_count(iter_count);
    }
    
    return max_front;
}

/**
 * MODULE: Front-mask generation kernel.
 * PURPOSE: Split `d_fronts` into binary masks for prior fronts and the last selected front.
 */
__global__ void nds_frontmask_kernel(
    const int* __restrict__ d_fronts, // input: (N_mix,) - front index per individual
    int* __restrict__ d_prior_mask,   // output: (N_mix,) - mask for fronts < max_front
    int* __restrict__ d_last_mask,    // output: (N_mix,) - mask for front == max_front
    int N_mix,                        // input: scalar - population size
    int max_front                     // input: scalar - last selected front index
)
{
    // ===== Step 1: Calculate global thread index =====
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_mix) return;
    
    // ===== Step 2: Read front level and compute masks =====
    int front = d_fronts[tid];
    
    d_prior_mask[tid] = (front >= 0 && front < max_front) ? 1 : 0;
    d_last_mask[tid]  = (front == max_front) ? 1 : 0;
}

/**
 * MODULE: Thrust mask-index extraction.
 * PURPOSE: Gather indices of non-zero mask entries on the specified CUDA stream.
 */
void extract_mask_idx_thrust(
    const int* d_mask,         // input: (N_mix,) - binary selection mask
    int* d_mask_idx,           // output: (N_count,) - indices with non-zero mask
    int N_mix,                 // input: scalar - mask length
    int N_count,               // input: scalar - expected output length
    cudaStream_t exec_stream   // input: handle - CUDA execution stream
)
{
    // ===== Step 1: Wrap device pointers with Thrust iterators =====
    thrust::device_ptr<const int> bitmask_ptr(d_mask);
    thrust::device_ptr<int> out_ptr(d_mask_idx);
    
    // ===== Step 2: Generate sequence and copy selected indices =====
    thrust::counting_iterator<int> idx_first(0);
    thrust::counting_iterator<int> idx_last = idx_first + N_mix;

    // Execute on specified stream instead of default stream
    thrust::copy_if(
        thrust::cuda::par.on(exec_stream),  // execution policy with stream
        idx_first, 
        idx_last, 
        bitmask_ptr, 
        out_ptr, 
        cuda::std::identity{}
    );
    CUDA_CHECK(cudaGetLastError());
}

// ================================================================================== //
/**
 * MODULE: NDS mask-size computation.
 * PURPOSE: Build prior/last masks and count `N_prior`, `N_last`, `N_nds`, `N_rem` for selection.
 */
NumNDSData get_nds_mask_size(
    int* d_mixfronts,           // input: (N_mix,) - front index per individual
    int* d_prior_mask,          // output: (N_mix,) - mask for fronts < max_front
    int* d_last_mask,           // output: (N_mix,) - mask for front == max_front
    int max_front,              // input: scalar - last selected front index
    int N,                      // input: scalar - target selected count
    cudaStream_t exec_stream    // input: handle - CUDA execution stream
)
{   
    const int N_mix = 2 * N;
    // ===== Step 2: Configure and launch bitmask generation kernel =====
    constexpr int BLK_SIZE_NL = 256;
    const int GRID_SIZE_NL = (N_mix + BLK_SIZE_NL - 1) / BLK_SIZE_NL;
    
    nds_frontmask_kernel<<<GRID_SIZE_NL, BLK_SIZE_NL, 0, exec_stream>>>(
        d_mixfronts, d_prior_mask, d_last_mask, N_mix, max_front
    );
    CUDA_CHECK(cudaGetLastError());
    
    // ===== Step 3: Count selected individuals using Thrust (on exec_stream) =====
    thrust::device_ptr<int> prior_mask_ptr(d_prior_mask);
    thrust::device_ptr<int> last_mask_ptr(d_last_mask);
    
    NumNDSData num_ndsdata;  // update: struct - {N_nds, N_prior, N_last, N_rem} 
    // Execute reductions on specified stream
    num_ndsdata.N_prior = thrust::reduce(
        thrust::cuda::par.on(exec_stream),
        prior_mask_ptr, 
        prior_mask_ptr + N_mix, 
        0
    );
    CUDA_CHECK(cudaGetLastError());

    num_ndsdata.N_last = thrust::reduce(
        thrust::cuda::par.on(exec_stream),
        last_mask_ptr,  
        last_mask_ptr + N_mix, 
        0
    );
    CUDA_CHECK(cudaGetLastError());
    
    num_ndsdata.N_rem = N - num_ndsdata.N_prior;
    num_ndsdata.N_nds = num_ndsdata.N_prior + num_ndsdata.N_last;

    return num_ndsdata;
}
// ========================================================================================= //
/**
 * MODULE: NDS fitness gather kernel.
 * PURPOSE: Gather selected individuals from `d_mixfv` into compact `d_ndsfv` using index indirection.
 */
template<int M>
__global__ void ndsfit_extraction_kernel(
    const float* __restrict__ d_mixfv,    // input: (M, N_mix) - source objective values
    const int* __restrict__ d_nds_gidx,   // input: (N_nds,) - selected global indices
    float* __restrict__ d_ndsfv,          // output: (M, N_nds) - gathered objective values
    int N_nds,                            // input: scalar - selected count
    int N_mix                             // input: scalar - source population size
)
{
    const int glb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride  = blockDim.x * gridDim.x;
    
    for (int idx = glb_idx; idx < N_nds; idx += stride) 
    {
        const int src_idx = d_nds_gidx[idx];
        
        if (src_idx < N_mix) {
            // Copy M objectives with loop unrolling for coalesced writes
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                // Gather read (non-coalesced due to indirect indexing, this cannot be avoided)
                const float value = d_mixfv[m * N_mix + src_idx];
                // Coalesced write (consecutive indices within warp)
                d_ndsfv[m * N_nds + idx] = value;
            }
        }
    }
}
// ========================================================================================= //
/**
 * MODULE: NDS selection data extraction.
 * PURPOSE: Extract prior/last-front indices and gather their fitness rows for downstream selection.
 */
void extract_ndsfit_mskidx(
    float* d_mixfv,                     // input: (M, N_mix) - mixed-population objective values
    int* d_prior_mask,                  // input: (N_mix,) - mask for prior fronts
    int* d_last_mask,                   // input: (N_mix,) - mask for last front
    int* d_prior_gidx,                  // output: (N_prior,) - prior-front global indices
    int* d_last_gidx,                   // output: (N_last,) - last-front global indices
    int* d_nds_gidx,                    // output: (N_nds,) - concatenated global indices
    float* d_ndsfv,                     // output: (M, N_nds) - gathered objective values
    const NumNDSData& num_ndsdata,      // input: struct - {N_nds, N_prior, N_last, N_rem}
    int M,                              // input: scalar - objective count
    int N,                              // input: scalar - target selected count
    cudaStream_t exec_stream            // input: handle - CUDA execution stream
)
{   
    const int N_mix = 2 * N;
    // ===== Step 1: Extract indices for prior and last fronts =====
    extract_mask_idx_thrust(d_prior_mask, d_prior_gidx, N_mix, num_ndsdata.N_prior, exec_stream);
    extract_mask_idx_thrust(d_last_mask,  d_last_gidx,  N_mix, num_ndsdata.N_last,  exec_stream);

    // ===== Step 2: Combine indices into single array (async on exec_stream) =====
    CUDA_CHECK(cudaMemcpyAsync(
        d_nds_gidx, 
        d_prior_gidx, 
        num_ndsdata.N_prior * INT_SIZE, 
        cudaMemcpyDeviceToDevice, 
        exec_stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        d_nds_gidx + num_ndsdata.N_prior, 
        d_last_gidx, 
        num_ndsdata.N_last * INT_SIZE, 
        cudaMemcpyDeviceToDevice, 
        exec_stream
    ));
    
    // ===== Step 3: Configure and launch fitness extraction kernel =====
    constexpr int BLK_SIZE = 256; 
    const int GRID_SIZE = (num_ndsdata.N_nds + BLK_SIZE - 1) / BLK_SIZE;
    
    // Template dispatch via switch-case (M ∈ [1, 8])
    #define LAUNCH_NDSFIT_EXTRACTION_KERNEL(M_VAL) \
        ndsfit_extraction_kernel<M_VAL><<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>( \
            d_mixfv, d_nds_gidx, d_ndsfv, num_ndsdata.N_nds, N_mix)
    
    switch (M) {
        case 1: LAUNCH_NDSFIT_EXTRACTION_KERNEL(1); break;
        case 2: LAUNCH_NDSFIT_EXTRACTION_KERNEL(2); break;
        case 3: LAUNCH_NDSFIT_EXTRACTION_KERNEL(3); break;
        case 4: LAUNCH_NDSFIT_EXTRACTION_KERNEL(4); break;
        case 5: LAUNCH_NDSFIT_EXTRACTION_KERNEL(5); break;
        case 6: LAUNCH_NDSFIT_EXTRACTION_KERNEL(6); break;
        case 7: LAUNCH_NDSFIT_EXTRACTION_KERNEL(7); break;
        case 8: LAUNCH_NDSFIT_EXTRACTION_KERNEL(8); break;
        default:
            std::cout << "ERROR (extract_ndsfit_mskidx): Unsupported objective count M = " << M
                      << " (max supported: 8)" << std::endl;
            return;
    }
    #undef LAUNCH_NDSFIT_EXTRACTION_KERNEL
    
    CUDA_CHECK(cudaGetLastError());
}
