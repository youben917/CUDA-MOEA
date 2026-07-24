#include <math_constants.h>

#include "cuda_moea/selection/rvea/kernels/reference_partition.cuh"

#include "cuda_moea/core/cuda/cuda_globals.cuh"

// ======================================================================================== //
// KERNEL: Partition Phase 1 - Max-Cosine Computation and Partial Reduction
// ======================================================================================== //
// PURPOSE:
//   Compute cosine similarity between translated fitness vectors and L2-normalized
//   reference points to assign each individual to its closest (most aligned) reference vector:
//     cos(θ) = (f' · w) / ||f'||    (since ||w|| = 1)
//
// ALGORITHM:
//   1. Cooperative data loading into shared memory with perfect coalescing:
//      - Load d_translated_fv[(m, fit_idx)] → reg_x[m]
//      - Load d_nrps[(m, rv_idx)] → s_nrps[m][tidx]
//   2. Compute ||f'||² and cache inv_norm_f = rsqrt(||f'||²) per fitness vector
//   3. Compute dot products using FMA tree: f' · w
//   4. Compute cosine similarity: cos(θ) = dot × inv_norm_f
//   5. Update best (max_cos, argmax_idx) with deterministic tie-breaking
//   6. Write partial results to global memory for Phase 2 reduction
//
// DETERMINISTIC TIE-BREAKING:
//   When multiple reference vectors have equal cosine values, the smaller index wins.
//   This ensures reproducible results across identical runs.
//
// TEMPLATE PARAMETERS:
//   WPB:    Warps per block (default: 8)
//   M:      Number of objectives (compile-time constant for loop unrolling)
//   UNROLL: Inner loop unroll factor (default: 4)
//
// PARAMETERS:
//   INPUT:
//     d_translated_fv: (M, pop_size) const float* - z_min-translated fitness vectors (transposed)
//                      Memory Layout: Column-major, element access: d_translated_fv[m * pop_size + idx]
//     d_nrps: (M, refvec_size) const float* - L2-normalized reference points (transposed)
//             Memory Layout: Column-major, element access: d_nrps[m * refvec_size + idx]
//             Guarantee: ||d_nrps[:, idx]|| = 1
//     pop_size:    int - Number of individuals (fitness vectors)
//     refvec_size: int - Number of reference vectors
//   OUTPUT:
//     d_part_blkx_max:  (N_btr, pop_size) float* - Partial maximum cosine values
//     d_part_blkx_amax: (N_btr, pop_size) int* - Partial argmax reference vector indices
// ========================================================================================================================================== //
template<int WPB = 8, int M = 3, int UNROLL = 4>
__global__ void partition_phase1_kernel(
    const float* __restrict__ d_translated_fv,        // input:  (M, pop_size)      - z_min-translated fitness values
    const float* __restrict__ d_nrps,                 // input:  (M, refvec_size)   - L2-normalized reference points
    float* __restrict__ d_part_blkx_max,              // output: (N_btr, pop_size)  - partial max cosine
    int*   __restrict__ d_part_blkx_amax,             // output: (N_btr, pop_size)  - partial argmax indices
    int pop_size,                                     // input:  scalar             - number of individuals
    int refvec_size                                   // input:  scalar             - number of reference vectors
)
{
    constexpr int BS = WPB * WS;

    const int  tidx      = threadIdx.x;
    const int  fit_idx   = blockIdx.x * BS + tidx;
    const bool valid_fit = (fit_idx < pop_size);

    __shared__ float s_nrps[M][BS];

    // ====================================================================
    // Merged loop: Load fitness + compute norm + load reference vectors
    // Combines operations to improve instruction scheduling
    // ====================================================================
    float reg_x[M];
    float sqr_fit = 0.f;

    const int rv_start = blockIdx.y * BS;
    const int rv_end   = min(rv_start + BS, refvec_size);
    const int rv_idx   = rv_start + tidx;

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        // Load fitness and accumulate norm (interleaved for better scheduling)
        float val = valid_fit ? d_translated_fv[m * pop_size + fit_idx] : 0.f;
        reg_x[m] = val;
        sqr_fit = fmaf(val, val, sqr_fit);

        // Cooperative load of reference points
        s_nrps[m][tidx] = (rv_idx < refvec_size) ? d_nrps[m * refvec_size + rv_idx] : 0.f;
    }

    const float inv_norm_fit = rsqrtf(sqr_fit + 1e-12f);
    __syncthreads();

    float my_max = -CUDART_INF_F;
    int my_amax  = -1;

    if (valid_fit) {
        const int num_rvs_in_tile = rv_end - rv_start;
        const int num_rvs_aligned = (num_rvs_in_tile / UNROLL) * UNROLL;

        // ================================================================
        // Main loop: Process UNROLL reference vectors per iteration
        // Increases arithmetic intensity and enables better ILP
        // ================================================================
        for (int r = 0; r < num_rvs_aligned; r += UNROLL) {
            float dot[UNROLL];
            float cos_theta[UNROLL];

            // Prefetch: compute all dot products first
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                dot[u] = 0.f;
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    dot[u] = fmaf(reg_x[m], s_nrps[m][r + u], dot[u]);
                }
            }

            // Compute all cosines (||w|| = 1, so cos = dot / ||f'||)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                cos_theta[u] = dot[u] * inv_norm_fit;
            }

            // Update maximum (sequential to maintain determinism)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int glb_rv_idx = rv_start + r + u;
                if (cos_theta[u] > my_max ||
                    (cos_theta[u] == my_max && glb_rv_idx < my_amax)) {
                    my_max  = cos_theta[u];
                    my_amax = glb_rv_idx;
                }
            }
        }

        // ================================================================
        // Tail loop: Handle remaining reference vectors
        // ================================================================
        for (int r = num_rvs_aligned; r < num_rvs_in_tile; ++r) {
            const int glb_rv_idx = rv_start + r;

            float dot = 0.f;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                dot = fmaf(reg_x[m], s_nrps[m][r], dot);
            }

            const float cos_theta = dot * inv_norm_fit;

            if (cos_theta > my_max ||
                (cos_theta == my_max && glb_rv_idx < my_amax)) {
                my_max  = cos_theta;
                my_amax = glb_rv_idx;
            }
        }

        const long long out_idx          = (long long)blockIdx.y * (long long)pop_size + (long long)fit_idx;
        d_part_blkx_max[out_idx]         = my_max;
        d_part_blkx_amax[out_idx]        = my_amax;
    }
}

// ================================================================================================= //
// LAUNCHER: Partition Phase 1 Kernel Launcher (Template Specialization)
// ================================================================================================= //
template<int M>
void launch_partition_phase1_kernel_with_M(
    const float* d_translated_fv,        // input:  (M, pop_size) - translated fitness
    const float* d_nrps,                 // input:  (M, refvec_size) - L2-normalized reference points
    float* d_part_blkx_max,              // buffer: (N_btr, pop_size) - partial max cosine
    int*   d_part_blkx_amax,             // buffer: (N_btr, pop_size) - partial argmax
    int    pop_size,                     // input:  scalar - number of individuals
    int    refvec_size,                  // input:  scalar - number of reference vectors
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    constexpr int WPB_PART = 8;
    constexpr int BLK_SIZE = WPB_PART * WS;                    // 256 threads
    const int N_btf = (pop_size + BLK_SIZE - 1) / BLK_SIZE;    // blocks in X (fitness)
    const int N_btr = (refvec_size + BLK_SIZE - 1) / BLK_SIZE; // blocks in Y (refs)
    dim3 grid_dim(N_btf, N_btr);

    partition_phase1_kernel<WPB_PART, M><<<grid_dim, BLK_SIZE, 0, exec_stream>>>
                    (d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size);
    CUDA_CHECK(cudaGetLastError());
}

// ================================================================================================= //
// LAUNCHER: Partition Phase 1 Dynamic Dispatcher
// ================================================================================================= //
void launch_partition_phase1_kernel(
    const float* d_translated_fv,        // input:  (M, pop_size) - translated fitness
    const float* d_nrps,                 // input:  (M, refvec_size) - L2-normalized reference points
    float* d_part_blkx_max,              // output: (N_btr, pop_size) - partial max cosine
    int*   d_part_blkx_amax,             // output: (N_btr, pop_size) - partial argmax
    int    pop_size,                     // input:  scalar - number of individuals
    int    refvec_size,                  // input:  scalar - number of reference vectors
    int    M,                            // input:  scalar - number of objectives
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
){
    switch (M) {
        case 1: launch_partition_phase1_kernel_with_M<1>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 2: launch_partition_phase1_kernel_with_M<2>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 3: launch_partition_phase1_kernel_with_M<3>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 4: launch_partition_phase1_kernel_with_M<4>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 5: launch_partition_phase1_kernel_with_M<5>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 6: launch_partition_phase1_kernel_with_M<6>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 7: launch_partition_phase1_kernel_with_M<7>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        case 8: launch_partition_phase1_kernel_with_M<8>(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax, pop_size, refvec_size, exec_stream); break;
        default:
            printf("ERROR (Partition): Unsupported objective count M = %d (max supported: 8)\n", M);
            return;
    }
}

// ======================================================================================================================= //
// KERNEL: Partition Phase 2 - Final Reduction Across Y-Blocks
// ======================================================================================================================= //
// PURPOSE:
//   Reduce partial results from Phase 1 across all Y-dimension blocks to produce
//   final partition angles and reference vector indices for each individual.
//
// ALGORITHM:
//   For each individual (fitness vector):
//     1. Sequential scan: Iterate through all N_btr partial results from Phase 1
//     2. Global maximum tracking: Maintain running maximum cosine and argmax index
//     3. Tie-breaking: Apply deterministic rule (smaller index wins)
//     4. Output conversion: Convert cos to angle via acos for downstream APD computation
//
// NOTE ON OUTPUT:
//   We output the ANGLE (in radians) rather than cosine value.
//   This is required by the APD formula: APD = (1 + M * theta * t/T) * ||f'||
//   where theta is the angle between the individual and its assigned reference vector.
//
// PARAMETERS:
//   INPUT:
//     d_part_blkx_max:  (N_btr, pop_size) const float* - Partial max cosine from Phase 1
//     d_part_blkx_amax: (N_btr, pop_size) const int* - Partial argmax indices from Phase 1
//     pop_size: int - Number of individuals
//     N_btr:    int - Number of Y-blocks
//   OUTPUT:
//     d_theta:      (pop_size,) float* - Final angle values (radians, smaller = better alignment)
//     d_refvec_idx: (pop_size,) int* - Assigned reference vector indices
// ======================================================================================================================= //
__global__ void partition_phase2_kernel(
    const float* __restrict__ d_part_blkx_max,       // input:  (N_btr, pop_size) - partial max cosine
    const int*   __restrict__ d_part_blkx_amax,      // input:  (N_btr, pop_size) - partial argmax indices
    float* __restrict__ d_theta,                      // output: (pop_size,) - final angle values (radians)
    int*   __restrict__ d_refvec_idx,                 // output: (pop_size,) - assigned reference vector indices
    int pop_size,
    int N_btr
) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= pop_size) return;

    // Initialize for maximum finding (max cosine = min angle)
    float local_max  = -CUDART_INF_F;
    int   local_amax = -1;

    // ============================================================================
    // STAGE 1: Scan all Y-blocks to find global maximum cosine
    // ============================================================================
    for (int blk_y_idx = 0; blk_y_idx < N_btr; blk_y_idx++) {
        long long offset = (long long)blk_y_idx * (long long)pop_size + (long long)tidx;

        float cos_val = d_part_blkx_max[offset];
        int   aidx    = d_part_blkx_amax[offset];

        // ========================================================================
        // Update local maximum with deterministic tie-breaking
        // Tie-breaking: equal cosine → smaller index wins
        // ========================================================================
        if (cos_val > local_max || (cos_val == local_max && aidx >= 0 && aidx < local_amax)) {
            local_max  = cos_val;
            local_amax = aidx;
        }
    }

    // ============================================================================
    // STAGE 2: Convert to angle and write final results
    // ============================================================================
    // Convert cosine to angle (radians) for downstream APD computation
    // Clamp to [0, 1] to handle numerical precision issues before acos
    if (local_max > -CUDART_INF_F) {
        float clamped_cos = fminf(fmaxf(local_max, 0.f), 1.f);
        d_theta[tidx] = acosf(clamped_cos);  // Output angle in radians
    } else {
        d_theta[tidx] = CUDART_INF_F;  // Invalid: set to infinity
    }

    d_refvec_idx[tidx] = local_amax;
}

// ================================================================================================= //
// LAUNCHER: Partition Phase 2 Kernel Launcher
// ================================================================================================= //
void launch_partition_phase2_kernel(
    float* d_part_blkx_max,             // input:  (N_btr, pop_size) - partial max cosine
    int*   d_part_blkx_amax,            // input:  (N_btr, pop_size) - partial argmax
    float* d_theta,                     // output: (pop_size,) - final angle values
    int*   d_refvec_idx,                // output: (pop_size,) - assigned ref vector indices
    int    pop_size,
    int    refvec_size,
    cudaStream_t exec_stream
){
    constexpr int WPB_PART = 8;
    constexpr int BLK_SIZE = WPB_PART * WS;
    const int N_btr        = (refvec_size + BLK_SIZE - 1) / BLK_SIZE;
    const int N_btf        = (pop_size + BLK_SIZE - 1) / BLK_SIZE;

    partition_phase2_kernel<<<N_btf, BLK_SIZE, 0, exec_stream>>>
                        (d_part_blkx_max, d_part_blkx_amax, d_theta, d_refvec_idx, pop_size, N_btr);
    CUDA_CHECK(cudaGetLastError());
}

// ================================================================================================= //
// KERNEL: Mark Dead Parent Slots with Invalid Ref-Index
// ================================================================================================= //
// PURPOSE:
//   Overwrite parent dead slots [N_active, N_parent) with invalid reference index refvec_size.
//   This quarantines dead-slot sentinels from polluting downstream niche segmentation.
// ================================================================================================= //
__global__ void refidx_deadslot_mark_kernel(
    int* d_refvec_idx,                             // update: (pop_size,) - assigned reference-vector indices
    int  N_active,                                 // input:  scalar      - first dead parent slot index
    int  N_parent,                                 // input:  scalar      - parent section end index
    int  invalid_refidx                            // input:  scalar      - invalid reference index marker
)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + N_active;
         idx < N_parent;
         idx += gridDim.x * blockDim.x) {
        d_refvec_idx[idx] = invalid_refidx;
    }
}

// ================================================================================================= //
// LAUNCHER: Dead-Slot Ref-Index Mark Kernel (Template Specialization)
// ================================================================================================= //
template<int BLK_SIZE>
void launch_refidx_deadslot_mark_kernel_with_cfg(
    int*         d_refvec_idx,                     // update: (pop_size,) - reference-vector indices to mark
    int          N_active,                         // input:  scalar      - first dead parent slot index
    int          N_parent,                         // input:  scalar      - parent section end index
    int          invalid_refidx,                   // input:  scalar      - invalid reference index marker
    cudaStream_t exec_stream                       // input:  scalar      - CUDA execution stream
)
{
    const int N_dead = N_parent - N_active;
    if (N_dead <= 0) return;

    const int GRID_SIZE = (N_dead + BLK_SIZE - 1) / BLK_SIZE;
    refidx_deadslot_mark_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_refvec_idx, N_active, N_parent, invalid_refidx);
    CUDA_CHECK(cudaGetLastError());
}

// ==============================================================================================
// MAIN FUNCTION: Execute Complete Max-Cosine Partition Pipeline
// ==============================================================================================
// PURPOSE:
//   Orchestrate the complete max-cosine partition process between translated
//   fitness vectors and L2-normalized reference points. Assigns each individual to
//   the reference vector with which it has the smallest angle (largest cosine).
//
// ALGORITHM OVERVIEW:
//   Given z_min-translated fitness vectors F' (pop_size × M) and
//   L2-normalized reference points W (refvec_size × M), where ||W[j]|| = 1:
//
//   Stage 1 - Phase 1 Kernel: Parallel Cosine Computation & Partial Reduction
//     For each (i, j) pair where i ∈ [0, pop_size) and j ∈ [0, refvec_size):
//       1. Compute ||F'[i]|| for each fitness vector
//       2. Compute dot product: F'[i] · W[j]
//       3. Compute cosine: cos(θ) = dot / ||F'[i]||    (since ||W[j]|| = 1)
//       4. Perform local argmax reduction within Y-blocks
//       5. Output partial results: (max_cos, argmax_idx) per Y-block
//
//   Stage 2 - Phase 2 Kernel: Global Reduction
//     For each individual i ∈ [0, pop_size):
//       1. Collect all N_btr partial results from Phase 1
//       2. Find global maximum: argmax_j(cos(θ[i,j])) = argmin_j(θ[i,j])
//       3. Convert to angle: θ = acos(max_cos)
//       4. Output final results: (theta[i], refvec_idx[i])
//
//   Stage 3 - Dead-Slot Quarantine
//     Overwrite dead parent slots [N_active, N_parent) in d_refvec_idx with
//     invalid marker refvec_size, so sentinels do not enter valid niches.
//
// PARAMETERS:
//   INPUT:
//     d_translated_fv: (M, pop_size) const float* - z_min-translated fitness vectors (transposed)
//     d_nrps:          (M, refvec_size) const float* - L2-normalized reference points (transposed)
//                      Guarantee: ||d_nrps[:, idx]|| = 1
//     refvec_size: int - Number of reference vectors
//     M:           int - Number of objectives
//     pop_size:    int - Total number of individuals (N_mix)
//     N_active:    int - Number of alive parent individuals
//     N_parent:    int - Parent section end index [0, N_parent)
//
//   INTERNAL BUFFERS:
//     d_part_blkx_max:  (N_btr, pop_size) float* - Partial maximum cosine values
//     d_part_blkx_amax: (N_btr, pop_size) int* - Partial argmax indices
//
//   OUTPUT:
//     d_theta:      (pop_size,) float* - Final partition angles (radians)
//                   Interpretation: Smaller value = smaller angle = better alignment
//     d_refvec_idx: (pop_size,) int* - Assigned reference vector indices
//                   Index of most aligned reference vector for each fitness vector
// ==============================================================================================
void rvea::execute_partition(
    const float* d_translated_fv,           // input:  (M, pop_size) - z_min-translated fitness
    const float* d_nrps,                    // input:  (M, refvec_size) - L2-normalized reference points 
    float*       d_part_blkx_max,           // buffer: (N_btr, pop_size) - partial max buffer
    int*         d_part_blkx_amax,          // buffer: (N_btr, pop_size) - partial argmax buffer
    float*       d_theta,                   // output: (pop_size,) - final angle values
    int*         d_refvec_idx,              // output: (pop_size,) - assigned ref vector indices
    int          refvec_size,               // input:  scalar - number of reference vectors
    int          M,                         // input:  scalar - number of objectives
    int          pop_size,                  // input:  scalar - total number of individuals (N_mix)
    int          N_active,                  // input:  scalar - number of alive parent individuals
    int          N_parent,                  // input:  scalar - parent section end index [0, N_parent)
    cudaStream_t exec_stream                // input:  scalar - CUDA execution stream
)
{
    // ============================================================================
    // STAGE 1: Partition Phase 1 - Cosine Computation & Partial Reduction
    // ============================================================================
    launch_partition_phase1_kernel(d_translated_fv, d_nrps, d_part_blkx_max, d_part_blkx_amax,
                                   pop_size, refvec_size, M, exec_stream);

    // ============================================================================
    // STAGE 2: Partition Phase 2 - Final Reduction (outputs angle in radians)
    // ============================================================================
    launch_partition_phase2_kernel(d_part_blkx_max, d_part_blkx_amax,
                                   d_theta, d_refvec_idx, pop_size, refvec_size, exec_stream);

    // ============================================================================
    // STAGE 3: Quarantine dead parent slots with invalid reference index marker
    // ============================================================================
    launch_refidx_deadslot_mark_kernel_with_cfg<256>(
        d_refvec_idx, N_active, N_parent, refvec_size, exec_stream);
}
