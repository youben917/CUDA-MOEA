#include <math_constants.h>
#include <cmath> 

#include "cuda_moea/core/cuda/time_utils.h"
#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"
#include "cuda_moea/core/cuda/cuda_transpose.cuh"
#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"
#include "cuda_moea/core/cuda/cuda_atomops.cuh"
#include "cuda_moea/core/cuda/cuda_vecops.cuh"

// ========================================================================================================== //
// ======================================================================================== //
// KERNEL: Association Phase 1 - Fused Distance Computation and Partial Reduction
// ======================================================================================== //
// PURPOSE:
//   Compute perpendicular distances between normalized fitness vectors and reference
//   points using the numerically stable residual formula:
//     dist² = ||nfit - (nfit · nrp) × nrp||² = Σ(nfit[m] - dot × nrp[m])²
//   This avoids catastrophic cancellation that occurs with: dist² = ||nfit||² - (nfit · nrp)²
//   Perform partial reduction across Y-dimension blocks to find local minima.
//
// ALGORITHM:
//   1. Cooperative data loading into shared memory with perfect coalescing:
//      - Load d_ndsfv[(m, nfit_idx)] → s_nfit[tile][f][m]
//      - Load d_nrps[(m, nrps_idx)] → s_nrps[tile][r][m]
//   2. Prefetch fitness vectors from SMEM to registers to hide latency
//   3. Compute dot products using FMA tree: nfit · nrp
//   4. Compute squared distances using residual formula (numerically stable):
//      dist² = Σ(nfit[m] - dot × nrp[m])²
//   5. Update best (min_dist², argmin_idx) with deterministic tie-breaking
//   6. Write partial results to global memory for Phase 2 reduction
//
// NUMERICAL STABILITY:
//   The residual formula avoids subtracting two nearly-equal large values when
//   nfit and nrp are nearly parallel. Instead, it computes small residuals first,
//   then squares and accumulates them - preserving precision throughout.
//
// DETERMINISTIC TIE-BREAKING:
//   When multiple reference points have equal distance², the smaller index wins.
//   This ensures reproducible results across identical runs.
//
// MEMORY HIERARCHY OPTIMIZATION:
//   Global Memory → L1 Cache → Shared Memory → Registers → ALU
//   - Perfect coalescing at Global→L1 stage (column-major transposed layout)
//   - Zero bank conflicts in SMEM access pattern
//   - Register-level computation for maximum throughput
//
// TEMPLATE PARAMETERS:
//   TF:  Tile size along N_nds dimension per warp (default: 8)
//        Each warp processes TF fitness vectors simultaneously
//   TR:  Tile size along N_ref dimension per warp (default: 32)
//        Each warp processes TR reference points per iteration
//   WPB: Warps per block (default: 8)
//        Total parallelism per thread block
//   M:   Number of objectives (compile-time constant for loop unrolling)
//        Enables aggressive compiler optimization
//
// BLOCK DECOMPOSITION:
//   Grid:  (N_btf, N_btr) 
//          N_btf = ceil(N_nds / BTF), N_btr = ceil(N_ref / BTR)
//   Block: Processes BTF × BTR tile 
//          BTF = WPB * TF (e.g., 8*8=64), BTR = WPB * TR (e.g., 8*32=256)
//   Warp:  Processes TF × BTR sub-tile
//          Each warp owns TF fitness vectors and iterates over all reference points
//
// PARAMETERS:
//   INPUT:
//     d_ndsfv: (M, N_nds) const float* - Normalized fitness vectors (transposed)
//                   Memory Layout: Column-major, element access: d_ndsfv[m * N_nds + idx]
//                   Ensures coalesced access when threads load consecutive individuals
//     d_nrps: (M, N_ref) const float* - Normalized reference points (transposed)
//                   Memory Layout: Column-major, element access: d_nrps[m * N_ref + idx]
//                   Enables coalesced loads for reference point data
//     N_nds:        int - Number of non-dominated individuals (fitness vectors)
//     N_ref:        int - Number of reference points
//   OUTPUT:
//     d_assoc_blkx_min:   (N_btr, N_nds) float* - Partial minimum squared distances
//                   Memory Layout: d_assoc_blkx_min[blockIdx.y * N_nds + nfit_idx]
//                   Each Y-block writes its local minima for Phase 2 reduction
//     d_assoc_blkx_amin:  (N_btr, N_nds) int* - Partial argmin reference point indices
//                   Memory Layout: d_assoc_blkx_amin[blockIdx.y * N_nds + nfit_idx]
//                   Stores corresponding reference point index for each minimum
// ======================================================================================================================================= //
template<int WPB = 8, int M = 3>
__global__ void association_phase1_kernel(
    const float* __restrict__ d_ndsfv,        // input: (M, N_nds), fitness normalized by intercept (note that could be larger than 1)
    const float* __restrict__ d_nrps,         // input: (M, N_ref), normalized reference points 
    float* __restrict__ d_assoc_blkx_min,     // output: min    of each block indexed by blockIdx.x
    int*   __restrict__ d_assoc_blkx_amin,    // output: argmin of each block indexed by blockIdx.x
    int N_nds,                                // input: scalar 
    int N_ref                                 // input: scalar 
)
{
    // ====================================================================
    // Constants - Keep similar tile sizes to original
    // ====================================================================
    constexpr int BS = WPB * WS;   // Match block size for better loading
    // ====================================================================
    // Thread indexing
    // ====================================================================
    const int  tidx       = threadIdx.x;
    const int  nfit_idx   = blockIdx.x * BS + tidx;
    const bool valid_nfit = (nfit_idx < N_nds);
    
    // ====================================================================
    // Shared memory for reference points
    // ====================================================================
    __shared__ float s_nrps[M][BS];
    
    // ====================================================================
    // Prepare for parallel loading
    // ====================================================================
    float reg_x[M];
    
    // Initialize per-thread reduction variables - SCALARS ONLY!
    float my_min = CUDART_INF_F;
    int my_amin  = -1;
    
    // Prepare reference tile boundaries
    const int ref_start = blockIdx.y * BS;
    const int ref_end   = min(ref_start + BS, N_ref);
    const int ref_idx   = ref_start + tidx;
    
    // ====================================================================
    // Merged loop: Load fitness vectors AND reference points cooperatively
    // All threads participate in both operations!
    // NOTE: Removed sqr_nfit accumulation - no longer needed with residual formula
    // ====================================================================
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        // Load fitness vector (per-thread, into registers)
        reg_x[m] = valid_nfit ? d_ndsfv[m * N_nds + nfit_idx] : 0.f;
        // Load reference points (cooperative, into shared memory)
        s_nrps[m][tidx] = (ref_idx < N_ref) ? d_nrps[m * N_ref + ref_idx] : 0.f;
    }
    __syncthreads();
    
    // ====================================================================
    // Each thread compares its fitness against all refs in shared memory
    // ====================================================================
    if (valid_nfit) {
        // Determine how many refs to process in this tile
        const int num_refs_in_tile = ref_end - ref_start;
        
        for (int r = 0; r < num_refs_in_tile; ++r) {
            const int glb_ref_idx = ref_start + r;
            
            // ============================================================
            // STAGE 1: Compute dot product (nfit · nrp)
            // ============================================================
            float dot = 0.f;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                dot = fmaf(reg_x[m], s_nrps[m][r], dot);
            }
            
            // ============================================================
            // STAGE 2: Compute perpendicular distance² using residual formula
            // ============================================================
            // Numerically stable formula: dist² = ||nfit - dot × nrp||²
            // This avoids catastrophic cancellation when nfit ≈ parallel to nrp
            // Each residual component is small, so squaring and summing is stable
            float dist_sq = 0.f;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                // residual[m] = nfit[m] - projection[m] = nfit[m] - dot * nrp[m]
                const float residual = reg_x[m] - dot * s_nrps[m][r];
                dist_sq = fmaf(residual, residual, dist_sq);
            }
            
            // ============================================================
            // STAGE 3: Update minimum with deterministic tie-breaking
            // ============================================================
            // Tie-breaking rule: when distances are equal, smaller index wins
            // This ensures reproducible results across identical runs
            if (dist_sq < my_min || (dist_sq == my_min && glb_ref_idx < my_amin)) {
                my_min  = dist_sq;
                my_amin = glb_ref_idx;
            }
        }

        // Write results immediately after computation
        const long long out_idx    = (long long)blockIdx.y * (long long)N_nds + (long long)nfit_idx;
        d_assoc_blkx_min[out_idx]  = my_min;
        d_assoc_blkx_amin[out_idx] = my_amin;
    }
}
// ================================================================================================= //
// LAUNCHER: Association Phase 1 Kernel Launcher (Template Specialization)
// ================================================================================================= //
// PURPOSE:
//   Configure and launch association_phase1_kernel with compile-time optimized
//   tile sizes based on the number of objectives M. This launcher provides the
//   bridge between runtime parameters and compile-time template instantiation.
//
// OPTIMIZATION STRATEGY:
//   By fixing M at compile-time, the compiler can:
//   - Fully unroll all M-dependent loops
//   - Optimize register allocation for FMA operations
//   - Eliminate runtime branching in inner loops
//   - Enable instruction-level parallelism (ILP)
//
// GRID CONFIGURATION:
//   Block size: WPB * WS = 8 * 32 = 256 threads
//               Optimal for SM occupancy and warp scheduling
//   Grid dim X: ceil(N_nds / BTF) where BTF = WPB * TF
//               Covers all fitness vectors with minimal padding
//   Grid dim Y: ceil(N_ref / BTR) where BTR = WPB * TR
//               Enables parallel processing of reference point partitions
//
// TEMPLATE PARAMETERS:
//   M: Number of objectives (determines loop unrolling depth)
//      Supported values: 1-8 (extensible by adding more instantiations)
//
// PARAMETERS:
//   INPUT:
//     d_ndsfv: (M, N_nds) float* - Normalized fitness vectors (transposed)
//                   Transposed layout ensures coalesced memory access
//     d_nrps: (M, N_ref) float* - Normalized reference points (transposed)
//                   Column-major format for efficient column loads
//     N_nds:        int - Number of individuals (fitness vectors)
//     N_ref:        int - Number of reference points
//   OUTPUT:
//     d_assoc_blkx_min:   (N_btr, N_nds) float* - Partial minimum squared distances
//                   Intermediate results for Phase 2 global reduction
//     d_assoc_blkx_amin:  (N_btr, N_nds) int* - Partial argmin indices
//                   Reference point indices corresponding to partial minima
// ================================================================================================= //
template<int M>
void launch_kernel_with_M(
    float* d_ndsfv,                // input:    (M, N_nds)
    float* d_nrps,                 // input:    (M, N_ref)
    float* d_assoc_blkx_min,       // buffer: (N_btr, N_nds)
    int*   d_assoc_blkx_amin,      // buffer: (N_btr, N_nds)
    int    N_nds,                  // input:    scalar
    int    N_ref,                  // input:    scalar
    cudaStream_t exec_stream       // input:  cudaStream_t - CUDA execution stream
) {   
    constexpr int WPB_ASSOC = 8;                           // Warps per block
    constexpr int BLK_SIZE = WPB_ASSOC * WS;               // 8 * 32 = 256 threads
    // FIXED: Grid X dimension based on actual block processing size
    const int N_btf = (N_nds + BLK_SIZE - 1) / BLK_SIZE;   // blockIdx.x direction (column)
    const int N_btr = (N_ref + BLK_SIZE - 1) / BLK_SIZE;   // blockIdx.y direction (row)            
    dim3 grid_dim(N_btf, N_btr);
    // Launch with correct block size
    association_phase1_kernel<WPB_ASSOC, M><<<grid_dim, BLK_SIZE, 0, exec_stream>>>
                    (d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref);
    CUDA_CHECK(cudaGetLastError());
}

// template<int M>
// void launch_kernel_with_M(
//     float* d_ndsfv,                // input:    (M, N_nds)
//     float* d_nrps,                // input:    (M, N_ref)
//     float* d_assoc_blkx_min,      // buffer: (N_btr, N_nds)
//     int* d_assoc_blkx_amin,       // buffer: (N_btr, N_nds)
//     int N_nds,                    // input:    scalar
//     int N_ref                     // input:    scalar
// ) {  
//     // Kernel configuration constants
//     constexpr int WPB = 8;            // Warps per block
//     constexpr int TF  = 8;            // Compile-time TF based on M
//     constexpr int BTF = WPB * TF;     // Block tile along N_nds (e.g., 8*8=64)
//     constexpr int BTR = WPB * WS;     // Block tile along N_ref (e.g., 8*32=256)

//     // Grid dimensions
//     const int N_btf = (N_nds + BTF - 1) / BTF;  // Number of blocks along N_nds
//     const int N_btr = (N_ref + BTR - 1) / BTR;  // Number of blocks along N_ref
//     dim3 grid_dim(N_btf, N_btr);

//     // Launch kernel with compile-time template parameters
//     association_phase1_kernel<WPB, M><<<grid_dim, BTR>>>
//                     (d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref);
//     CUDA_CHECK(cudaGetLastError());
// }

// ================================================================================================= //
// LAUNCHER: Association Phase 1 Dynamic Dispatcher
// ================================================================================================= //
// PURPOSE:
//   Dispatch to the appropriate compile-time specialized launcher based on
//   runtime value of M. This function serves as the dynamic entry point that
//   enables full template unrolling while supporting runtime flexibility.
//
// DESIGN RATIONALE:
//   CUDA kernels require compile-time template parameters for optimal performance,
//   but applications often need runtime flexibility. This dispatcher bridges the gap
//   by mapping runtime M values to compile-time template instantiations.
//
// SUPPORTED M VALUES: 
//   1, 2, 3, 4, 5, 6, 7, 8
//   (extensible by adding cases to the switch statement)
//
// ERROR HANDLING:
//   Unsupported M values trigger an error message and graceful return.
//   This prevents undefined behavior while maintaining API flexibility.
//
// PARAMETERS:
//   INPUT:
//     d_ndsfv: (M, N_nds) float* - Normalized fitness vectors (transposed)
//                   Column-major layout for optimal memory coalescing
//     d_nrps: (M, N_ref) float* - Normalized reference points (transposed)
//                   Transposed format enables efficient parallel loads
//     M:            int - Number of objectives (runtime parameter)
//                   Dispatches to corresponding template specialization
//     N_nds:        int - Number of individuals
//     N_ref:        int - Number of reference points
//   OUTPUT:
//     d_assoc_blkx_min:   (N_btr, N_nds) float* - Partial minimum squared distances
//                   Stores local minima for each Y-block partition
//     d_assoc_blkx_amin:  (N_btr, N_nds) int* - Partial argmin indices
//                   Reference point indices for Phase 2 global reduction
// ================================================================================================= //
void launch_assoc_phase1_kernel(
    float* d_ndsfv,              // input: (M, N_nds) - transposed fitness
    float* d_nrps,               // input: (M, N_ref) - transposed ref points
    float* d_assoc_blkx_min,     // output: (N_btr, N_nds) - partial min
    int*   d_assoc_blkx_amin,    // output: (N_btr, N_nds) - partial argmin
    int    N_nds,           
    int    N_ref,
    int    M,
    cudaStream_t exec_stream     // input:  cudaStream_t - CUDA execution stream
){
    // Runtime dispatch to compile-time specialized templates
    switch (M) {
        case 1: launch_kernel_with_M<1>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 2: launch_kernel_with_M<2>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 3: launch_kernel_with_M<3>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 4: launch_kernel_with_M<4>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 5: launch_kernel_with_M<5>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 6: launch_kernel_with_M<6>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 7: launch_kernel_with_M<7>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        case 8: launch_kernel_with_M<8>(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, exec_stream); break;
        default:
            printf("ERROR (Association): Unsupported objective count M = %d (max supported: 8)\n", M);
            return;
    }
}

// ======================================================================================================================= //
// KERNEL: Association Phase 2 - Final Reduction Across Y-Blocks
// ======================================================================================================================= //
// PURPOSE:
//   Reduce partial results from Phase 1 across all Y-dimension blocks to produce
//   final association distances and reference point indices for each individual.
//   This kernel performs the global reduction that completes the association process.
//
// ALGORITHM:
//   For each individual (fitness vector):
//     1. Sequential scan: Iterate through all N_btr partial results from Phase 1
//     2. Global minimum tracking: Maintain running minimum distance² and argmin index
//     3. Tie-breaking: Apply deterministic rule when distances are equal (smaller index wins)
//     4. Distance conversion: Apply sqrt to convert distance² to Euclidean distance
//     5. Atomic-free writeback: Write final results directly to output arrays
//
// PARALLELIZATION STRATEGY:
//   - Embarrassingly parallel: Each thread independently processes one complete row
//   - No inter-thread communication required (no shared memory or atomics)
//   - Grid-stride loop not used: Assumes N_nds < max_grid_size * block_size
//   - Simple 1D thread mapping: tidx = blockIdx.x * blockDim.x + threadIdx.x
//
// DETERMINISTIC TIE-BREAKING:
//   When multiple reference points have equal distance, select the one with
//   smallest index to ensure deterministic and reproducible results across runs.
//   Condition: (dist == local_min && aidx >= 0 && aidx < local_amin)
//   This rule is consistent with Phase 1 kernel for end-to-end determinism.
//
// MEMORY LAYOUT:
//   INPUT:  d_assoc_blkx_min[blk_idx * N_nds + nfit_idx]  (row-major, N_btr rows × N_nds cols)
//   OUTPUT: d_asdist[nfit_idx], d_asrpts_idx[nfit_idx]  (1D arrays of length N_nds)
//
// NUMERICAL STABILITY:
//   The sqrt operation is deferred until this final stage to:
//   - Reduce floating-point operations in Phase 1
//   - Maintain precision during distance comparisons
//   - Enable efficient comparison using squared distances
//
// PARAMETERS:
//   INPUT:
//     d_assoc_blkx_min:  (N_btr, N_nds) const float* - Partial min squared distances from Phase 1
//                  Memory Layout: Row-major, d_assoc_blkx_min[blk_idx][nfit_idx]
//                  Each row contains partial results from one Y-block
//     d_assoc_blkx_amin: (N_btr, N_nds) const int* - Partial argmin indices from Phase 1
//                  Memory Layout: Row-major, d_assoc_blkx_amin[blk_idx][nfit_idx]
//                  Corresponding reference point indices for partial minima
//     N_nds:       int - Number of individuals (output array length)
//     N_btr:       int - Number of Y-blocks (reduction dimension size)
//   OUTPUT:
//     d_asdist:    (N_nds,) float* - Final association distances
//                  Memory Layout: 1D array, d_asdist[nfit_idx] = sqrt(min_dist²)
//                  Euclidean distances to nearest reference points
//     d_asrpts_idx:  (N_nds,) int* - Final reference point indices
//                  Memory Layout: 1D array, d_asrpts_idx[nfit_idx] = argmin_idx
//                  Index of nearest reference point for each individual
// ======================================================================================================================= //
__global__ void association_phase2_kernel(
    const float* __restrict__ d_assoc_blkx_min,   // input: (N_btr, N_nds) - partial min distances²
    const int*   __restrict__ d_assoc_blkx_amin,  // input: (N_btr, N_nds) - partial argmin indices
    float* __restrict__ d_asdist,                 // output: (N_nds,) - final distances
    int*   __restrict__ d_asrpts_idx,             // output: (N_nds,) - final reference point indices
    int N_nds,
    int N_btr
) {
    // Thread index mapping (one thread per individual)
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N_nds) return;
    
    // Local accumulators for minimum tracking
    float local_min  = CUDART_INF_F;  // Best distance² found so far
    int   local_amin = -1;            // Best reference point index found so far
    
    // ============================================================================
    // STAGE 1: Scan all Y-blocks to find global minimum
    // ============================================================================
    // Each thread independently scans all N_btr partial results for its row
    // Scan order is deterministic (0 → N_btr-1), ensuring reproducible tie-breaking
    for (int blk_x_idx = 0; blk_x_idx < N_btr; blk_x_idx++) {
        // Compute linear index for 2D array access
        long long offset = (long long)blk_x_idx * (long long)N_nds + (long long)tidx;
        
        // Load partial results from Phase 1
        float dist = d_assoc_blkx_min[offset];
        int   aidx = d_assoc_blkx_amin[offset];
        
        // ========================================================================
        // Update local minimum with deterministic tie-breaking
        // ========================================================================
        // Tie-breaking rule: when distances are equal, smaller index wins
        // This is consistent with Phase 1 kernel for end-to-end determinism
        // Condition breakdown:
        //   - dist < local_min: strictly better distance found
        //   - dist == local_min && aidx >= 0 && aidx < local_amin: equal distance, smaller valid index
        if (dist < local_min || (dist == local_min && aidx >= 0 && aidx < local_amin)) {
            local_min = dist;
            local_amin = aidx;
        }
    }
    
    // ============================================================================
    // STAGE 2: Apply sqrt and write final results
    // ============================================================================
    // OUTPUT: Convert distance² to distance and write to global memory
    // Apply sqrt only once at final stage for efficiency
    d_asdist[tidx]     = (local_min != CUDART_INF_F) ? sqrtf(local_min) : CUDART_INF_F;
    d_asrpts_idx[tidx] = local_amin;
}

// ================================================================================================= //
// LAUNCHER: Association Phase 2 Kernel Launcher
// ================================================================================================= //
// PURPOSE:
//   Configure and launch association_phase2_kernel to perform final reduction
//   across Y-blocks and produce the final association results. This launcher
//   handles grid configuration and kernel invocation for the reduction stage.
//
// ALGORITHM SUMMARY:
//   Phase 2 completes the two-stage reduction pattern:
//   - Phase 1: Parallel distance computation with local reduction per Y-block
//   - Phase 2: Global reduction across Y-blocks to find true minimum
//
// GRID CONFIGURATION:
//   Block size: BTR = WPB * TR = 8 * 32 = 256 threads
//               Fixed block size for optimal occupancy
//   Grid size:  ceil(N_nds / BTR) blocks
//               One thread per individual (fitness vector)
//
// DESIGN RATIONALE:
//   The block size (256) is chosen to:
//   - Maximize SM occupancy
//   - Avoid register pressure
//   - Enable efficient global memory coalescing
//   - Match Phase 1 tile dimensions for load balancing
//
// PARAMETERS:
//   INPUT:
//     d_assoc_blkx_min:  (N_btr, N_nds) float* - Partial min squared distances from Phase 1
//                  Row-major layout with N_btr rows (one per Y-block)
//     d_assoc_blkx_amin: (N_btr, N_nds) int* - Partial argmin indices from Phase 1
//                  Corresponding reference point indices for partial results
//     N_nds:       int - Number of individuals (determines output array size)
//     N_ref:       int - Number of reference points (used to compute N_btr)
//   OUTPUT:
//     d_asdist:    (N_nds,) float* - Final association distances
//                  Euclidean distances to nearest reference points
//     d_asrpts_idx:  (N_nds,) int* - Final reference point indices
//                  Index of nearest reference point for each individual
// ================================================================================================= //
void launch_assoc_phase2_kernel(
    float* d_assoc_blkx_min,      // input: (N_btr, N_nds) - partial min
    int*   d_assoc_blkx_amin,     // input: (N_btr, N_nds) - partial argmin
    float* d_asdist,              // output: (N_nds,) - final distances
    int*   d_asrpts_idx,          // output: (N_nds,) - final ref point indices
    int    N_nds,
    int    N_ref,
    cudaStream_t exec_stream      // input:  cudaStream_t - CUDA execution stream
){
    // Kernel configuration constants
    constexpr int WPB_ASSOC = 8;                                 // Warps per block
    constexpr int BLK_SIZE  = WPB_ASSOC * WS;                    // Block tile size (256)
    // Compute number of Y-blocks from Phase 1
    const int N_btr         = (N_ref + BLK_SIZE - 1) / BLK_SIZE; // Number of blocks along N_ref dimension
    // Grid dimension
    const int N_btf         = (N_nds + BLK_SIZE - 1) / BLK_SIZE; // Number of blocks needed
    
    // Launch kernel
    association_phase2_kernel<<<N_btf, BLK_SIZE, 0, exec_stream>>>
                        (d_assoc_blkx_min, d_assoc_blkx_amin, d_asdist, d_asrpts_idx, N_nds, N_btr);
    CUDA_CHECK(cudaGetLastError());
}

// ==============================================================================================
// MAIN FUNCTION: Execute Complete Association Pipeline
// ==============================================================================================
// PURPOSE:
//   Orchestrate the complete association process between normalized fitness vectors
//   and reference points. This function coordinates a two-stage GPU pipeline that
//   efficiently computes perpendicular distances and finds nearest reference points.
//
// ALGORITHM OVERVIEW:
//   Given normalized fitness vectors F (N_nds × M) and reference points R (N_ref × M):
//   
//   Stage 1 - Phase 1 Kernel: Parallel Distance Computation & Partial Reduction
//     For each (i, j) pair where i ∈ [0, N_nds) and j ∈ [0, N_ref):
//       1. Compute dot product: F[i] · R[j]
//       2. Compute perpendicular distance² using numerically stable residual formula:
//          dist² = Σ(F[i][m] - dot × R[j][m])²
//       3. Perform local argmin reduction within Y-blocks with deterministic tie-breaking
//       4. Output partial results: (min_dist², argmin_idx) per Y-block
//   
//   Stage 2 - Phase 2 Kernel: Global Reduction
//     For each individual i ∈ [0, N_nds):
//       1. Collect all N_btr partial results from Phase 1
//       2. Find global minimum with deterministic tie-breaking: argmin_j(dist²[i,j])
//       3. Convert to Euclidean distance: dist[i] = sqrt(min_dist²)
//       4. Output final results: (dist[i], argmin_idx[i])
//
// MEMORY LAYOUT DESIGN:
//   All matrices use column-major (transposed) format to enable perfect coalescing:
//   - d_ndsfv[m * N_nds + i]: Fitness vector F[i] dimension m
//   - d_nrps[m * N_ref + j]: Reference point R[j] dimension m
//   This layout ensures consecutive threads access consecutive memory locations.
//
// TWO-STAGE REDUCTION RATIONALE:
//   Direct single-kernel approach requires global atomics (slow)
//   Two-stage approach enables:
//   - Lock-free parallel processing in Phase 1
//   - Embarrassingly parallel reduction in Phase 2
//   - Optimal memory access patterns throughout
//   - Scalability to large N_ref values
//
// PARAMETERS:
//   INPUT:
//     d_ndsfv: (M, N_nds) float* - Normalized fitness vectors (transposed)
//                   Memory Layout: Column-major, d_ndsfv[m * N_nds + idx]
//                   Normalization: ||F[i]|| = 1 for all i
//     d_nrps: (M, N_ref) float* - Normalized reference points (transposed)
//                   Memory Layout: Column-major, d_nrps[m * N_ref + idx]
//                   Normalization: ||R[j]|| = 1 for all j
//     N_nds:        int - Number of non-dominated individuals (fitness vectors)
//     M:            int - Number of objectives (vector dimensions)
//     N_ref:        int - Number of reference points
//     enable_h_save: bool - Flag for saving debug output (reserved for future use)
//     print_time:   bool - Flag for timing measurements (reserved for future use)
//   
//   INTERNAL BUFFERS:
//     d_assoc_blkx_min:   (N_btr, N_nds) float* - Partial minimum squared distances from Phase 1
//                   Allocated externally, must have size: N_btr * N_nds * sizeof(float)
//                   where N_btr = ceil(N_ref / (WPB * TR))
//     d_assoc_blkx_amin:  (N_btr, N_nds) int* - Partial argmin indices from Phase 1
//                   Allocated externally, must have size: N_btr * N_nds * sizeof(int)
//   
//   OUTPUT:
//     d_asdist:     (N_nds,) float* - Final association distances
//                   Memory Layout: 1D array, d_asdist[i] = min_j(dist(F[i], R[j]))
//                   Euclidean distance from each fitness vector to nearest reference point
//     d_asrpts_idx:   (N_nds,) int* - Associated reference point indices
//                   Memory Layout: 1D array, d_asrpts_idx[i] = argmin_j(dist(F[i], R[j]))
//                   Index of nearest reference point for each fitness vector
//
// PERFORMANCE CHARACTERISTICS:
//   Computational Complexity: O(N_nds * N_ref * M)
//   Memory Bandwidth: O(N_nds * M + N_ref * M + N_nds * N_btr)
//   Parallelism: Exploits both N_nds and N_ref dimensions
// ==============================================================================================
void execute_association(
    float* d_ndsfv,              // input:  (M, N_nds)     - normalized fitness (transposed)
    float* d_nrps,               // input:  (M, N_ref)     - normalized ref points (transposed)
    float* d_assoc_blkx_min,     // buffer: (N_btr, N_nds) - partial min buffer
    int*   d_assoc_blkx_amin,    // buffer: (N_btr, N_nds) - partial argmin buffer
    float* d_asdist,             // output: (N_nds,)       - final distances
    int*   d_asrpts_idx,         // output: (N_nds,)       - final ref point indices
    int    N_ref,                // number of reference points
    int    M,                    // number of objectives
    int    N_nds,                // number of non-dominated individuals
    cudaStream_t exec_stream     // input:  cudaStream_t - CUDA execution stream
)
{   
    // ============================================================================
    // STAGE 1: Fused Association Phase 1 - Distance Computation & Partial Reduction
    // ============================================================================
    launch_assoc_phase1_kernel(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref, M, exec_stream);
    // ============================================================================
    // STAGE 2: Association Phase 2 - Final Reduction
    // ============================================================================
    launch_assoc_phase2_kernel(d_assoc_blkx_min, d_assoc_blkx_amin, d_asdist, d_asrpts_idx, N_nds, N_ref, exec_stream);

}
