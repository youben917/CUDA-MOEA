#include <math_constants.h>

#include "cuda_moea/core/cuda/cuda_globals.cuh"

// ======================================================================================== //
// KERNEL: Association Phase 1 - Angle-Based Distance Computation and Partial Reduction
// ======================================================================================== //
// PURPOSE:
//   Compute cosine similarity between normalized fitness vectors and reference points
//   using the angle-based association formula from I-NSGA-III:
//     cos(θ) = (f^n · w) / (||f^n|| × ||w||)
//   Since reference points are pre-normalized (||w|| = 1), this simplifies to:
//     cos(θ) = (f^n · w) / ||f^n||
//
// ALGORITHM:
//   1. Cooperative data loading into shared memory with perfect coalescing:
// - Load d_ndsfv[(m, nfit_idx)] → reg_x[m]
// - Load d_nrps[(m, nrps_idx)] → s_nrps[m][tidx]
//   2. Compute ||f^n||² and cache inv_norm = rsqrt(||f^n||²) per fitness vector
//   3. Compute dot products using FMA tree: f^n · w
//   4. Compute cosine similarity: cos(θ) = dot × inv_norm
//   5. Update best (max_cos, argmax_idx) with deterministic tie-breaking
//   6. Write partial results to global memory for Phase 2 reduction
//
// KEY INSIGHT:
//   In the non-negative orthant (all f^n_m ≥ 0, all w_m ≥ 0), cos(θ) ∈ [0, 1]
//   and cos is monotonically decreasing on [0°, 90°], therefore:
//     max(cos(θ)) ⟺ min(θ)
//   This allows us to find minimum angle by finding maximum cosine value.
//
// DETERMINISTIC TIE-BREAKING:
//   When multiple reference points have equal cosine values, the smaller index wins.
//   This ensures reproducible results across identical runs.
//
// TEMPLATE PARAMETERS:
//   WPB: Warps per block (default: 8)
//   M:   Number of objectives (compile-time constant for loop unrolling)
//
// PARAMETERS:
//   INPUT:
//     d_ndsfv: (M, N_nds) const float* - Normalized fitness vectors (transposed)
//                   Memory Layout: Column-major, element access: d_ndsfv[m * N_nds + idx]
//                   Note: Intercept-normalized, NOT L2-normalized
//     d_nrps: (M, N_ref) const float* - L2-normalized reference points (transposed)
//                   Memory Layout: Column-major, element access: d_nrps[m * N_ref + idx]
//                   Guarantee: ||d_nrps[:, idx]|| = 1 for all idx
//     N_nds:        int - Number of non-dominated individuals (fitness vectors)
//     N_ref:        int - Number of reference points
//   OUTPUT:
//     d_assoc_blkx_min:   (N_btr, N_nds) float* - Partial maximum cosine values (name kept for compatibility)
//     d_assoc_blkx_amin:  (N_btr, N_nds) int* - Partial argmax reference point indices
// ========================================================================================================================================== //
template<int WPB = 8, int M = 3, int UNROLL = 4>
__global__ void association_phase1_kernel(
    const float* __restrict__ d_ndsfv,        // input:  (M, N_nds)     - normalized fitness value (normalized by intercept or max in the 'normalization' process)
    const float* __restrict__ d_nrps,         // input:  (M, N_ref)     - L2-normalized reference points (||w||=1)
    float* __restrict__ d_assoc_blkx_min,     // output: (N_btr, N_nds) - partial max cosine (name kept for API compatibility)
    int*   __restrict__ d_assoc_blkx_amin,    // output: (N_btr, N_nds) - partial argmax indices
    int N_nds,                                // input:  scalar         - size of popluation after non-dominated sorting
    int N_ref                                 // input:  scalar         - number of reference points
)
{
    constexpr int BS = WPB * WS;
    
    const int  tidx       = threadIdx.x;
    const int  nfit_idx   = blockIdx.x * BS + tidx;
    const bool valid_nfit = (nfit_idx < N_nds);
    
    __shared__ float s_nrps[M][BS];
    
    // ====================================================================
    // Merged loop: Load fitness + compute norm + load reference points
    // Combines operations to improve instruction scheduling
    // ====================================================================
    float reg_x[M];
    float sqr_nfit = 0.f;
    
    const int ref_start = blockIdx.y * BS;
    const int ref_end   = min(ref_start + BS, N_ref);
    const int ref_idx   = ref_start + tidx;
    
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        // Load fitness and accumulate norm (interleaved for better scheduling)
        float val = valid_nfit ? d_ndsfv[m * N_nds + nfit_idx] : 0.f;
        reg_x[m] = val;
        sqr_nfit = fmaf(val, val, sqr_nfit);
        
        // Cooperative load of reference points
        s_nrps[m][tidx] = (ref_idx < N_ref) ? d_nrps[m * N_ref + ref_idx] : 0.f;
    }
    
    const float inv_norm_nfit = rsqrtf(sqr_nfit + 1e-12f);
    __syncthreads();
    
    float my_max = -CUDART_INF_F;
    int my_amax  = -1;
    
    if (valid_nfit) {
        const int num_refs_in_tile = ref_end - ref_start;
        const int num_refs_aligned = (num_refs_in_tile / UNROLL) * UNROLL;
        
        // ================================================================
        // Main loop: Process UNROLL reference points per iteration
        // Increases arithmetic intensity and enables better ILP
        // ================================================================
        for (int r = 0; r < num_refs_aligned; r += UNROLL) {
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
            
            // Compute all cosines
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                cos_theta[u] = dot[u] * inv_norm_nfit;
            }
            
            // Update maximum (sequential to maintain determinism)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int glb_ref_idx = ref_start + r + u;
                if (cos_theta[u] > my_max || 
                    (cos_theta[u] == my_max && glb_ref_idx < my_amax)) {
                    my_max  = cos_theta[u];
                    my_amax = glb_ref_idx;
                }
            }
        }
        
        // ================================================================
        // Tail loop: Handle remaining references
        // ================================================================
        for (int r = num_refs_aligned; r < num_refs_in_tile; ++r) {
            const int glb_ref_idx = ref_start + r;
            
            float dot = 0.f;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                dot = fmaf(reg_x[m], s_nrps[m][r], dot);
            }
            
            const float cos_theta = dot * inv_norm_nfit;
            
            if (cos_theta > my_max || 
                (cos_theta == my_max && glb_ref_idx < my_amax)) {
                my_max  = cos_theta;
                my_amax = glb_ref_idx;
            }
        }

        const long long out_idx    = (long long)blockIdx.y * (long long)N_nds + (long long)nfit_idx;
        d_assoc_blkx_min[out_idx]  = my_max;
        d_assoc_blkx_amin[out_idx] = my_amax;
    }
}
// ========================================================================================================================================== //
// template<int WPB = 8, int M = 3>
// __global__ void association_phase1_kernel(
//     const float* __restrict__ d_ndsfv,        // input: (M, N_nds) - intercept - normalized fitness
//     const float* __restrict__ d_nrps,         // input: (M, N_ref) - L2 - normalized reference points (||w||=1)
//     float* __restrict__ d_assoc_blkx_min,     // output: (N_btr, N_nds) - partial max cosine (name kept for API compatibility)
//     int*   __restrict__ d_assoc_blkx_amin,    // output: (N_btr, N_nds) - partial argmax indices
//     int N_nds,                                // input: scalar - scalar 
//     int N_ref                                 // input: scalar - scalar 
// )
// {
//     // ====================================================================
//     // Constants
//     // ====================================================================
//     constexpr int BS = WPB * WS;   // Block size = 256
    
//     // ====================================================================
//     // Thread indexing
//     // ====================================================================
//     const int  tidx       = threadIdx.x;
//     const int  nfit_idx   = blockIdx.x * BS + tidx;
//     const bool valid_nfit = (nfit_idx < N_nds);
    
//     // ====================================================================
//     // Shared memory for reference points (L2-normalized, ||w||=1)
//     // ====================================================================
//     __shared__ float s_nrps[M][BS];
    
//     // ====================================================================
//     // Load fitness vector and compute L2 norm squared
//     // ====================================================================
//     float reg_x[M];
//     float sqr_nfit = 0.f;
    
//     #pragma unroll
//     for (int m = 0; m < M; ++m) {
//         float val = valid_nfit ? d_ndsfv[m * N_nds + nfit_idx] : 0.f;
//         reg_x[m] = val;
//         sqr_nfit = fmaf(val, val, sqr_nfit);  // ||f||² accumulation
//     }
    
//     // Compute inverse norm with numerical safety
//     // rsqrtf is faster than sqrtf + division on NVIDIA GPUs
//     constexpr float EPSILON = 1e-12f;
//     const float inv_norm_nfit = rsqrtf(sqr_nfit + EPSILON);
    
//     // ====================================================================
//     // Prepare reference tile boundaries
//     // ====================================================================
//     const int ref_start = blockIdx.y * BS;
//     const int ref_end   = min(ref_start + BS, N_ref);
//     const int ref_idx   = ref_start + tidx;
    
//     // ====================================================================
//     // Cooperative loading of reference points into shared memory
//     // Since ||w||=1, no need to compute or store reference norms
//     // ====================================================================
//     #pragma unroll
//     for (int m = 0; m < M; ++m) {
//         s_nrps[m][tidx] = (ref_idx < N_ref) ? d_nrps[m * N_ref + ref_idx] : 0.f;
//     }
//     __syncthreads();
    
//     // ====================================================================
//     // Initialize per-thread reduction variables
//     // Finding MAXIMUM cosine (minimum angle)
//     // ====================================================================
//     float my_max = -CUDART_INF_F;  // Initialize to -INF for max finding
//     int my_amax  = -1;
    
//     // ====================================================================
//     // Each thread compares its fitness against all refs in shared memory
//     // ====================================================================
//     if (valid_nfit) {
//         const int num_refs_in_tile = ref_end - ref_start;
        
//         for (int r = 0; r < num_refs_in_tile; ++r) {
//             const int glb_ref_idx = ref_start + r;
            
//             // ============================================================
//             // STAGE 1: Compute dot product (f^n · w)
//             // ============================================================
//             float dot = 0.f;
//             #pragma unroll
//             for (int m = 0; m < M; ++m) {
//                 dot = fmaf(reg_x[m], s_nrps[m][r], dot);
//             }
            
//             // ============================================================
//             // STAGE 2: Compute cosine similarity
//             // Since ||w|| = 1: cos(θ) = dot / ||f|| = dot × inv_norm_nfit
//             // ============================================================
//             const float cos_theta = dot * inv_norm_nfit;
            
//             // ============================================================
//             // STAGE 3: Update maximum with deterministic tie-breaking
//             // max(cos) ⟺ min(angle) in [0°, 90°] range
//             // Tie-breaking: equal cosine → smaller index wins
//             // ============================================================
//             if (cos_theta > my_max || (cos_theta == my_max && glb_ref_idx < my_amax)) {
//                 my_max  = cos_theta;
//                 my_amax = glb_ref_idx;
//             }
//         }

//         // Write partial results to global memory
//         const long long out_idx      = (long long)blockIdx.y * (long long)N_nds + (long long)nfit_idx;
//         d_assoc_blkx_min[out_idx]    = my_max;   // Store cosine value (higher = better)
//         d_assoc_blkx_amin[out_idx]   = my_amax;
//     }
// }

// ================================================================================================= //
// LAUNCHER: Association Phase 1 Kernel Launcher (Template Specialization)
// ================================================================================================= //
template<int M>
void launch_kernel_with_M(
    float* d_ndsfv,                // input: (M, N_nds) - intercept-normalized fitness
    float* d_nrps,                 // input: (M, N_ref) - L2-normalized ref points
    float* d_assoc_blkx_min,       // buffer: (N_btr, N_nds) - partial max cosine
    int*   d_assoc_blkx_amin,      // buffer: (N_btr, N_nds) - partial argmax
    int    N_nds,                  // input: scalar - number of non-dominated individuals
    int    N_ref,                  // input: scalar - number of reference points
    cudaStream_t exec_stream       // input: scalar - CUDA execution stream
)
{
    constexpr int WPB_ASSOC = 8;
    constexpr int BLK_SIZE = WPB_ASSOC * WS;               // 256 threads
    const int N_btf = (N_nds + BLK_SIZE - 1) / BLK_SIZE;   // blocks in X (fitness)
    const int N_btr = (N_ref + BLK_SIZE - 1) / BLK_SIZE;   // blocks in Y (refs)
    dim3 grid_dim(N_btf, N_btr);
    
    association_phase1_kernel<WPB_ASSOC, M><<<grid_dim, BLK_SIZE, 0, exec_stream>>>
                    (d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, N_nds, N_ref);
    CUDA_CHECK(cudaGetLastError());
}

// ================================================================================================= //
// LAUNCHER: Association Phase 1 Dynamic Dispatcher
// ================================================================================================= //
void launch_assoc_phase1_kernel(
    float* d_ndsfv,              // input: (M, N_nds) - intercept-normalized fitness
    float* d_nrps,               // input: (M, N_ref) - L2-normalized ref points
    float* d_assoc_blkx_min,     // output: (N_btr, N_nds) - partial max cosine
    int*   d_assoc_blkx_amin,    // output: (N_btr, N_nds) - partial argmax
    int    N_nds,                // input: scalar - number of non-dominated individuals
    int    N_ref,                // input: scalar - number of reference points
    int    M,                    // input: scalar - number of objectives
    cudaStream_t exec_stream     // input: scalar - CUDA execution stream
){
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
// KERNEL: Association Phase 2 - Final Reduction Across Y-Blocks (Angle-Based)
// ======================================================================================================================= //
// PURPOSE:
//   Reduce partial results from Phase 1 across all Y-dimension blocks to produce
//   final association cosine values and reference point indices for each individual.
//
// ALGORITHM:
//   For each individual (fitness vector):
//     1. Sequential scan: Iterate through all N_btr partial results from Phase 1
//     2. Global maximum tracking: Maintain running maximum cosine and argmax index
//     3. Tie-breaking: Apply deterministic rule (smaller index wins)
//     4. Output conversion: Convert cos to angle via acos for compatibility
//
// NOTE ON OUTPUT:
//   To maintain semantic compatibility with perpendicular distance version,
//   we output the ANGLE (in radians) rather than cosine value.
//   This ensures "smaller d_asdist = better association" semantic is preserved.
//
// PARAMETERS:
//   INPUT:
//     d_assoc_blkx_min:  (N_btr, N_nds) const float* - Partial max cosine from Phase 1
//     d_assoc_blkx_amin: (N_btr, N_nds) const int* - Partial argmax indices from Phase 1
//     N_nds:       int - Number of individuals
//     N_btr:       int - Number of Y-blocks
//   OUTPUT:
//     d_asdist:      (N_nds,) float* - Final angle values (radians, smaller = better)
//     d_asrpts_idx:  (N_nds,) int* - Final reference point indices
// ======================================================================================================================= //
__global__ void association_phase2_kernel(
    const float* __restrict__ d_assoc_blkx_min,   // input: (N_btr, N_nds) - partial max cosine
    const int*   __restrict__ d_assoc_blkx_amin,  // input: (N_btr, N_nds) - partial argmax indices
    float* __restrict__ d_asdist,                 // output: (N_nds,) - final angle values (radians)
    int*   __restrict__ d_asrpts_idx,             // output: (N_nds,) - final reference point indices
    int N_nds,
    int N_btr
) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N_nds) return;
    
    // Initialize for maximum finding (max cosine = min angle)
    float local_max  = -CUDART_INF_F;
    int   local_amax = -1;
    
    // ============================================================================
    // STAGE 1: Scan all Y-blocks to find global maximum cosine
    // ============================================================================
    for (int blk_y_idx = 0; blk_y_idx < N_btr; blk_y_idx++) {
        long long offset = (long long)blk_y_idx * (long long)N_nds + (long long)tidx;
        
        float cos_val = d_assoc_blkx_min[offset];
        int   aidx    = d_assoc_blkx_amin[offset];
        
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
    // Convert cosine to angle (radians) to maintain "smaller = better" semantic
    // Clamp to [0, 1] to handle numerical precision issues before acos
    if (local_max > -CUDART_INF_F) {
        // Clamp cosine to valid range [0, 1] for numerical stability
        // In non-negative orthant, cos(θ) should be in [0, 1]
        float clamped_cos = fminf(fmaxf(local_max, 0.f), 1.f);
        d_asdist[tidx] = acosf(clamped_cos);  // Output angle in radians
    } else {
        d_asdist[tidx] = CUDART_INF_F;  // Invalid: set to infinity
    }
    
    d_asrpts_idx[tidx] = local_amax;
}

// ================================================================================================= //
// LAUNCHER: Association Phase 2 Kernel Launcher
// ================================================================================================= //
void launch_assoc_phase2_kernel(
    float* d_assoc_blkx_min,      // input: (N_btr, N_nds) - partial max cosine
    int*   d_assoc_blkx_amin,     // input: (N_btr, N_nds) - partial argmax
    float* d_asdist,              // output: (N_nds,) - final angle values
    int*   d_asrpts_idx,          // output: (N_nds,) - final ref point indices
    int    N_nds,
    int    N_ref,
    cudaStream_t exec_stream
){
    constexpr int WPB_ASSOC = 8;
    constexpr int BLK_SIZE  = WPB_ASSOC * WS;
    const int N_btr         = (N_ref + BLK_SIZE - 1) / BLK_SIZE;
    const int N_btf         = (N_nds + BLK_SIZE - 1) / BLK_SIZE;
    
    association_phase2_kernel<<<N_btf, BLK_SIZE, 0, exec_stream>>>
                        (d_assoc_blkx_min, d_assoc_blkx_amin, d_asdist, d_asrpts_idx, N_nds, N_btr);
    CUDA_CHECK(cudaGetLastError());
}

// ==============================================================================================
// MAIN FUNCTION: Execute Complete Angle-Based Association Pipeline
// ==============================================================================================
// PURPOSE:
//   Orchestrate the complete angle-based association process between normalized
//   fitness vectors and L2-normalized reference points. Implements the I-NSGA-III
//   angle-based association strategy from Eq. (17) of the Wu et al. paper.
//
// ALGORITHM OVERVIEW:
//   Given intercept-normalized fitness vectors F (N_nds × M) and 
//   L2-normalized reference points W (N_ref × M) where ||W[j]|| = 1:
//   
//   Stage 1 - Phase 1 Kernel: Parallel Cosine Computation & Partial Reduction
//     For each (i, j) pair where i ∈ [0, N_nds) and j ∈ [0, N_ref):
//       1. Compute ||F[i]|| once per fitness vector
//       2. Compute dot product: F[i] · W[j]
//       3. Compute cosine: cos(θ) = dot / ||F[i]||  (since ||W[j]|| = 1)
//       4. Perform local argmax reduction within Y-blocks
//       5. Output partial results: (max_cos, argmax_idx) per Y-block
//   
//   Stage 2 - Phase 2 Kernel: Global Reduction
//     For each individual i ∈ [0, N_nds):
//       1. Collect all N_btr partial results from Phase 1
//       2. Find global maximum: argmax_j(cos(θ[i,j])) = argmin_j(θ[i,j])
//       3. Convert to angle: θ = acos(max_cos)
//       4. Output final results: (angle[i], argmax_idx[i])
//
// OUTPUT SEMANTIC COMPATIBILITY:
//   d_asdist outputs ANGLE (radians) instead of cosine to maintain
//   "smaller value = better association" semantic, compatible with
//   perpendicular distance version for downstream niche counting.
//
// PARAMETERS:
//   INPUT:
//     d_ndsfv: (M, N_nds) float* - Intercept-normalized fitness vectors (transposed)
//     d_nrps:  (M, N_ref) float* - L2-normalized reference points (transposed, ||w||=1)
//     N_nds:   int - Number of non-dominated individuals
//     M:       int - Number of objectives
//     N_ref:   int - Number of reference points
//   
//   INTERNAL BUFFERS:
//     d_assoc_blkx_min:  (N_btr, N_nds) float* - Partial maximum cosine values
//     d_assoc_blkx_amin: (N_btr, N_nds) int* - Partial argmax indices
//   
//   OUTPUT:
//     d_asdist:     (N_nds,) float* - Final association angles (radians)
//                   Interpretation: Smaller value = smaller angle = better alignment
//     d_asrpts_idx: (N_nds,) int* - Associated reference point indices
//                   Index of most aligned reference point for each fitness vector
// ==============================================================================================
void execute_association(
    float* d_ndsfv,              // input: (M, N_nds) - intercept-normalized fitness (transposed)
    float* d_nrps,               // input: (M, N_ref) - L2-normalized ref points (transposed)
    float* d_assoc_blkx_min,     // buffer: (N_btr, N_nds) - partial max buffer
    int*   d_assoc_blkx_amin,    // buffer: (N_btr, N_nds) - partial argmax buffer
    float* d_asdist,             // output: (N_nds,) - final angle values
    int*   d_asrpts_idx,         // output: (N_nds,) - final ref point indices
    int    N_ref,                // input: scalar - number of reference points
    int    M,                    // input: scalar - number of objectives
    int    N_nds,                // input: scalar - number of non-dominated individuals
    cudaStream_t exec_stream     // input: scalar - CUDA execution stream
)
{   
    // ============================================================================
    // STAGE 1: Angle-Based Association Phase 1 - Cosine Computation & Partial Reduction
    // ============================================================================
    launch_assoc_phase1_kernel(d_ndsfv, d_nrps, d_assoc_blkx_min, d_assoc_blkx_amin, 
                               N_nds, N_ref, M, exec_stream);
    
    // ============================================================================
    // STAGE 2: Association Phase 2 - Final Reduction (outputs angle in radians)
    // ============================================================================
    launch_assoc_phase2_kernel(d_assoc_blkx_min, d_assoc_blkx_amin, 
                               d_asdist, d_asrpts_idx, N_nds, N_ref, exec_stream);
}
