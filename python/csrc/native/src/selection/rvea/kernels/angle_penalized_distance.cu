#include <math_constants.h>

#include "cuda_moea/selection/rvea/kernels/angle_penalized_distance.cuh"

#include "cuda_moea/core/cuda/cuda_globals.cuh"

namespace {

constexpr float NORM_EPS   = 1e-12f;   // avoid unstable APD for near-zero translated vectors
constexpr float GAMMA_EPS  = 1e-6f;

constexpr float METRIC_MAX = 1e30f;    // bounded sentinel for invalid APD

__device__ __forceinline__ int bool_to_mask(const bool pred)
{
    return -static_cast<int>(pred);
}

__device__ __forceinline__ float bitselect_f32(
    const int mask,                     // input: 0xFFFFFFFF -> take_true, 0x00000000 -> take_false
    const float take_true,
    const float take_false
)
{
    const unsigned int umask = static_cast<unsigned int>(mask);
    const unsigned int a     = __float_as_uint(take_true);
    const unsigned int b     = __float_as_uint(take_false);
    return __uint_as_float((a & umask) | (b & ~umask));
}

} // namespace

// ========================================================================================================== //
// KERNEL: APD Computation (thread-per-individual, grid-stride)
// ========================================================================================================== //
// PURPOSE:
//   Compute APD metric for each individual:
//     APD_i = (1 + penalty_scale * theta_i / gamma_ref(i)) * ||f'_i||
//
// NOTES:
//   1) d_translated_fv uses transposed layout: d_translated_fv[m * pop_size + i]
//   2) Near-zero norm individuals get APD = 0 (skips global reads for theta/gamma)
// ========================================================================================================== //
template<int M = 3>
__global__ void apd_kernel(
    const float* __restrict__ d_translated_fv,        // input:  (M, pop_size) - translated fitness (transposed)
    const float* __restrict__ d_theta,                // input:  (pop_size,)   - partition angle (radians)
    const float* __restrict__ d_gamma,                  // input:  (n_refvec,)   - gamma per reference vector
    const int*   __restrict__ d_refvec_idx,           // input:  (pop_size,)   - assigned reference vector index
    float*       __restrict__ d_apd,                  // output: (pop_size,)   - APD metric
    float*       __restrict__ d_norm,                 // output: (pop_size,)   - translated-fitness L2 norm
    float penalty_scale,                              // input:  scalar        - M * progress^alpha
    int pop_size,                                     // input:  scalar        - number of individuals
    int n_refvec                                      // input:  scalar        - number of reference vectors
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < pop_size;
         i += gridDim.x * blockDim.x) {

        // ==================================================================
        // STAGE 1: Compute norm of translated fitness vector ||f'_i||
        // ==================================================================
        float norm_sq = 0.f;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const float val = d_translated_fv[m * pop_size + i];
            norm_sq = fmaf(val, val, norm_sq);
        }

        const float norm = sqrtf(norm_sq);
        d_norm[i] = norm;

        // ==================================================================
        // STAGE 2 & 3: Read inputs, compute APD, sanitize
        // ==================================================================
        if (norm > NORM_EPS) {
            const int   ref_idx    = d_refvec_idx[i];

            // Defensive: invalid ref_idx → assign maximum APD
            if (ref_idx < 0 || ref_idx >= n_refvec) {
                d_apd[i] = METRIC_MAX;
                continue;
            }

            const float theta_safe = fminf(fmaxf(d_theta[i], 0.f), CUDART_PI_F);
            const float gamma_safe = fmaxf(d_gamma[ref_idx], GAMMA_EPS);
            // Cheng et al. (2016), Eq. (8)--(9): only the angle penalty
            // is normalized by the associated reference vector's gamma.
            const float apd = (1.f + penalty_scale * theta_safe / gamma_safe) * norm;

            d_apd[i] = (!isfinite(apd) || apd < 0.f) ? METRIC_MAX : apd;
        } else {
            d_apd[i] = 0.f;
        }
    }
}

// ========================================================================================================== //
// LAUNCHER: APD Kernel Launcher (Template Specialization)
// ========================================================================================================== //
template<int M>
void launch_apd_kernel_with_M(
    const float* d_translated_fv,        // input:  (M, pop_size) - translated fitness
    const float* d_theta,                // input:  (pop_size,) - partition angle
    const float* d_gamma,                  // input:  (n_refvec,) - gamma values
    const int*   d_refvec_idx,           // input:  (pop_size,) - ref-vector assignment
    float*       d_apd,                  // output: (pop_size,) - APD metric
    float*       d_norm,                 // output: (pop_size,) - translated-fitness norm
    float        penalty_scale,          // input:  scalar - precomputed APD penalty scale
    int          pop_size,               // input:  scalar - number of individuals
    int          n_refvec,               // input:  scalar - number of reference vectors
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE    = (pop_size + BLK_SIZE - 1) / BLK_SIZE;

    apd_kernel<M><<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================================== //
// LAUNCHER: APD Kernel Runtime Dispatcher
// ========================================================================================================== //
void launch_apd_kernel(
    const float* d_translated_fv,        // input:  (M, pop_size) - translated fitness
    const float* d_theta,                // input:  (pop_size,) - partition angle
    const float* d_gamma,                  // input:  (n_refvec,) - gamma values
    const int*   d_refvec_idx,           // input:  (pop_size,) - ref-vector assignment
    float*       d_apd,                  // output: (pop_size,) - APD metric
    float*       d_norm,                 // output: (pop_size,) - translated-fitness norm
    float        penalty_scale,          // input:  scalar - precomputed APD penalty scale
    int          pop_size,               // input:  scalar - number of individuals
    int          M,                      // input:  scalar - number of objectives
    int          n_refvec,               // input:  scalar - number of reference vectors
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    switch (M) {
        case 1: launch_apd_kernel_with_M<1>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 2: launch_apd_kernel_with_M<2>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 3: launch_apd_kernel_with_M<3>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 4: launch_apd_kernel_with_M<4>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 5: launch_apd_kernel_with_M<5>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 6: launch_apd_kernel_with_M<6>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 7: launch_apd_kernel_with_M<7>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        case 8: launch_apd_kernel_with_M<8>(d_translated_fv, d_theta, d_gamma, d_refvec_idx, d_apd, d_norm, penalty_scale, pop_size, n_refvec, exec_stream); break;
        default:
            printf("ERROR (APD): Unsupported objective count M = %d (max supported: 8)\n", M);
            return;
    }
}

// ========================================================================================================== //
// MAIN FUNCTION: Compute APD
// ========================================================================================================== //
void rvea::compute_apd(
    const float* d_translated_fv,        // input:  (M, pop_size) - translated fitness
    const float* d_theta,                // input:  (pop_size,) - partition angle
    const float* d_gamma,                  // input:  (n_refvec,) - gamma values
    const int*   d_refvec_idx,           // input:  (pop_size,) - ref-vector assignment
    float*       d_apd,                  // output: (pop_size,) - APD metric
    float*       d_norm,                 // output: (pop_size,) - translated-fitness norm
    int          pop_size,               // input:  scalar - number of individuals
    int          M,                      // input:  scalar - number of objectives
    int          n_refvec,               // input:  scalar - number of reference vectors
    int          n_iter,                 // input:  scalar - current iteration
    int          N_iter,                 // input:  scalar - total iterations
    float        alpha,                  // input:  scalar - APD penalty exponent
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    if (pop_size <= 0) return;

    // Host-side scalar precompute: avoid per-thread powf in hot kernel path.
    const float iter_total    = fmaxf(static_cast<float>(N_iter), 1.0f);
    const float progress      = fminf(fmaxf(static_cast<float>(n_iter) / iter_total, 0.0f), 1.0f);
    const float penalty_scale = static_cast<float>(M) * powf(progress, alpha);

    launch_apd_kernel(d_translated_fv, d_theta, d_gamma, d_refvec_idx,
                      d_apd, d_norm, penalty_scale, pop_size, M, n_refvec, exec_stream);
}

// ====================================================================================================================== //
// KERNEL: Gamma Phase 1 - Tile-wise Max-Cosine Reduction
// ====================================================================================================================== //
// PURPOSE:
//   For each query reference vector j, compute tile-local maximum cosine against
//   candidate vectors k in current Y tile, excluding self pair (k == j).
//
// OUTPUT:
//   d_gamma_blkx_max[(blk_y, j)] = partial max cosine for query j in candidate tile blk_y
// ====================================================================================================================== //
template<int WPB = 8, int M = 3, int UNROLL = 4>
__global__ void gamma_phase1_kernel(
    const float* __restrict__ d_refvec,                   // input:  (M, n_refvec) - normalized ref vectors (transposed)
    float*       __restrict__ d_gamma_blkx_max,           // output: (N_btr, n_refvec) - partial max cosine
    int n_refvec                                          // input:  scalar - number of reference vectors
)
{
    constexpr int BS = WPB * WS;

    const int tidx       = threadIdx.x;
    const int query_j    = blockIdx.x * BS + tidx;
    const int rv_start   = blockIdx.y * BS;
    const int rv_end     = min(rv_start + BS, n_refvec);
    const int rv_idx     = rv_start + tidx;
    const int query_mask = bool_to_mask(query_j < n_refvec);
    const int rv_mask    = bool_to_mask(rv_idx < n_refvec);

    const int safe_query_j = min(query_j, n_refvec - 1);
    const int safe_rv_idx  = min(rv_idx, n_refvec - 1);

    __shared__ float s_refvec[M][BS];

    float reg_query[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const float raw_query = d_refvec[m * n_refvec + safe_query_j];
        const float raw_rv    = d_refvec[m * n_refvec + safe_rv_idx];

        reg_query[m]    = bitselect_f32(query_mask, raw_query, 0.f);
        s_refvec[m][tidx] = bitselect_f32(rv_mask, raw_rv, 0.f);
    }
    __syncthreads();

    float my_max = -CUDART_INF_F;

    const int num_rvs_in_tile = rv_end - rv_start;
    const int num_rvs_aligned = (num_rvs_in_tile / UNROLL) * UNROLL;

    // ==========================================================================================
    // Main loop: UNROLL-way dot products + branch-reduced self exclusion via mask selection
    // ==========================================================================================
    for (int r = 0; r < num_rvs_aligned; r += UNROLL) {
        float dot[UNROLL];

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            dot[u] = 0.f;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                dot[u] = fmaf(reg_query[m], s_refvec[m][r + u], dot[u]);
            }
        }

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int glb_k         = rv_start + r + u;
            const int valid_k_mask  = bool_to_mask(glb_k < n_refvec);
            const int not_self_mask = bool_to_mask(glb_k != query_j);
            const int keep_mask     = query_mask & valid_k_mask & not_self_mask;
            const float safe_cos    = bitselect_f32(keep_mask, dot[u], -CUDART_INF_F);
            my_max = fmaxf(my_max, safe_cos);
        }
    }

    // ==========================================================================================
    // Tail loop: process remaining candidates in tile
    // ==========================================================================================
    for (int r = num_rvs_aligned; r < num_rvs_in_tile; ++r) {
        const int glb_k = rv_start + r;

        float dot = 0.f;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            dot = fmaf(reg_query[m], s_refvec[m][r], dot);
        }

        const int valid_k_mask  = bool_to_mask(glb_k < n_refvec);
        const int not_self_mask = bool_to_mask(glb_k != query_j);
        const int keep_mask     = query_mask & valid_k_mask & not_self_mask;
        const float safe_cos    = bitselect_f32(keep_mask, dot, -CUDART_INF_F);
        my_max = fmaxf(my_max, safe_cos);
    }

    if (query_j < n_refvec) {
        const long long out_idx = static_cast<long long>(blockIdx.y) * static_cast<long long>(n_refvec)
                                + static_cast<long long>(query_j);
        d_gamma_blkx_max[out_idx] = my_max;
    }
}

// ========================================================================================================== //
// LAUNCHER: Gamma Phase 1 Kernel Launcher (Template Specialization)
// ========================================================================================================== //
template<int M>
void launch_gamma_phase1_kernel_with_M(
    const float* d_refvec,               // input:  (M, n_refvec) - normalized reference vectors
    float*       d_gamma_blkx_max,       // output: (N_btr, n_refvec) - partial max cosine
    int          n_refvec,               // input:  scalar - number of reference vectors
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    constexpr int WPB_GAMMA = 8;
    constexpr int BLK_SIZE  = WPB_GAMMA * WS;

    const int N_btf = (n_refvec + BLK_SIZE - 1) / BLK_SIZE;  // blocks in X (query vectors)
    const int N_btr = (n_refvec + BLK_SIZE - 1) / BLK_SIZE;  // blocks in Y (candidate tiles)
    dim3 grid_dim(N_btf, N_btr);

    gamma_phase1_kernel<WPB_GAMMA, M><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(
        d_refvec, d_gamma_blkx_max, n_refvec);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================================== //
// LAUNCHER: Gamma Phase 1 Runtime Dispatcher
// ========================================================================================================== //
void launch_gamma_phase1_kernel(
    const float* d_refvec,               // input:  (M, n_refvec) - normalized reference vectors
    float*       d_gamma_blkx_max,       // output: (N_btr, n_refvec) - partial max cosine
    int          n_refvec,               // input:  scalar - number of reference vectors
    int          M,                      // input:  scalar - number of objectives
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    switch (M) {
        case 1: launch_gamma_phase1_kernel_with_M<1>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 2: launch_gamma_phase1_kernel_with_M<2>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 3: launch_gamma_phase1_kernel_with_M<3>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 4: launch_gamma_phase1_kernel_with_M<4>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 5: launch_gamma_phase1_kernel_with_M<5>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 6: launch_gamma_phase1_kernel_with_M<6>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 7: launch_gamma_phase1_kernel_with_M<7>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        case 8: launch_gamma_phase1_kernel_with_M<8>(d_refvec, d_gamma_blkx_max, n_refvec, exec_stream); break;
        default:
            printf("ERROR (Gamma): Unsupported objective count M = %d (max supported: 8)\n", M);
            return;
    }
}

// ====================================================================================================================== //
// KERNEL: Gamma Phase 2 - Final Reduction + acos Conversion
// ====================================================================================================================== //
__global__ void gamma_phase2_kernel(
    const float* __restrict__ d_gamma_blkx_max,         // input:  (N_btr, n_refvec) - partial max cosine
    float*       __restrict__ d_gamma,                  // output: (n_refvec,) - gamma angle
    int n_refvec,                                       // input:  scalar - number of reference vectors
    int N_btr                                           // input:  scalar - number of candidate tiles
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= n_refvec) return;

    float local_max = -CUDART_INF_F;
    for (int blk_y_idx = 0; blk_y_idx < N_btr; ++blk_y_idx) {
        const long long offset = static_cast<long long>(blk_y_idx) * static_cast<long long>(n_refvec)
                               + static_cast<long long>(tidx);
        const float cos_val = d_gamma_blkx_max[offset];
        local_max = fmaxf(local_max, cos_val);
    }

    // Clamp to [0, 1] before acos, consistent with downstream angle kernels.
    const float clamped_cos = fminf(fmaxf(local_max, 0.f), 1.f);
    const float gamma_val   = acosf(clamped_cos);

    const int valid_mask = bool_to_mask(local_max > -CUDART_INF_F);
    d_gamma[tidx] = bitselect_f32(valid_mask, gamma_val, CUDART_PI_F);
}

// ========================================================================================================== //
// LAUNCHER: Gamma Phase 2 Kernel Launcher
// ========================================================================================================== //
void launch_gamma_phase2_kernel(
    const float* d_gamma_blkx_max,       // input:  (N_btr, n_refvec) - partial max cosine
    float*       d_gamma,                // output: (n_refvec,) - gamma angle
    int          n_refvec,               // input:  scalar - number of reference vectors
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    constexpr int WPB_GAMMA = 8;
    constexpr int BLK_SIZE  = WPB_GAMMA * WS;

    const int N_btr = (n_refvec + BLK_SIZE - 1) / BLK_SIZE;
    const int N_btf = (n_refvec + BLK_SIZE - 1) / BLK_SIZE;

    gamma_phase2_kernel<<<N_btf, BLK_SIZE, 0, exec_stream>>>(d_gamma_blkx_max, d_gamma, n_refvec, N_btr);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================================== //
// MAIN FUNCTION: Compute Gamma (retained for optional gamma-based mode)
// ========================================================================================================== //
void rvea::compute_gamma(
    const float* d_refvec,               // input:  (M, n_refvec) - normalized reference vectors
    float*       d_gamma_blkx_max,       // buffer: (N_btr, n_refvec) - partial max cosine
    float*       d_gamma,                // output: (n_refvec,) - gamma angle
    int          M,                      // input:  scalar - number of objectives
    int          n_refvec,               // input:  scalar - number of reference vectors
    cudaStream_t exec_stream             // input:  scalar - CUDA execution stream
)
{
    if (n_refvec <= 0) return;

    // This kernel path is retained for optional gamma-based mode.
    // Current RVEA flow may skip invoking it.
    if (M < 1 || M > 8) {
        printf("ERROR (Gamma): Unsupported objective count M = %d (max supported: 8)\n", M);
        return;
    }
    
    // Single reference vector has no neighbor; define gamma as PI.
    if (n_refvec == 1) {
        const float pi_val = CUDART_PI_F;
        CUDA_CHECK(cudaMemcpyAsync(d_gamma, &pi_val, FLOAT_SIZE, cudaMemcpyHostToDevice, exec_stream));
        return;
    }
    
    // Stage 1: partial max-cosine reduction over candidate tiles.
    launch_gamma_phase1_kernel(d_refvec, d_gamma_blkx_max, n_refvec, M, exec_stream);
    
    // Stage 2: final reduction + acos conversion.
    launch_gamma_phase2_kernel(d_gamma_blkx_max, d_gamma, n_refvec, exec_stream);
}
