#include <math_constants.h>
#include <algorithm>

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"
#include "cuda_moea/core/cuda/cuda_atomops.cuh"

#include "cuda_moea/selection/rvea/kernels/angle_penalized_distance.cuh"
#include "cuda_moea/reference/rvea_reference_adaptation.cuh"

namespace {

constexpr float RANGE_EPS = 1e-12f;
constexpr float NORM_EPS  = 1e-12f;

// ============================================================================================= //
// KERNEL: initialize zmin/zmax
// ============================================================================================= //
__global__ void init_zmin_zmax_kernel(
    float* d_zmin,                  // output: (M,) - set to +INF
    float* d_zmax,                  // output: (M,) - set to -INF
    int M
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        d_zmin[idx] = CUDART_INF_F;
        d_zmax[idx] = -CUDART_INF_F;
    }
}

// ============================================================================================= //
// KERNEL: row-wise zmin/zmax reduction with adaptive vector width
// ============================================================================================= //
template<const int WPB, const int VecLen>
__global__ void zmin_zmax_reduction_kernel(
    const float* __restrict__ d_fv, // input:  (M, N_stride) - row-major
    float*       __restrict__ d_zmin,
    float*       __restrict__ d_zmax,
    int N_active,
    int N_stride
)
{
    const int tarobj_m  = blockIdx.y;
    const int tidx      = threadIdx.x;
    const int vec_idx   = blockIdx.x * blockDim.x + tidx;
    const int col_start = vec_idx * VecLen;
    const int warp_idx  = tidx / WS;
    const int lane_idx  = tidx & (WS - 1);

    __shared__ float smem_min[WPB];
    __shared__ float smem_max[WPB];
    if (tidx < WPB) {
        smem_min[tidx] = CUDART_INF_F;
        smem_max[tidx] = -CUDART_INF_F;
    }
    __syncthreads();

    float local_min = CUDART_INF_F;
    float local_max = -CUDART_INF_F;

    if (col_start < N_active) {
        if constexpr (VecLen == 4) {
            if (col_start + 3 < N_active) {
                const float4 vec = get_vec<float4, float>(&d_fv[tarobj_m * N_stride + col_start]);
                local_min = fminf(fminf(vec.x, vec.y), fminf(vec.z, vec.w));
                local_max = fmaxf(fmaxf(vec.x, vec.y), fmaxf(vec.z, vec.w));
            } else {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int col = col_start + u;
                    if (col < N_active) {
                        const float v = d_fv[tarobj_m * N_stride + col];
                        local_min = fminf(local_min, v);
                        local_max = fmaxf(local_max, v);
                    }
                }
            }
        } else if constexpr (VecLen == 2) {
            if (col_start + 1 < N_active) {
                const float2 vec = get_vec<float2, float>(&d_fv[tarobj_m * N_stride + col_start]);
                local_min = fminf(vec.x, vec.y);
                local_max = fmaxf(vec.x, vec.y);
            } else {
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int col = col_start + u;
                    if (col < N_active) {
                        const float v = d_fv[tarobj_m * N_stride + col];
                        local_min = fminf(local_min, v);
                        local_max = fmaxf(local_max, v);
                    }
                }
            }
        } else {
            const float v = d_fv[tarobj_m * N_stride + col_start];
            local_min = v;
            local_max = v;
        }
    }

    warp_reduce_min_f32<WS>(local_min);
    warp_reduce_max_f32<WS>(local_max);
    if (lane_idx == 0) {
        smem_min[warp_idx] = local_min;
        smem_max[warp_idx] = local_max;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_min = (lane_idx < WPB) ? smem_min[lane_idx] : CUDART_INF_F;
        float block_max = (lane_idx < WPB) ? smem_max[lane_idx] : -CUDART_INF_F;
        warp_reduce_min_f32<WS>(block_min);
        warp_reduce_max_f32<WS>(block_max);
        if (lane_idx == 0) {
            atomicMinFloat(&d_zmin[tarobj_m], block_min);
            atomicMaxFloat(&d_zmax[tarobj_m], block_max);
        }
    }
}

// ============================================================================================= //
// KERNEL: adapt refvecs by objective range and normalize per column
// ============================================================================================= //
template<int M>
__global__ void adapt_refvec_scale_normalize_kernel(
    const float* __restrict__ d_nrps_init, // input:  (M, N_ref)
    const float* __restrict__ d_zmin,      // input:  (M,)
    const float* __restrict__ d_zmax,      // input:  (M,)
    float*       __restrict__ d_nrps,      // output: (M, N_ref)
    int N_ref
)
{
    __shared__ float s_range[M];

    if (threadIdx.x < M) {
        const int m = threadIdx.x;
        float range = d_zmax[m] - d_zmin[m];
        if (!isfinite(range) || range < RANGE_EPS) {
            range = RANGE_EPS;
        }
        s_range[m] = range;
    }
    __syncthreads();

    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < N_ref;
         j += gridDim.x * blockDim.x) {

        float base_reg[M];
        float scaled_reg[M];
        float base_norm_sq = 0.f;
        float scaled_norm_sq = 0.f;

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float base_val = d_nrps_init[m * N_ref + j];
            if (!isfinite(base_val)) base_val = 0.f;
            base_reg[m] = base_val;
            base_norm_sq = fmaf(base_val, base_val, base_norm_sq);

            float scaled_val = base_val * s_range[m];
            if (!isfinite(scaled_val)) scaled_val = 0.f;
            scaled_reg[m] = scaled_val;
            scaled_norm_sq = fmaf(scaled_val, scaled_val, scaled_norm_sq);
        }

        const bool use_scaled = isfinite(scaled_norm_sq) && (scaled_norm_sq > NORM_EPS);
        const bool use_base   = isfinite(base_norm_sq)   && (base_norm_sq > NORM_EPS);

        const float inv_scaled = use_scaled ? rsqrtf(scaled_norm_sq) : 0.f;
        const float inv_base   = use_base   ? rsqrtf(base_norm_sq)   : 0.f;

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float out_val = 0.f;
            if (use_scaled) {
                out_val = scaled_reg[m] * inv_scaled;
            } else if (use_base) {
                out_val = base_reg[m] * inv_base;
            } else {
                out_val = (m == 0) ? 1.f : 0.f;
            }
            d_nrps[m * N_ref + j] = out_val;
        }
    }
}

template<int M>
void launch_adapt_refvec_scale_normalize_kernel_with_M(
    const float* d_nrps_init,
    const float* d_zmin,
    const float* d_zmax,
    float*       d_nrps,
    int          N_ref,
    cudaStream_t exec_stream
)
{
    constexpr int BLK_SIZE = 256;
    const int grid = (N_ref + BLK_SIZE - 1) / BLK_SIZE;

    adapt_refvec_scale_normalize_kernel<M><<<grid, BLK_SIZE, 0, exec_stream>>>(
        d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref);
    CUDA_CHECK(cudaGetLastError());
}

void launch_adapt_refvec_scale_normalize_kernel(
    const float* d_nrps_init,
    const float* d_zmin,
    const float* d_zmax,
    float*       d_nrps,
    int          N_ref,
    int          M,
    cudaStream_t exec_stream
)
{
    switch (M) {
        case 1: launch_adapt_refvec_scale_normalize_kernel_with_M<1>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 2: launch_adapt_refvec_scale_normalize_kernel_with_M<2>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 3: launch_adapt_refvec_scale_normalize_kernel_with_M<3>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 4: launch_adapt_refvec_scale_normalize_kernel_with_M<4>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 5: launch_adapt_refvec_scale_normalize_kernel_with_M<5>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 6: launch_adapt_refvec_scale_normalize_kernel_with_M<6>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 7: launch_adapt_refvec_scale_normalize_kernel_with_M<7>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        case 8: launch_adapt_refvec_scale_normalize_kernel_with_M<8>(d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, exec_stream); break;
        default:
            printf("ERROR (RVEA Adapt): Unsupported objective count M = %d (max supported: 8)\n", M);
            return;
    }
}

} // namespace

void rvea::compute_zmin_zmax(
    const float* d_fv_next,         // input:  (M, N_stride)
    float*       d_zmin,            // output: (M,)
    float*       d_zmax,            // output: (M,)
    int          M,                 // input:  scalar
    int          N_active,          // input:  scalar - active column count
    int          N_stride,          // input:  scalar - row stride in d_fv
    cudaStream_t exec_stream        // input:  scalar
)
{
    if (M <= 0 || N_active <= 0) return;

    constexpr int INIT_BLK = 256;
    const int init_grid = (M + INIT_BLK - 1) / INIT_BLK;
    init_zmin_zmax_kernel<<<init_grid, INIT_BLK, 0, exec_stream>>>(d_zmin, d_zmax, M);
    CUDA_CHECK(cudaGetLastError());

    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;

    // Vectorization requires BOTH N_active (element count) and N_stride (row stride)
    // to be aligned, since row starts are at tarobj_m * N_stride.
    int n_vec = N_active;
    int vec_len = 1;
    if (((N_active & 3) == 0) && ((N_stride & 3) == 0)) {
        n_vec = N_active >> 2;
        vec_len = 4;
    } else if (((N_active & 1) == 0) && ((N_stride & 1) == 0)) {
        n_vec = N_active >> 1;
        vec_len = 2;
    }

    const int grid_x = (n_vec + BLK_SIZE - 1) / BLK_SIZE;
    dim3 grid_dim(grid_x, M);

    if (vec_len == 4) {
        zmin_zmax_reduction_kernel<WPB, 4><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv_next, d_zmin, d_zmax, N_active, N_stride);
    } else if (vec_len == 2) {
        zmin_zmax_reduction_kernel<WPB, 2><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv_next, d_zmin, d_zmax, N_active, N_stride);
    } else {
        zmin_zmax_reduction_kernel<WPB, 1><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv_next, d_zmin, d_zmax, N_active, N_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rvea::adapt_refvecs(
    const float* d_nrps_init,       // input:  (M, N_ref)
    const float* d_fv_next,         // input:  (M, N_stride)
    float*       d_nrps,            // output: (M, N_ref)
    float*       d_zmin,            // buffer: (M,)
    float*       d_zmax,            // buffer: (M,)
    int          n_iter,            // input:  scalar
    int          N_iter,            // input:  scalar
    float        fr,                // input:  scalar
    float&       next_fr_trigger,   // update: (1,)
    // float*       d_gamma_blkx_max,  // buffer: (N_btr_gamma, N_ref)
    // float*       d_gamma,           // output: (N_ref,)
    int          M,                 // input:  scalar
    int          N_ref,             // input:  scalar
    int          N_active,          // input:  scalar - active column count
    int          N_stride,          // input:  scalar - row stride in d_fv
    cudaStream_t exec_stream        // input:  scalar
)
{
    const float progress = static_cast<float>(n_iter + 1) / static_cast<float>(std::max(1, N_iter));
    if (fr <= 0.0f || progress < next_fr_trigger) return;
    next_fr_trigger += fr;

    if (N_ref <= 0 || N_active <= 0) return;
    if (M < 1 || M > 8) {
        printf("ERROR (RVEA Adapt): Unsupported objective count M = %d (max supported: 8)\n", M);
        return;
    }

    // Stage 1: per-objective extrema from selected next population.
    compute_zmin_zmax(d_fv_next, d_zmin, d_zmax, M, N_active, N_stride, exec_stream);

    // Stage 2: scale V0 by objective ranges and re-normalize columns -> Vt.
    launch_adapt_refvec_scale_normalize_kernel(
        d_nrps_init, d_zmin, d_zmax, d_nrps, N_ref, M, exec_stream);

    // // Stage 3: gamma must be refreshed immediately after any reference-vector update.
    // compute_gamma(d_nrps, d_gamma_blkx_max, d_gamma, M, N_ref, exec_stream);
}
