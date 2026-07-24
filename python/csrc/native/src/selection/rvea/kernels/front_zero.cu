#include "cuda_moea/selection/rvea/kernels/front_zero.cuh"

#include <math_constants.h>
#include <algorithm>
#include <cstdio>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace {

template<int M>
constexpr int padded_M() {
    return (M % 2 == 0) ? (M + 1) : M;
}

template<int M = 3, int WPB = 6>
__global__ void constraint_dominatee_bitmask_kernel_active(
    const float* __restrict__ d_cv,              // input:  (N,)                     - constraint values
    const float* __restrict__ d_fv,              // input:  (M, N_stride)            - fitness values
    uint32_t*    __restrict__ d_dominatee_mask,  // output: (N_active, N_tiles)      - dominatee bitmask
    int          N_active,                       // input:  scalar                    - number of active individuals
    int          N_stride                        // input:  scalar                    - row stride for d_fv
)
{
    constexpr uint32_t FMASK = 0xFFFFFFFFu;
    constexpr int M_PAD = padded_M<M>();

    const int N_tiles = (N_active + WS - 1) / WS;
    const int TOT_TILES = N_tiles * N_tiles;

    const int tidx = threadIdx.x;
    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    __shared__ float    s_tiles_p[WPB][WS][M_PAD];
    __shared__ float    s_tiles_q[WPB][WS][M_PAD];
    __shared__ uint32_t s_bitmask[WPB][WS];

    const int tile_base = blockIdx.x * WPB + warp_idx;
    const int stride = gridDim.x * WPB;

    for (int tile_idx = tile_base; tile_idx < TOT_TILES; tile_idx += stride) {
        const int tile_row = tile_idx / N_tiles;
        const int tile_col = tile_idx % N_tiles;
        const int tile_p_base = tile_row * WS;
        const int tile_q_base = tile_col * WS;

        const int glb_p = tile_p_base + lane_idx;
        const int glb_q = tile_q_base + lane_idx;

        const float cv_q = (glb_q < N_active) ? __ldg(&d_cv[glb_q]) : CUDART_INF_F;
        const float cv_p_reg = (glb_p < N_active) ? __ldg(&d_cv[glb_p]) : CUDART_INF_F;

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            s_tiles_p[warp_idx][lane_idx][m] =
                (glb_p < N_active) ? __ldg(&d_fv[m * N_stride + glb_p]) : CUDART_INF_F;
            s_tiles_q[warp_idx][lane_idx][m] =
                (glb_q < N_active) ? __ldg(&d_fv[m * N_stride + glb_q]) : CUDART_INF_F;
        }
        __syncwarp();

        const bool q_feas = (cv_q == FEASIBLE_CV);

        #pragma unroll
        for (int bmsk_idx = 0; bmsk_idx < WS; ++bmsk_idx) {
            const int glb_row = tile_p_base + bmsk_idx;
            const int glb_col = tile_q_base + lane_idx;

            const bool lane_valid = (glb_row < N_active) && (glb_col < N_active) && (glb_row != glb_col);

            const float cv_p = __shfl_sync(FMASK, cv_p_reg, bmsk_idx);
            const bool p_feas = (cv_p == FEASIBLE_CV);

            const bool fs_dom = q_feas && !p_feas;
            const bool cv_dom = !q_feas && !p_feas && (cv_q < cv_p);
            const bool need_fv = q_feas && p_feas;

            bool lane_nw = lane_valid;
            bool lane_sb = false;

            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float pv = s_tiles_p[warp_idx][bmsk_idx][m];
                const float qv = s_tiles_q[warp_idx][lane_idx][m];
                const bool nw_step = (qv <= pv);
                const bool sb_step = (qv < pv);
                lane_nw &= (nw_step || !need_fv);
                lane_sb |= (sb_step && need_fv);
            }

            const bool fv_dom = need_fv && lane_nw && lane_sb;
            const bool lane_dom = lane_valid && (fs_dom || cv_dom || fv_dom);

            const uint32_t q_dom_p = __ballot_sync(FMASK, lane_dom);
            if (lane_idx == 0) {
                s_bitmask[warp_idx][bmsk_idx] = q_dom_p;
            }
        }

        const int glb_row = tile_p_base + lane_idx;
        if (glb_row < N_active) {
            d_dominatee_mask[glb_row * N_tiles + tile_col] = s_bitmask[warp_idx][lane_idx];
        }
    }
}

void compute_dominatee_bitmask_active(
    const float* d_cv,                // input:  (N,)                - constraint values
    const float* d_fv,                // input:  (M, N_stride)       - fitness values
    uint32_t*    d_dominatee_mask,    // output: (N_active, N_tiles) - dominatee bitmask
    int          N_active,            // input:  scalar              - number of active individuals
    int          N_stride,            // input:  scalar              - row stride for d_fv
    int          M,                   // input:  scalar              - number of objectives
    cudaStream_t exec_stream          // input:  cudastream_t        - CUDA stream
)
{
    if (N_active <= 0) return;

    constexpr int WPB = 6;
    const int BLK_SIZE = WPB * WS;
    const int N_tiles = (N_active + WS - 1) / WS;
    const int TOT_TILES = N_tiles * N_tiles;
    const int GRID_SIZE = (TOT_TILES + WPB - 1) / WPB;
    int APPLIED_GRID_SIZE = std::min(GRID_SIZE, SM_COUNT * BLK_SIZE);
    APPLIED_GRID_SIZE = std::max(1, APPLIED_GRID_SIZE);

    #define LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL(M_VAL) \
        constraint_dominatee_bitmask_kernel_active<M_VAL, WPB> \
            <<<APPLIED_GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(d_cv, d_fv, d_dominatee_mask, N_active, N_stride)

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
            printf("ERROR (RVEA frontzero): Unsupported objective count M = %d (max supported: 8)\n", M);
            return;
    }
    #undef LAUNCH_CONSTRAINT_DOMINATEE_BITMASK_KERNEL

    CUDA_CHECK(cudaGetLastError());
}

template<unsigned int WPB>
__global__ void count_dominatee_from_bitmask_kernel_active(
    const uint32_t* __restrict__ d_dominatee_mask,  // input:  (N_active, N_tiles) - dominatee bitmask
    int*            __restrict__ d_dominatee,       // output: (N_active,)          - dominatee count
    int             N_active,                        // input:  scalar               - number of active individuals
    int             N_tiles                          // input:  scalar               - bitmask words per row
)
{
    constexpr unsigned int BLK_SIZE = WPB * WS;

    const int row = blockIdx.x;
    if (row >= N_active) return;

    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(N_tiles);
    int local_sum = 0;

    const bool is_aligned = ((row_offset & 1) == 0);
    if (is_aligned && N_tiles >= 2) {
        const uint2* row_ptr_u2 = reinterpret_cast<const uint2*>(d_dominatee_mask + row_offset);
        const int u2_count = N_tiles / 2;

        for (int i = threadIdx.x; i < u2_count; i += BLK_SIZE) {
            const uint2 data = row_ptr_u2[i];
            local_sum += __popc(data.x) + __popc(data.y);
        }
        if ((N_tiles & 1) && threadIdx.x == 0) {
            local_sum += __popc(d_dominatee_mask[row_offset + N_tiles - 1]);
        }
    } else {
        for (int i = threadIdx.x; i < N_tiles; i += BLK_SIZE) {
            local_sum += __popc(d_dominatee_mask[row_offset + i]);
        }
    }

    #pragma unroll
    for (int offset = WS >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }

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

void count_dominatee_from_bitmask_active(
    const uint32_t* d_dominatee_mask,  // input:  (N_active, N_tiles) - dominatee bitmask
    int*            d_dominatee,       // output: (N_active,)          - dominatee count
    int             N_active,          // input:  scalar               - number of active individuals
    cudaStream_t    exec_stream        // input:  cudastream_t         - CUDA stream
)
{
    if (N_active <= 0) return;

    const int N_tiles = (N_active + WS - 1) / WS;
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;

    count_dominatee_from_bitmask_kernel_active<WPB>
        <<<N_active, BLK_SIZE, 0, exec_stream>>>(d_dominatee_mask, d_dominatee, N_active, N_tiles);
    CUDA_CHECK(cudaGetLastError());
}

struct is_zero_to_one_op {
    __host__ __device__ int operator()(int x) const {
        return (x == 0) ? 1 : 0;
    }
};

struct is_zero_pred {
    __host__ __device__ bool operator()(int x) const {
        return x == 0;
    }
};

} // namespace

void rvea::extract_ndsort_frontzero(
    const float* d_cv,                // input:  (N,)             - constraint values
    const float* d_fv,                // input:  (M, N)           - fitness values
    uint32_t*    d_dominatee_bitmask, // buffer: (N, N_tiles_max) - dominatee bitmask where N_tiles_max = (N + WS - 1) / WS
    int*         d_dominatee,         // buffer: (N,)             - dominatee count
    int*         d_frontzero_idx,     // output: (N,)             - global indices of front-zero individuals
    int&         frontzero_cnt,       // output: scalar           - number of front-zero individuals
    int          N_active,            // input:  scalar           - number of active individuals [0, N_active)
    int          M,                   // input:  scalar           - number of objectives
    int          N,                   // input:  scalar           - population capacity
    cudaMemPool_t exec_pool,          // input:  cudamempool_t    - memory pool (reserved)
    cudaStream_t exec_stream          // input:  cudastream_t     - CUDA stream
)
{
    (void)exec_pool;
    frontzero_cnt = 0;

    if (d_cv == nullptr || d_fv == nullptr || d_dominatee_bitmask == nullptr ||
        d_dominatee == nullptr || d_frontzero_idx == nullptr) {
        printf("ERROR (RVEA frontzero): Null pointer input.\n");
        return;
    }
    if (N <= 0 || N_active < 0 || N_active > N || M < 1 || M > 8) {
        printf("ERROR (RVEA frontzero): Invalid input. N=%d, N_active=%d, M=%d\n", N, N_active, M);
        return;
    }
    if (N_active == 0) {
        return;
    }

    compute_dominatee_bitmask_active(d_cv, d_fv, d_dominatee_bitmask, N_active, N, M, exec_stream);
    count_dominatee_from_bitmask_active(d_dominatee_bitmask, d_dominatee, N_active, exec_stream);

    frontzero_cnt = thrust::transform_reduce(
        thrust::cuda::par.on(exec_stream),
        d_dominatee,
        d_dominatee + N_active,
        is_zero_to_one_op{},
        0,
        thrust::plus<int>());

    thrust::copy_if(
        thrust::cuda::par.on(exec_stream),
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(N_active),
        d_dominatee,
        d_frontzero_idx,
        is_zero_pred{});
    CUDA_CHECK(cudaGetLastError());
}
