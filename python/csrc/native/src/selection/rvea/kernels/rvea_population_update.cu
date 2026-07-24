#include "cuda_moea/selection/rvea/kernels/rvea_population_update.cuh"

#include <math_constants.h>
#include <thrust/sort.h>

namespace {

// Vertical Stack Two-Pass Kernel - Single Source Copy
__global__ void vstack_twopass_kernel(
    const float* __restrict__ d_src,  // input: (N, D) - d_src
    float* __restrict__ d_dst,        // output: (N_mix, D) - d_dst
    int total_elements,               // input: scalar - N * D
    int offset                        // input: scalar - 0 or N * D
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_elements) {
        d_dst[tid + offset] = d_src[tid];
    }
}

// Horizontal Stack Kernel for Fitness Matrices
__global__ void matrices_hstack_kernel(
    const float* __restrict__ d_fit,    // input: (M, N) - d_fit
    const float* __restrict__ d_offit,  // input: (M, N) - d_offit
    float* __restrict__       d_mixfit, // output: (M, N_mix) - merged fitness matrix
    int                       M,        // input: scalar - number of objectives
    int                       N         // input: scalar - population size
)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= N) return;

    for (int m = 0; m < M; m++) {
        int input_idx = m * N + col_idx;

        float val_fit   = d_fit[input_idx];
        float val_offit = d_offit[input_idx];

        int output_row_base = m * (2 * N);

        d_mixfit[output_row_base + col_idx]     = val_fit;
        d_mixfit[output_row_base + N + col_idx] = val_offit;
    }
}

// Horizontal Stack Kernel for Constraint Values
__global__ void constraint_value_hstack_kernel(
    const float* __restrict__ d_cv,
    const float* __restrict__ d_offcv,
    float* __restrict__       d_mixcv,
    int                       N
)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    d_mixcv[tidx]     = d_cv[tidx];
    d_mixcv[N + tidx] = d_offcv[tidx];
}

// Population Update Kernel - Float4 Vectorized Path (D % 4 == 0)
__global__ void pop_update_float4_kernel(
    const float* __restrict__ d_mixpop,
    const int*   __restrict__ d_nextpop_idx,
    float*       __restrict__ d_pop,
    int N,
    int D
)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    const int src_row = d_nextpop_idx[row_idx];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int D4 = D >> 2;

    const int src_base = src_row * D;
    const int dst_base = row_idx * D;

    const float4* __restrict__ src_ptr4 = reinterpret_cast<const float4*>(d_mixpop + src_base);
    float4*       __restrict__ dst_ptr4 = reinterpret_cast<float4*>(d_pop + dst_base);

    #pragma unroll 4
    for (int col4 = tid; col4 < D4; col4 += stride) {
        dst_ptr4[col4] = src_ptr4[col4];
    }
}

// Population Update Kernel - Scalar Fallback Path
__global__ void pop_update_scalar_kernel(
    const float* __restrict__ d_mixpop,
    const int*   __restrict__ d_nextpop_idx,
    float*       __restrict__ d_pop,
    int N,
    int D
)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    const int src_row = d_nextpop_idx[row_idx];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int src_base = src_row * D;
    const int dst_base = row_idx * D;

    for (int col = tid; col < D; col += stride) {
        d_pop[dst_base + col] = d_mixpop[src_base + col];
    }
}

void launch_pop_update_kernel(
    const float* d_mixpop,
    const int*   d_nextpop_idx,
    float*       d_pop,
    int          N,
    int          D,
    cudaStream_t exec_stream
)
{
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = N;

    if ((D & 3) == 0) {
        pop_update_float4_kernel<<<grid_size, BLOCK_SIZE, 0, exec_stream>>>(d_mixpop, d_nextpop_idx, d_pop, N, D);
    } else {
        pop_update_scalar_kernel<<<grid_size, BLOCK_SIZE, 0, exec_stream>>>(d_mixpop, d_nextpop_idx, d_pop, N, D);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Fitness + constraint values update kernel (RVEA variant: no fronts gather).
__global__ void fv_cv_update_kernel(
    const float* __restrict__ d_mixcv,
    const float* __restrict__ d_mixfv,
    const int*   __restrict__ d_nextpop_idx,
    float*       __restrict__ d_cv,
    float*       __restrict__ d_fv,
    int N_active,
    int N_mix,
    int N_stride,
    int M
)
{
    int glb_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (glb_idx >= N_active) return;

    const int src_idx = d_nextpop_idx[glb_idx];

    #pragma unroll
    for (int m = 0; m < M; m++) {
        d_fv[m * N_stride + glb_idx] = d_mixfv[m * N_mix + src_idx];
    }

    d_cv[glb_idx] = d_mixcv[src_idx];
}

void launch_fv_cv_update_kernel(
    const float* d_mixcv,
    const float* d_mixfv,
    int*         d_nextpop_idx,
    float*       d_cv,
    float*       d_fv,
    int          N_active,
    int          N_mix,
    int          N_stride,
    int          M,
    cudaStream_t exec_stream
)
{
    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE = (N_active + BLK_SIZE - 1) / BLK_SIZE;
    fv_cv_update_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_mixcv, d_mixfv, d_nextpop_idx, d_cv, d_fv, N_active, N_mix, N_stride, M);
    CUDA_CHECK(cudaGetLastError());
}

constexpr float SENTINEL_FV = 1e15f;
constexpr float SENTINEL_CV = 1e15f;

__global__ void write_sentinels_kernel(
    float* __restrict__ d_pop,
    float* __restrict__ d_fv,
    float* __restrict__ d_cv,
    int N_active, int N, int D, int N_stride, int M
)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int dead_idx = N_active + tidx;
    if (dead_idx >= N) return;

    d_cv[dead_idx] = SENTINEL_CV;

    for (int d = 0; d < D; ++d) {
        d_pop[dead_idx * D + d] = CUDART_NAN_F;
    }

    for (int m = 0; m < M; m++) {
        d_fv[m * N_stride + dead_idx] = SENTINEL_FV;
    }
}

} // namespace

void rvea::merge_pop(
    const PopData& d_pop,     // input: struct - {(N, D)} - parent population
    const PopData& d_off,     // input: struct - {(N, D)} - offspring population
    PopData&       d_mixpop,  // output: struct - {(N_mix, D)} - merged population
    const int      D,         // input: scalar - number of decision variables
    const int      N,         // input: scalar - population size
    cudaStream_t   exec_stream
)
{
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    const int tot_eles = N * D;
    const int N_BLKS   = (tot_eles + BLK_SIZE - 1) / BLK_SIZE;

    // Pass 1: Copy d_pop to first half
    vstack_twopass_kernel<<<N_BLKS, BLK_SIZE, 0, exec_stream>>>(d_pop.d_pop, d_mixpop.d_pop, tot_eles, 0);
    // Pass 2: Copy d_off to second half
    vstack_twopass_kernel<<<N_BLKS, BLK_SIZE, 0, exec_stream>>>(d_off.d_pop, d_mixpop.d_pop, tot_eles, tot_eles);
    CUDA_CHECK(cudaGetLastError());
}

void rvea::merge_fv(
    const float* d_fv,       // input: (M, N) - parent fitness (row-major)
    const float* d_offfv,    // input: (M, N) - offspring fitness (row-major)
    float*       d_mixfv,    // output: (M, N_mix) - merged fitness (row-major)
    const int    M,          // input: scalar - number of objectives
    const int    N,          // input: scalar - population size
    cudaStream_t exec_stream
)
{
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    const int N_BLKS = (N + BLK_SIZE - 1) / BLK_SIZE;

    matrices_hstack_kernel<<<N_BLKS, BLK_SIZE, 0, exec_stream>>>(d_fv, d_offfv, d_mixfv, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void rvea::merge_cv(
    const float* d_cv,       // input: (N,) - parent constraint values
    const float* d_offcv,    // input: (N,) - offspring constraint values
    float*       d_mixcv,    // output: (N_mix,) - merged constraint values
    const int    N,          // input: scalar - population size
    cudaStream_t exec_stream
)
{
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    const int N_BLKS = (N + BLK_SIZE - 1) / BLK_SIZE;

    constraint_value_hstack_kernel<<<N_BLKS, BLK_SIZE, 0, exec_stream>>>(d_cv, d_offcv, d_mixcv, N);
    CUDA_CHECK(cudaGetLastError());
}

void rvea::update_iter_vars(
    const PopData& d_mixpop,       // input: struct - {(N_mix, D)} - merged population
    const float*   d_mixcv,        // input: (N_mix,) - merged constraint values
    const float*   d_mixfv,        // input: (M, N_mix) - merged fitness (row-major)
    int*           d_newpop_gidx,  // input: (N_active,) - selection indices (SORTED IN-PLACE)
    PopData&       d_pop,          // output: struct - {(N, D)} - next generation population
    float*         d_cv,           // output: (N,) - next generation constraint values
    float*         d_fv,           // output: (M, N) - next generation fitness (row-major)
    const int      D,              // input: scalar - number of decision variables
    const int      M,              // input: scalar - number of objectives
    const int      N_active,       // input: scalar - number of winners to gather
    const int      N,              // input: scalar - total population size (row stride)
    cudaStream_t   exec_stream
)
{
    const int N_mix = 2 * N;

    // Sort only N_active indices for better memory coalescing in gather kernels.
    thrust::sort(thrust::cuda::par.on(exec_stream), d_newpop_gidx, d_newpop_gidx + N_active);

    launch_pop_update_kernel(d_mixpop.d_pop, d_newpop_gidx, d_pop.d_pop, N_active, D, exec_stream);
    launch_fv_cv_update_kernel(d_mixcv, d_mixfv, d_newpop_gidx, d_cv, d_fv, N_active, N_mix, N, M, exec_stream);
    const int N_dead = N - N_active;
    if (N_dead <= 0) return;

    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE = (N_dead + BLK_SIZE - 1) / BLK_SIZE;
    write_sentinels_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_pop.d_pop, d_fv, d_cv, N_active, N, D, N, M);
    CUDA_CHECK(cudaGetLastError());
}
