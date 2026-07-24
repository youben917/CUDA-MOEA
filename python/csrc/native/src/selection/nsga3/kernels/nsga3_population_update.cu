#include "cuda_moea/selection/nsga3/kernels/nsga3_population_update.cuh"

#include <thrust/sort.h>
// ==================================================================================== //
// Vertical Stack Two-Pass Kernel - Single Source Copy
// ==================================================================================== //
__global__ void vstack_twopass_kernel(
    const float* __restrict__ d_src,   // input: (N, D) - d_src
    float* __restrict__ d_dst,         // output: (N_mix, D) - d_dst
    int total_elements,                // input: scalar - N * D
    int offset                         // input: scalar - 0 or N * D
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_elements) {
        d_dst[tid + offset] = d_src[tid];
    }
}
// ==================================================================================== //
void nsga3::merge_pop(
    const PopData& d_pop,
    const PopData& d_off,
    PopData&       d_mixpop,
    const int D,
    const int N,
    cudaStream_t exec_stream
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
// ==================================================================================== //
// Horizontal Stack Kernel for Fitness Matrices
// ==================================================================================== //
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

        d_mixfit[output_row_base + col_idx] = val_fit;
        d_mixfit[output_row_base + N + col_idx] = val_offit;
    }
}
// ==================================================================================== //
void nsga3::merge_fv(
    const float* d_fv,
    const float* d_offfv,
    float*       d_mixfv,
    const int M,
    const int N,
    cudaStream_t exec_stream
)
{
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    const int N_BLKS = (N + BLK_SIZE - 1) / BLK_SIZE;

    matrices_hstack_kernel<<<N_BLKS, BLK_SIZE, 0, exec_stream>>>(d_fv, d_offfv, d_mixfv, M, N);
    CUDA_CHECK(cudaGetLastError());
}
// ==================================================================================== //
// Horizontal Stack Kernel for Constraint Values
// ==================================================================================== //
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
// ==================================================================================== //
void nsga3::merge_cv(
    const float* d_cv,
    const float* d_offcv,
    float*       d_mixcv,
    const int N,
    cudaStream_t exec_stream
)
{
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    const int N_BLKS = (N + BLK_SIZE - 1) / BLK_SIZE;

    constraint_value_hstack_kernel<<<N_BLKS, BLK_SIZE, 0, exec_stream>>>(d_cv, d_offcv, d_mixcv, N);
    CUDA_CHECK(cudaGetLastError());
}
// ==================================================================================== //
// Population Update Kernel - Float4 Vectorized Path (D % 4 == 0)
// ==================================================================================== //
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
// ==================================================================================== //
// Population Update Kernel - Scalar Fallback Path
// ==================================================================================== //
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
// ==================================================================================== //
// Host Function: Population Update Launcher with Smart Dispatch
// ==================================================================================== //
void launch_pop_update_kernel(
    const float* d_mixpop,
    const int*   d_nextpop_idx,
    float*       d_pop,
    int N,
    int D,
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
// ==================================================================================== //
// Fitness, Constraint Values and Fronts Update Kernel
// ==================================================================================== //
__global__ void fit_and_fronts_update_kernel(
    const float* __restrict__ d_mixcv,
    const float* __restrict__ d_mixfit,
    const int*   __restrict__ d_mixfronts,
    const int*   __restrict__ d_nextpop_idx,
    float*       __restrict__ d_cv,
    float*       __restrict__ d_fit,
    int*         __restrict__ d_fronts,
    int N,
    int N_mix,
    int M
)
{
    int glb_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (glb_idx >= N) return;

    int src_idx = d_nextpop_idx[glb_idx];

    // Update fitness values
    #pragma unroll
    for (int m = 0; m < M; m++) {
        d_fit[m * N + glb_idx] = d_mixfit[m * N_mix + src_idx];
    }

    // Update pareto fronts
    d_fronts[glb_idx] = d_mixfronts[src_idx];

    // Update constraint values
    d_cv[glb_idx] = d_mixcv[src_idx];
}
// ==================================================================================== //
void launch_fit_and_fronts_update_kernel(
    const float* d_mixcv,
    const float* d_mixfit,
    const int*   d_mixfronts,
    int*         d_nextpop_idx,
    float*       d_cv,
    float*       d_fit,
    int*         d_fronts,
    int N,
    int N_mix,
    int M,
    cudaStream_t exec_stream
)
{
    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE = (N + BLK_SIZE - 1) / BLK_SIZE;
    fit_and_fronts_update_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_mixcv, d_mixfit, d_mixfronts, d_nextpop_idx, d_cv, d_fit, d_fronts, N, N_mix, M);
    CUDA_CHECK(cudaGetLastError());
}
// ==================================================================================== //
void nsga3::update_iter_vars(
    const PopData& d_mixpop,
    const float*   d_mixcv,
    const float*   d_mixfv,
    const int*     d_mixfronts,
    const float*   d_idpts,
    int*           d_newpop_gidx,
    PopData&       d_pop,
    float*         d_cv,
    float*         d_fv,
    int*           d_fronts,
    float*         d_idpts_last,
    const int D,
    const int M,
    const int N,
    cudaStream_t   exec_stream
)
{
    const int N_mix = 2 * N;

    // Sort d_newpop_gidx for better memory access coalescing
    thrust::sort(thrust::cuda::par.on(exec_stream), d_newpop_gidx, d_newpop_gidx + N);

    CUDA_CHECK(cudaMemsetAsync(d_fronts, -1, N * INT_SIZE, exec_stream));

    // Update population
    launch_pop_update_kernel(d_mixpop.d_pop, d_newpop_gidx, d_pop.d_pop, N, D, exec_stream);

    // Update fitness, constraint values and fronts
    launch_fit_and_fronts_update_kernel(d_mixcv, d_mixfv, d_mixfronts, d_newpop_gidx, d_cv, d_fv, d_fronts, N, N_mix, M, exec_stream);

    // Update ideal points (copy to last)
    CUDA_CHECK(cudaMemcpyAsync(d_idpts_last, d_idpts, M * FLOAT_SIZE, cudaMemcpyDeviceToDevice, exec_stream));
}
