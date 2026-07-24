#include <math_constants.h>

#include "cuda_moea/selection/rvea/kernels/objective_translation.cuh"

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"
#include "cuda_moea/core/cuda/cuda_atomops.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////
// Initialization kernel for z_min array.
// Must be called BEFORE zmin_reduction_kernel to avoid cross-block race conditions.
// Parameters:
//   d_zmin: Output array (size M) to be initialized to CUDART_INF_F.
//   M: Number of elements (number of objectives).
__global__ void init_zmin_kernel(
    float* d_zmin,               // output: (M,) - will be set to INF
    const int M                  // input: scalar - number of objectives
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        d_zmin[tid] = CUDART_INF_F;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// Computes the minimum value for each row of the M x N_col matrix d_fv,
// storing results in d_zmin[tarobj_m] for each row (tarobj_m = 0 to M-1).
// Uses vectorized loading (float4 or float2) and block-level reduction with atomic updates.
// Template parameter VecLen controls vector width: 4 for float4, 2 for float2.
// IMPORTANT: d_zmin array MUST be pre-initialized to CUDART_INF_F before calling this kernel.
// Parameters:
//   d_fv: Input matrix (M x N_col, row-major).
//   d_zmin: Output array (size M) for row minimums (must be pre-initialized to INF).
//   N_col: Number of columns (must be multiple of VecLen).
template<const int WPB, const int VecLen>
__global__ void zmin_reduction_kernel(
    const float* __restrict__ d_fv,   // input: (M, N_col) - fitness values
    float* d_zmin,                    // output: (M,) - MUST be pre-initialized to INF
    const int N_col                   // input: scalar - number of individuals (N_col % VecLen == 0)
)
{
    // Select vector type based on VecLen template parameter
    using VecT    = typename std::conditional<VecLen == 4, float4, float2>::type;
    using ScalarT = float;

    const int tarobj_m  = blockIdx.y;
    const int tidx      = threadIdx.x;
    const int col_idx   = blockIdx.x * blockDim.x + tidx;
    const int col_start = col_idx * VecLen;
    const int warp_idx  = tidx / WS;
    const int lane_idx  = tidx & (WS - 1);

    // Initialize shared memory for warp-level partial minimums
    __shared__ ScalarT smem[WPB];
    if (tidx < WPB) smem[tidx] = CUDART_INF_F;
    __syncthreads();

    // Each thread loads VecLen elements via vectorized access using get_vec helper
    ScalarT min_row = CUDART_INF_F;
    if (col_start < N_col) {
        VecT vec = get_vec<VecT, ScalarT>(&d_fv[tarobj_m * N_col + col_start]);

        // Compute minimum within vector (branch-free via constexpr)
        if constexpr (VecLen == 4) {
            min_row = fminf(fminf(vec.x, vec.y), fminf(vec.z, vec.w));
        } else {
            min_row = fminf(vec.x, vec.y);
        }
    }

    // Warp-level reduction
    warp_reduce_min_f32<WS>(min_row);
    if (lane_idx == 0) smem[warp_idx] = min_row;
    __syncthreads();

    // Block-level reduction
    if (warp_idx == 0) {
        float block_min = (lane_idx < WPB) ? smem[lane_idx] : CUDART_INF_F;
        warp_reduce_min_f32<WS>(block_min);
        if (lane_idx == 0) atomicMinFloat(&d_zmin[tarobj_m], block_min);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// Translation kernel: computes f'[m][i] = f[m][i] - z_min[m] for each element.
// Grid: dim3(ceil(N_col/256), M), Block: 256
// Parameters:
//   d_fv: Input matrix (M x N_col) - original fitness values.
//   d_translated_fv: Output matrix (M x N_col) - translated fitness values.
//   d_zmin: Input array (M,) - per-objective minimums.
//   N_col: Number of columns (individuals).
__global__ void translate_fv_kernel(
    const float* __restrict__ d_fv,              // input:  (M, N_col) - fitness values
    float* __restrict__ d_translated_fv,         // output: (M, N_col) - translated fitness values
    const float* __restrict__ d_zmin,            // input:  (M,) - per-objective minimums
    const int N_col                              // input:  scalar - number of individuals
)
{
    const int tarobj_m = blockIdx.y;
    const int col_idx  = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_idx < N_col) {
        const float zmin_m = d_zmin[tarobj_m];
        const int idx = tarobj_m * N_col + col_idx;
        // EvoX applies maximum(obj, 1e-32) after ideal-point translation.
        d_translated_fv[idx] = fmaxf(d_fv[idx] - zmin_m, 1e-32f);
    }
}

// ============================================================================================= //
// Computes z_min-translated fitness values: f'[m][i] = f[m][i] - min_i(f[m][i]).
// This function handles initialization, z_min reduction, and translation in a single call.
// Automatically selects optimal vectorization width based on N_col's alignment
// (float4 if N_col%4==0, else float2).
// Parameters:
//   d_fv: Input matrix (M x N_col) - fitness values in row-major order.
//   d_translated_fv: Output matrix (M x N_col) - translated fitness values.
//   d_zmin: Buffer array (M) - will contain per-objective minimums after call.
//   M: Number of objectives (rows).
//   N_col: Number of individuals (columns), must be multiple of 2.
//   exec_stream: CUDA stream for execution.
void rvea::translate_fv(
    const float* d_fv,              // input:  (M, N_col) - fitness values
    float*       d_translated_fv,   // output: (M, N_col) - translated fitness values
    float*       d_zmin,            // buffer: (M,) - per-objective minimums
    int M,                          // input:  scalar - number of objectives
    int N_col,                      // input:  scalar - number of individuals (N_col % 2 == 0)
    cudaStream_t exec_stream        // input:  scalar - CUDA execution stream
)
{
    // Step 1: Initialize z_min array to INF
    // This kernel completes before the reduction kernel due to same-stream serialization
    init_zmin_kernel<<<1, 32, 0, exec_stream>>>(d_zmin, M);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Z_min reduction kernel with adaptive vectorization
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;

    // Select vectorization width based on N_col's alignment
    if ((N_col & 3) == 0) {
        // N_col is multiple of 4: use float4 for maximum throughput
        const int N4 = N_col >> 2;
        int GRID_SIZE = (N4 + BLK_SIZE - 1) / BLK_SIZE;
        dim3 grid_dim(GRID_SIZE, M);
        zmin_reduction_kernel<WPB, 4><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv, d_zmin, N_col);
    } else {
        // N_col is multiple of 2: use float2 as fallback
        const int N2 = N_col >> 1;
        int GRID_SIZE = (N2 + BLK_SIZE - 1) / BLK_SIZE;
        dim3 grid_dim(GRID_SIZE, M);
        zmin_reduction_kernel<WPB, 2><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv, d_zmin, N_col);
    }
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Translation kernel - f'[m][i] = f[m][i] - z_min[m]
    {
        constexpr int TRANS_BLK = 256;
        int GRID_X = (N_col + TRANS_BLK - 1) / TRANS_BLK;
        dim3 grid_dim(GRID_X, M);
        translate_fv_kernel<<<grid_dim, TRANS_BLK, 0, exec_stream>>>(d_fv, d_translated_fv, d_zmin, N_col);
        CUDA_CHECK(cudaGetLastError());
    }
}
