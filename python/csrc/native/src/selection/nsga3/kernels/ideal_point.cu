#include <math_constants.h>

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"
#include "cuda_moea/core/cuda/cuda_atomops.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////
// Initialization kernel for ideal points array.
// Must be called BEFORE idpts_generation_kernel to avoid cross-block race conditions.
// Parameters:
//   idpts: Output array (size M) to be initialized to CUDART_INF_F.
//   M: Number of elements (number of objectives).
__global__ void init_idpts_kernel(
    float* idpts,                // output: (M,) - will be set to INF
    const int M                  // input: scalar - number of objectives
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        idpts[tid] = CUDART_INF_F;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// Computes the minimum value for each row of the M x N matrix d_fv,
// storing results in idpts[tarobj_m] for each row (tarobj_m = 0 to M-1).
// Uses vectorized loading (float4 or float2) and block-level reduction with atomic updates.
// Template parameter VecLen controls vector width: 4 for float4, 2 for float2.
// IMPORTANT: idpts array MUST be pre-initialized to CUDART_INF_F before calling this kernel.
// Parameters:
//   d_fv: Input matrix (M x N, row-major).
//   idpts: Output array (size M) for row minimums (must be pre-initialized to INF).
//   N: Number of columns (must be multiple of VecLen).
template<const int WPB, const int VecLen>
__global__ void idpts_generation_kernel( 
    const float* __restrict__ d_fv,   // input: (M, N) - d_fv
    float* idpts,                     // output: (M,) - MUST be pre-initialized to INF
    const int N                       // input: scalar - population size (N % VecLen == 0)
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

    // NOTE: Initialization removed from here to avoid cross-block race condition.
    // The idpts array must be initialized externally via init_idpts_kernel.

    // Initialize shared memory for warp-level partial minimums
    __shared__ ScalarT smem[WPB];
    if (tidx < WPB) smem[tidx] = CUDART_INF_F;
    __syncthreads();

    // Each thread loads VecLen elements via vectorized access using get_vec helper
    ScalarT min_row = CUDART_INF_F;
    if (col_start < N) {
        VecT vec = get_vec<VecT, ScalarT>(&d_fv[tarobj_m * N + col_start]);
        
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
        if (lane_idx == 0) atomicMinFloat(&idpts[tarobj_m], block_min);
    }
}

// ============================================================================================= //
// Computes ideal points (row-wise minimums) for the fitness value matrix.
// This function handles proper initialization and automatically selects optimal
// vectorization width based on N's alignment (float4 if N%4==0, else float2).
// Parameters:
//   d_fv: Input matrix (M x N) - fitness values in row-major order.
//   d_idpts: Output array (M) - will contain minimum value of each row.
//   M: Number of objectives (rows).
//   N: Population size (columns), must be multiple of 2.
//   exec_stream: CUDA stream for execution.
void get_idpts(
    const float* d_fv,           // input: (M, N) - fitness values
    float* d_idpts,              // output: (M,) - ideal points
    const int M,                 // input: scalar - number of objectives
    const int N,                 // input: scalar - population size (N % 2 == 0)                   
    cudaStream_t exec_stream     // input: scalar - CUDA execution stream
)
{  
    // Step 1: Initialize output array to INF
    // This kernel completes before the main kernel due to same-stream serialization
    init_idpts_kernel<<<1, 32, 0, exec_stream>>>(d_idpts, M);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Main reduction kernel with adaptive vectorization
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    
    // Select vectorization width based on N's alignment
    if ((N & 3) == 0) {
        // N is multiple of 4: use float4 for maximum throughput
        const int N4 = N >> 2;
        int GRID_SIZE = (N4 + BLK_SIZE - 1) / BLK_SIZE;
        dim3 grid_dim(GRID_SIZE, M);
        idpts_generation_kernel<WPB, 4><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv, d_idpts, N);
    } else {
        // N is multiple of 2: use float2 as fallback
        const int N2 = N >> 1;
        int GRID_SIZE = (N2 + BLK_SIZE - 1) / BLK_SIZE;
        dim3 grid_dim(GRID_SIZE, M);
        idpts_generation_kernel<WPB, 2><<<grid_dim, BLK_SIZE, 0, exec_stream>>>(d_fv, d_idpts, N);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================================= //
// Updates ideal points by taking element-wise minimum of previous and new values.
// Parameters:
//   d_idpts_last: Previous ideal points (M,) - also updated in-place at the end.
//   d_idpts_off: New candidate ideal points from offspring (M,).
//   d_idpts: Output array (M,) for updated ideal points.
//   M: Number of objectives.
__global__ void idealpoints_update_kernel(
    const float* d_idpts_last,           // input: (M,) - ideal points of last iteration
    const float* d_idpts_off,            // input: (M,) - ideal points of offspring 
    float* d_idpts,                      // output: (M,) - ideal points of current iteration
    const int M                          // input: scalar - number of objectives
)  
{
    int tid = threadIdx.x;
    if (tid < M) {
        d_idpts[tid] = fminf(d_idpts_last[tid], d_idpts_off[tid]);
    }
}

// ============================================================================================= //
// Host wrapper for updating ideal points.
// Computes element-wise minimum and also updates d_idpts_last for next iteration.
void update_idpts(
    float* d_idpts_last,         // update: (M,) - ideal points of last iteration
    const float* d_idpts_off,    // input: (M,) - ideal points of offspring 
    float* d_idpts,              // output: (M,) - ideal points of current iteration
    const int M,                 // input: scalar - number of objectives
    cudaStream_t exec_stream     // input: scalar - CUDA execution stream
)
{   // Note that M <= 8
    idealpoints_update_kernel<<<1, 32, 0, exec_stream>>>(d_idpts_last, d_idpts_off, d_idpts, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(d_idpts_last, d_idpts, M * FLOAT_SIZE, cudaMemcpyDeviceToDevice, exec_stream));
}
