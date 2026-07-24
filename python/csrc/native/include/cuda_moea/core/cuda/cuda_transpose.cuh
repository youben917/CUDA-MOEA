#pragma once

#include "cuda_utils.cuh"

// swizzling version (without vectorized memory access)
template<typename T, const int WARP_SIZE_S>
__global__ void mat_transpose_kernel(T *x, T *y, const int row, const int col) {
    __shared__ T tile[WARP_SIZE_S][WARP_SIZE_S];

    const int loc_x = threadIdx.x;
    const int loc_y = threadIdx.y;
    const int glb_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int glb_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (glb_y < row && glb_x < col) {
        tile[loc_y][loc_x ^ loc_y] = x[glb_y * col + glb_x];
    }
    __syncthreads();

    const int glb_x_trans = blockIdx.y * blockDim.y + threadIdx.x;
    const int glb_y_trans = blockIdx.x * blockDim.x + threadIdx.y;

    if (glb_y_trans < col && glb_x_trans < row) {
        y[glb_y_trans * row + glb_x_trans] = tile[loc_x][loc_y ^ loc_x];
    }
}

template<typename T>
T* transpose_matrix(T* d_matrix, int num_rows, int num_cols) //(num_rows, num_cols) tp (num_cols, num_rows)
{
    size_t data_size = sizeof(T);
    T* d_matrix_trans = nullptr;
    CUDA_CHECK(cudaMalloc(&d_matrix_trans, (num_rows * num_cols) * data_size));
    constexpr int BLK_SIZE_T = 16;
    dim3 block_trans(BLK_SIZE_T, BLK_SIZE_T);
    dim3 grid_trans((num_cols + block_trans.x - 1) / block_trans.x, (num_rows + block_trans.y - 1) / block_trans.y);
    mat_transpose_kernel<T, BLK_SIZE_T><<<grid_trans, block_trans>>>(d_matrix, d_matrix_trans, num_rows, num_cols);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
    return d_matrix_trans;
}

// overload
template<typename T>
void transpose_matrix(T* d_matrix, T* d_matrix_trans, int num_rows, int num_cols, cudaStream_t stream)
{
    constexpr int BLK_SIZE_T = 16;
    dim3 block_trans(BLK_SIZE_T, BLK_SIZE_T);
    dim3 grid_trans((num_cols + block_trans.x - 1) / block_trans.x, (num_rows + block_trans.y - 1) / block_trans.y);
    mat_transpose_kernel<T, BLK_SIZE_T><<<grid_trans, block_trans, 0 , stream>>>(d_matrix, d_matrix_trans, num_rows, num_cols);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// // overload
// template<typename T>
// void transpose_matrix(T* d_matrix, T* d_matrix_trans, int num_rows, int num_cols) //(num_rows, num_cols) tp (num_cols, num_rows)
// {
//     constexpr int BLK_SIZE_T = 16;
//     dim3 block_trans(BLK_SIZE_T, BLK_SIZE_T);
//     dim3 grid_trans((num_cols + block_trans.x - 1) / block_trans.x, (num_rows + block_trans.y - 1) / block_trans.y);
//     mat_transpose_kernel<T, BLK_SIZE_T><<<grid_trans, block_trans>>>(d_matrix, d_matrix_trans, num_rows, num_cols);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
// }