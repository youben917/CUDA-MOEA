#pragma once

#include <cstdint>

#include "cuda_moea/core/cuda/cuda_globals.cuh"

namespace rvea {

void extract_ndsort_frontzero(
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
);

} // namespace rvea
