#pragma once

#include "cuda_moea/problem/dtlz/dtlz_kernels.cuh"
// ===================================================================================================== //
namespace nsga3 {

void merge_pop(
    const PopData& d_pop,     // input: struct - {(N, D)} - parent population
    const PopData& d_off,     // input: struct - {(N, D)} - offspring population
    PopData&       d_mixpop,  // output: struct - {(N_mix, D)} - merged population
    const int D,              // input: scalar - number of decision variables
    const int N,              // input: scalar - population size
    cudaStream_t exec_stream  // input: scalar - CUDA stream
);

void merge_fv(
    const float* d_fv,       // input: (M, N) - float - parent fitness (row-major)
    const float* d_offfv,    // input: (M, N) - float - offspring fitness (row-major)
    float*       d_mixfv,    // output: (M, N_mix) - float - merged fitness (row-major)
    const int M,             // input: scalar - number of objectives
    const int N,             // input: scalar - population size
    cudaStream_t exec_stream // input: scalar - CUDA stream
);

void merge_cv(
    const float* d_cv,       // input: (N,) - float - parent constraint values
    const float* d_offcv,    // input: (N,) - float - offspring constraint values
    float*       d_mixcv,    // output: (N_mix,) - float - merged constraint values
    const int N,             // input: scalar - population size
    cudaStream_t exec_stream // input: scalar - CUDA stream
);

void update_iter_vars(
    const PopData& d_mixpop,       // input: struct - {(N_mix, D)} - merged population
    const float*   d_mixcv,        // input: (N_mix,) - float - merged constraint values
    const float*   d_mixfv,        // input: (M, N_mix) - float - merged fitness (row-major)
    const int*     d_mixfronts,    // input: (N_mix,) - int - merged front indices
    const float*   d_idpts,        // input: (M,) - float - current ideal points
    int*           d_newpop_gidx,  // input: (N,) - int - selection indices (SORTED IN-PLACE)
    PopData&       d_pop,          // output: struct - {(N, D)} - next generation population
    float*         d_cv,           // output: (N,) - float - next gen constraint values
    float*         d_fv,           // output: (M, N) - float - next gen fitness (row-major)
    int*           d_fronts,       // output: (N,) - int - next gen front indices
    float*         d_idpts_last,   // output: (M,) - float - ideal points for next iteration
    const int D,                   // input: scalar - number of decision variables
    const int M,                   // input: scalar - number of objectives
    const int N,                   // input: scalar - population size
    cudaStream_t   exec_stream     // input: scalar - CUDA stream
);

} // namespace nsga3
