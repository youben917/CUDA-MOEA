#pragma once

#include "cuda_moea/problem/dtlz/dtlz_kernels.cuh"

namespace rvea {

void merge_pop(
    const PopData& d_pop,     // input: struct - {(N, D)} - parent population
    const PopData& d_off,     // input: struct - {(N, D)} - offspring population
    PopData&       d_mixpop,  // output: struct - {(N_mix, D)} - merged population
    const int      D,         // input: scalar - number of decision variables
    const int      N,         // input: scalar - population size
    cudaStream_t   exec_stream
);

void merge_fv(
    const float* d_fv,       // input: (M, N) - parent fitness (row-major)
    const float* d_offfv,    // input: (M, N) - offspring fitness (row-major)
    float*       d_mixfv,    // output: (M, N_mix) - merged fitness (row-major)
    const int    M,          // input: scalar - number of objectives
    const int    N,          // input: scalar - population size
    cudaStream_t exec_stream
);

void merge_cv(
    const float* d_cv,       // input: (N,) - parent constraint values
    const float* d_offcv,    // input: (N,) - offspring constraint values
    float*       d_mixcv,    // output: (N_mix,) - merged constraint values
    const int    N,          // input: scalar - population size
    cudaStream_t exec_stream
);

void update_iter_vars(
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
);

} // namespace rvea
