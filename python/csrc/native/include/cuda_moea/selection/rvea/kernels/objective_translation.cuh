#pragma once

namespace rvea {

void translate_fv(
    const float* d_fv,              // input:  (M, N_col) - fitness values (transposed)
    float*       d_translated_fv,   // output: (M, N_col) - translated fitness values
    float*       d_zmin,            // buffer: (M,) - per-objective minimums
    int M,                          // input:  scalar - number of objectives
    int N_col,                      // input:  scalar - number of individuals
    cudaStream_t exec_stream        // input:  scalar - CUDA execution stream
);

} // namespace rvea
