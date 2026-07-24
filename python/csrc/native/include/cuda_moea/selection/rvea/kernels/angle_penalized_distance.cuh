#pragma once

namespace rvea {

void compute_apd(
    const float* d_translated_fv,     // input:  (M, pop_size) - z_min-translated fitness (transposed)
    const float* d_theta,             // input:  (pop_size,)   - partition angle per individual (radians)
    const float* d_gamma,             // input:  (n_refvec,)   - minimum reference-vector angle per ref
    const int*   d_refvec_idx,        // input:  (pop_size,)   - assigned reference-vector index
    float*       d_apd,               // output: (pop_size,)   - APD metric
    float*       d_norm,              // output: (pop_size,)   - L2 norm of translated fitness
    int          pop_size,            // input:  scalar        - number of individuals
    int          M,                   // input:  scalar        - number of objectives
    int          n_refvec,            // input:  scalar        - number of reference vectors
    int          n_iter,              // input:  scalar        - current iteration
    int          N_iter,              // input:  scalar        - total iterations
    float        alpha,               // input:  scalar        - APD penalty exponent
    cudaStream_t exec_stream          // input:  scalar        - CUDA execution stream
);

void compute_gamma(
    const float* d_refvec,            // input:  (M, n_refvec) - L2-normalized reference vectors (transposed)
    float*       d_gamma_blkx_max,    // buffer: (N_btr, n_refvec) - partial max cosine across candidate tiles
    float*       d_gamma,             // output: (n_refvec,)   - gamma angle per reference vector (radians)
    int          M,                   // input:  scalar        - number of objectives
    int          n_refvec,            // input:  scalar        - number of reference vectors
    cudaStream_t exec_stream          // input:  scalar        - CUDA execution stream
);

} // namespace rvea
