#pragma once

#include "cuda_runtime.h"

namespace rvea {

void compute_zmin_zmax(
    const float* d_fv_next,         // input:  (M, N_stride) - next-population fitness
    float*       d_zmin,            // output: (M,)          - per-objective minimum
    float*       d_zmax,            // output: (M,)          - per-objective maximum
    int          M,                 // input:  scalar        - objective count
    int          N_active,          // input:  scalar        - active column count (loop bound)
    int          N_stride,          // input:  scalar        - row stride in d_fv
    cudaStream_t exec_stream        // input:  scalar        - CUDA stream
);

void adapt_refvecs(
    const float* d_nrps_init,       // input:  (M, N_ref) - immutable initial reference vectors V0
    const float* d_fv_next,         // input:  (M, N_stride) - selected next-population fitness
    float*       d_nrps,            // output: (M, N_ref) - adapted reference vectors Vt
    float*       d_zmin,            // buffer: (M,)    - caller-owned zmin buffer
    float*       d_zmax,            // buffer: (M,)    - caller-owned zmax buffer
    int          n_iter,            // input:  scalar     - current iteration index
    int          N_iter,            // input:  scalar     - total number of iterations
    float        fr,                // input:  scalar     - refvec adaptation interval ratio
    float&       next_fr_trigger,   // update: (1,)       - next adaptation trigger threshold
    // float*       d_gamma_blkx_max,  // buffer: (N_btr_gamma, N_ref) - gamma phase1 buffer
    // float*       d_gamma,           // output: (N_ref,)   - recomputed gamma
    int          M,                 // input:  scalar     - objective count
    int          N_ref,             // input:  scalar     - reference vector count
    int          N_active,          // input:  scalar     - active column count
    int          N_stride,          // input:  scalar     - row stride in d_fv
    cudaStream_t exec_stream        // input:  scalar     - CUDA stream
);

} // namespace rvea
