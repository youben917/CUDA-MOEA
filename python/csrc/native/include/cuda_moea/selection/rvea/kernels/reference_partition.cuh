#pragma once

namespace rvea {

void execute_partition(
    const float* d_translated_fv,       // input:  (M, pop_size) - z_min-translated fitness
    const float* d_nrps,                // input:  (M, refvec_size) - L2-normalized reference points
    float*       d_part_blkx_max,       // buffer: (N_btr, pop_size) - partial max cosine buffer
    int*         d_part_blkx_amax,      // buffer: (N_btr, pop_size) - partial argmax buffer
    float*       d_theta,               // output: (pop_size,) - final angle values (radians)
    int*         d_refvec_idx,          // output: (pop_size,) - assigned reference vector indices
    int          refvec_size,           // input:  scalar - number of reference vectors
    int          M,                     // input:  scalar - number of objectives
    int          pop_size,              // input:  scalar - total population size (N_mix)
    int          N_active,              // input:  scalar - number of alive parent individuals
    int          N_parent,              // input:  scalar - parent section end index [0, N_parent)
    cudaStream_t exec_stream            // input:  scalar - CUDA execution stream
);

} // namespace rvea
