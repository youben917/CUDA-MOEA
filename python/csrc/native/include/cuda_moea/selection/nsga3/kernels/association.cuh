#pragma once

void execute_association(
    float* d_ndsfv,                // input: (M, N_nds) - normalized fitness (transposed)
    float* d_nrps,                // input: (M, N_ref) - normalized ref points (transposed)
    float* d_assoc_blkx_min,      // buffer: (N_btr, N_nds) - partial min buffer
    int*   d_assoc_blkx_amin,     // buffer: (N_btr, N_nds) - partial argmin buffer
    float* d_asdist,              // output: (N_nds,) - final distances
    int*   d_asrpts_idx,            // output: (N_nds,) - final ref point indices
    int    N_ref,                 // input: scalar - number of reference points
    int    M,                     // input: scalar - number of objectives
    int    N_nds,                 // input: scalar - number of non-dominated individuals
    cudaStream_t exec_stream      // input: scalar - CUDA execution stream
);
