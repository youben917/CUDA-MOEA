#pragma once

void get_idpts(
    const float* d_fv,           // input: (M, N) - fitness values
    float* d_idpts,              // output: (M,) - ideal points
    const int M,                 // input: scalar - number of objectives
    const int N,                 // input: scalar - population size (N % 2 == 0)                   
    cudaStream_t exec_stream     // input: scalar - CUDA execution stream
);

void update_idpts(
    float* d_idpts_last,         // update: (M,) - ideal points of last iteration
    const float* d_idpts_off,    // input: (M,) - ideal points of offspring 
    float* d_idpts,              // output: (M,) - ideal points of current iteration
    const int M,                 // input: scalar - number of objectives
    cudaStream_t exec_stream     // input: scalar - CUDA execution stream
);