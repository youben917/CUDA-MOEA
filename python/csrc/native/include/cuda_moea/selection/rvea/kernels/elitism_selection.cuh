#pragma once

#include "cuda_moea/core/cuda/cuda_globals.cuh"

namespace rvea {

void run_elitism_selection(
    const int*           d_refvec_idx,     // input:  (N_mix,)                      - assigned reference-vector index
    const float*         d_mixcv,          // input:  (N_mix,)                      - merged constraint value
    const float*         d_apd,            // input:  (N_mix,)                      - APD metric
    ull*                 d_sort_keys_ping, // buffer: (N_mix,)                      - radix-sort key ping buffer
    ull*                 d_sort_keys_pong, // buffer: (N_mix,)                      - radix-sort key pong buffer
    int*                 d_sort_idx_ping,  // buffer: (N_mix,)                      - radix-sort index ping buffer
    int*                 d_sort_idx_pong,  // buffer: (N_mix,)                      - radix-sort index pong buffer
    int*                 d_seg_head_idx,   // buffer: (N_mix,)                      - segment-head workspace after boundary scan
    int*                 d_sorted_rpts_idx, // buffer: (N_mix,)                     - reference index extracted from sorted keys
    int*                 d_niche_rank,     // buffer: (N_mix,)                      - local rank inside each reference segment
    int*                 d_winner_gidx,    // buffer: (N_mix,)                      - winner indices in merged-population space
    int*                 d_winner_count,   // buffer: (1,)                          - number of segment winners
    int*                 d_newpop_gidx,    // output: (N,)                          - final selected indices for next population
    int                  N,                // input:  scalar                        - next-generation population size
    int                  N_mix,            // input:  scalar                        - merged-population size
    int                  N_ref,            // input:  scalar                        - number of valid reference vectors
    int&                 N_active,         // update: scalar                        - active population count after selection
    cudaMemPool_t&       exec_pool,        // input:  scalar                        - memory pool for temporary storage
    cudaStream_t         exec_stream       // input:  scalar                        - execution stream
);

void build_constrained_key(
    const int*           d_refvec_idx,     // input:  (N_mix,)                      - assigned reference-vector index
    const float*         d_mixcv,          // input:  (N_mix,)                      - merged constraint value
    const float*         d_apd,            // input:  (N_mix,)                      - APD metric
    unsigned long long*  d_sort_keys,      // output: (N_mix,)                      - packed constrained sorting key
    int                  N_mix,            // input:  scalar                        - merged-population size
    cudaStream_t         exec_stream       // input:  scalar                        - execution stream
);

void extract_segment_winners(
    const int*   d_niche_rank,      // input:  (N_mix,)                      - local rank inside each segment (winner iff rank==0)
    const int*   d_sorted_mix_idx,  // input:  (N_mix,)                      - sorted original indices
    const int*   d_sorted_rpts_idx, // input:  (N_mix,)                      - sorted reference-vector index per item
    int*         d_winner_gidx,     // output: (N_mix,)                      - winner indices in merged-population space
    int*         d_winner_count,    // output: (1,)                          - number of winners
    int          N_mix,             // input:  scalar                        - merged-population size
    int          N_ref,             // input:  scalar                        - number of valid reference vectors
    cudaStream_t exec_stream        // input:  scalar                        - execution stream
);

void generate_newpop_gidx(
    const int*                 d_winner_gidx, // input:  (N_winners,)                  - winner indices in merged-population space
    int                        N_winners,     // input:  scalar                        - winner count
    const unsigned long long*  d_sort_keys,   // input:  (N_mix,)                      - sorted constrained keys (compatibility placeholder)
    const int*                 d_sorted_idx,  // input:  (N_mix,)                      - sorted original indices
    const float*               d_mixcv,       // input:  (N_mix,)                      - merged constraint value
    const float*               d_apd,         // input:  (N_mix,)                      - APD metric
    int*                       d_newpop_gidx, // output: (N,)                          - selected indices for next generation
    int                        N,             // input:  scalar                        - next-generation population size
    int                        N_mix,         // input:  scalar                        - merged-population size
    cudaMemPool_t&             exec_pool,     // input:  scalar                        - memory pool for temporary storage
    cudaStream_t               exec_stream    // input:  scalar                        - execution stream
);

} // namespace rvea
