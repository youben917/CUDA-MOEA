#pragma once

#include "cuda_globals.cuh"

namespace ref_segment_ops {

void run_cub_radix_sorting(
    ull*&          d_keys_buf,         // update: (1,)                        - active key buffer pointer after sorting
    int*&          d_locidx_buf,       // update: (1,)                        - active value-index buffer pointer after sorting
    ull*           d_keys_buf_alt,     // buffer: (N,)                        - alternate key buffer for DoubleBuffer ping-pong
    int*           d_locidx_buf_alt,   // buffer: (N,)                        - alternate value-index buffer for DoubleBuffer ping-pong
    int            N,                  // input:  scalar                      - number of key-value pairs to sort
    cudaMemPool_t& exec_pool,          // input:  scalar                      - memory pool for CUB temporary storage
    cudaStream_t   exec_stream,        // input:  scalar                      - execution stream
    int            begin_bit = 0,      // input:  scalar                      - radix sort begin bit (inclusive)
    int            end_bit   = 64      // input:  scalar                      - radix sort end bit (exclusive)
);

void mark_boundaries_extract_ref(
    const ull*     d_sorted_keys,      // input:  (N,)                        - sorted packed keys, reference index in high 32 bits
    int*           d_seg_head_idx,     // output: (N,)                        - boundary marker/segment-head workspace (-1 or head index)
    int*           d_sorted_rpts_idx,  // output: (N,)                        - extracted reference indices aligned with sorted order
    int            N,                  // input:  scalar                      - element count
    cudaStream_t   exec_stream         // input:  scalar                      - execution stream
);

void run_segment_head_scan(
    int*            d_seg_head_idx,    // update: (N,)                        - in-place propagation of segment-head index
    int             N,                 // input:  scalar                      - element count
    cudaMemPool_t&  exec_pool,         // input:  scalar                      - memory pool for CUB temporary storage
    cudaStream_t    exec_stream        // input:  scalar                      - execution stream
);

void compute_niche_rank(
    const int*     d_seg_head_idx,     // input:  (N,)                        - propagated segment-head index per sorted position
    int*           d_niche_rank,       // output: (N,)                        - local rank in segment (position - head index)
    int            N,                  // input:  scalar                      - element count
    cudaStream_t   exec_stream         // input:  scalar                      - execution stream
);

} // namespace ref_segment_ops
