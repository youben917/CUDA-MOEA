#pragma once

#include "cuda_moea/core/cuda/cuda_manager.cuh"
#include "non_dominated_sort.cuh"
// ====================================================================================================================================== //
cudaError_t malloc_dynmem( 
    cudaStreamSync     cuda_streams,        // input: struct - data struct of CUDA multi-streams synchronization and events record
    cudaMemPools&      cuda_mempools,       // input: struct - handles of cuda_mempools including fcal_pool and exec_pool
    // Mask indices in non-dominated sorting process
    int*&              d_prior_gidx,        // input: (N_prior,) - d_prior_gidx
    int*&              d_last_gidx,         // input: (N_last,) - d_last_gidx
    int*&              d_nds_gidx,          // input: (N_nds,) - nds index list in merged-population space
    float*&            d_ndsfv,             // input: (M, N_nds) - d_ndsfv
    float*&            d_expts_blk_min,     // input: (N_tiles, M) - d_expts_blk_min
    int*&              d_expts_blk_amin,    // input: (N_tiles, M) - argmin indices for block-level ASF reduction
    float*&            d_assoc_blkx_min,    // input: (N_btr, N_ref) - d_assoc_blkx_min
    int*&              d_assoc_blkx_amin,   // input: (N_btr, N_ref) - d_assoc_blkx_amin
    // Niching process
    float*&            d_asdist,            // input: (N_nds,) - d_asdist
    int*&              d_asrpts_idx,        // input: (N_nds,) - d_asrpts_idx

    ull*&              d_sorted_keys_ping,  // input: (N_last,) - d_sorted_keys_ping
    ull*&              d_sorted_keys_pong,  // input: (N_last,) - d_sorted_keys_pong
    int*&              d_sorted_lfidx_ping, // input: (N_last,) - d_sorted_lfidx_ping
    int*&              d_sorted_lfidx_pong, // input: (N_last,) - d_sorted_lfidx_pong
    int*&              d_seg_head_idx,      // input: (N_last,) - d_seg_head_idx
    int*&              d_sorted_rpts_idx,   // input: (N_last,) - d_sorted_rpts_idx
    int*&              d_niche_lf_rank,     // input: (N_last,) - d_niche_lf_rank
    uint*&             d_uint32_niching,    // input: (N_last,) - d_uint32_niching

    const NumNDSData&  num_ndsdata,         // input: struct - {int N_nds, int N_prior, int N_last, int N_rem} 
    int                N_ref,               // input: scalar - number of reference points
    int                M                    // input: scalar - number of objectives
);
// =================================================================================================================================================================== //
cudaError_t free_dynmem(
    cudaStream_t exec_stream,          // input: struct - data struct of CUDA multi-streams synchronization and events record
    // non-dominated sorting - front mask indices
    int*&        d_prior_gidx,         // input: (N_prior,) - d_prior_gidx
    int*&        d_last_gidx,          // input: (N_last,) - d_last_gidx
    int*&        d_nds_gidx,           // input: (N_nds,) - nds index list in merged-population space
    float*&      d_ndsfv,              // input: (M, N_nds) - d_ndsfv
    float*&      d_expts_blk_min,      // input: (N_tiles, M) - d_expts_blk_min
    int*&        d_expts_blk_amin,     // input: (N_tiles, M) - argmin indices for block-level ASF reduction
    float*&      d_assoc_blkx_min,     // input: (N_btr, N_ref) - d_assoc_blkx_min
    int*&        d_assoc_blkx_amin,    // input: (N_btr, N_ref) - d_assoc_blkx_amin
    // Niching process
    float*&      d_asdist,             // input: (N_nds,) - d_asdist
    int*&        d_asrpts_idx,         // input: (N_nds,) - d_asrpts_idx
    ull*&        d_sorted_keys_ping,   // input: (N_last,) - d_sorted_keys_ping
    ull*&        d_sorted_keys_pong,   // input: (N_last,) - d_sorted_keys_pong
    int*&        d_sorted_lfidx_ping,  // input: (N_last,) - d_sorted_lfidx_ping
    int*&        d_sorted_lfidx_pong,  // input: (N_last,) - d_sorted_lfidx_pong
    int*&        d_seg_head_idx,       // input: (N_last,) - d_seg_head_idx
    int*&        d_sorted_rpts_idx,    // input: (N_last,) - d_sorted_rpts_idx
    int*&        d_niche_lf_rank,      // input: (N_last,) - d_niche_lf_rank
    uint*&       d_uint32_niching      // input: (N_last,) - random uint32 workspace for niching
);
