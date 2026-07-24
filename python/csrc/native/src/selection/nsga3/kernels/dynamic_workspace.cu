#include "cuda_moea/selection/nsga3/kernels/dynamic_workspace.cuh"
// ======================================================================================================================================== //
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
)
{   
    cudaStream_t&  exec_stream = cuda_streams.exec_stream;
    cudaMemPool_t& exec_pool   = cuda_mempools.exec_pool;

    const int N_prior = num_ndsdata.N_prior;
    const int N_last  = num_ndsdata.N_last;
    const int N_nds   = num_ndsdata.N_nds;

    // Mask & mask indices in non-dominated sorting process
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_prior_gidx, N_prior * INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_last_gidx,  N_last  * INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_nds_gidx,    N_nds  * INT_SIZE, exec_pool, exec_stream));
    
    // Normalization
    constexpr int WPB_NORM = 8;
    constexpr int BPT      = 2;
    constexpr int TS       = WPB_NORM * WS * BPT;
    const int N_tiles      = (N_nds + TS - 1) / TS;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_ndsfv,           (M   * N_nds) * FLOAT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_expts_blk_min,   (N_tiles * M) * FLOAT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_expts_blk_amin,  (N_tiles * M) *   INT_SIZE, exec_pool, exec_stream));

    // Association
    constexpr int WPB_ASSOC = 8;
    constexpr int BTR       = WPB_ASSOC * WS;           // block tile along reference points dimension
    const     int N_btr     = (N_ref + BTR - 1) / BTR;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_assoc_blkx_min,  (N_btr * N_nds) * FLOAT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_assoc_blkx_amin, (N_btr * N_nds) *   INT_SIZE, exec_pool, exec_stream));

    // Niching process
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_asdist,             N_nds * FLOAT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_asrpts_idx,         N_nds *   INT_SIZE, exec_pool, exec_stream));

    CUDA_CHECK(cudaMallocFromPoolAsync(&d_sorted_keys_ping,  N_last *   ULL_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_sorted_keys_pong,  N_last *   ULL_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_sorted_lfidx_ping, N_last *   INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_sorted_lfidx_pong, N_last *   INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_seg_head_idx,      N_last *   INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_sorted_rpts_idx,   N_last *   INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_niche_lf_rank,     N_last *   INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_uint32_niching,    N_last *  UINT_SIZE, exec_pool, exec_stream));

    return cudaSuccess;
}
// ============================================================================================================================== //
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
) 
{   
    // Generate ndsorted frontmask indices & mixpopfit
    CUDA_CHECK(cudaFreeAsync(d_prior_gidx,        exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_last_gidx,         exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_nds_gidx,          exec_stream));
    // Normalization
    CUDA_CHECK(cudaFreeAsync(d_ndsfv,             exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_expts_blk_min,     exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_expts_blk_amin,    exec_stream));
    // Association
    CUDA_CHECK(cudaFreeAsync(d_assoc_blkx_min,    exec_stream)); 
    CUDA_CHECK(cudaFreeAsync(d_assoc_blkx_amin,   exec_stream)); 
    CUDA_CHECK(cudaFreeAsync(d_asdist,            exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_asrpts_idx ,       exec_stream));
    // Niching process
    CUDA_CHECK(cudaFreeAsync(d_sorted_keys_ping,  exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_sorted_keys_pong,  exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_sorted_lfidx_ping, exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_sorted_lfidx_pong, exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_seg_head_idx,      exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_sorted_rpts_idx,   exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_niche_lf_rank,     exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_uint32_niching,    exec_stream));

    return cudaSuccess;
}
// ============================================================================================================================== //
