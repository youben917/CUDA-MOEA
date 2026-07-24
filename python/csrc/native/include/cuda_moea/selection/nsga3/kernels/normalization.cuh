#pragma once

// =========================================================================================================================================================================================== //
void cleanup_normalization_handle_pool();

// =========================================================================================================================================================================================== //
void execute_normalization(
    float*  d_ndsfv,                // update: (M, N_nds) - input: raw fitness; intermediate: shifted; output: normalized
    const float*  d_idpts,           // input: (M,) - ideal points
    float*  d_expts_blk_min,         // buffer: (num_blks_nds, M) - block-level min ASF values
    int*    d_expts_blk_amin,        // buffer: (num_blks_nds, M) - block-level argmin indices
    int*    d_expts_idx,             // buffer: (M,) - global extreme point indices
    float*  d_expts,                 // buffer: (M, M) - extreme points matrix (ROW-MAJOR; destroyed by SVD)

    float*  d_rcp_b,                 // buffer: (M,) - reciprocal intercepts

    int*    d_svdinfo,               // buffer: (1,) - SVD convergence info
    float*  d_U,                     // buffer: (M, M) - left singular vectors
    float*  d_sigma,                 // buffer: (M,) - singular values
    float*  d_Vh,                    // buffer: (M, M) - right singular vectors transposed
    int*    d_illcond,               // buffer: (1,) - ill-conditioning flag (device)
    float*  d_sigma_cross,           // buffer: (M,) - pseudo-inverse sigma
    const float*  d_ones,            // input: (M,) - vector of ones
    float*  d_u,                     // buffer: (M,) - intermediate: U^T * ones
    float*  d_s,                     // buffer: (M,) - intermediate: sigma_cross .* u
    int     M,                       // input: scalar - number of objectives
    int     N_nds,                   // input: scalar - N_prior + N_last
    cudaMemPool_t exec_pool,         // input: scalar - CUDA memory pool
    cudaStream_t exec_stream         // input: scalar - CUDA execution stream
);
// =========================================================================================================================================================================================== //
// overload: rescaled version
void execute_normalization_rescale(
    float*  d_ndsfv,                 // update: (M, N_nds) - input: raw fitness; intermediate: shifted; output: normalized
    const float*  d_idpts,            // input: (M,) - ideal points
    float*  d_obj_ranges,             // buffer: (M,) - per-objective dynamic ranges
    float*  d_expts_blk_min,          // buffer: (num_blks_nds, M) - block-level min ASF values
    int*    d_expts_blk_amin,         // buffer: (num_blks_nds, M) - block-level argmin indices
    int*    d_expts_idx,              // buffer: (M,) - global extreme point indices
    float*  d_expts,                  // buffer: (M, M) - extreme points matrix (ROW-MAJOR; destroyed by SVD)

    float*  d_rcp_b,                  // buffer: (M,) - reciprocal intercepts

    int*    d_svdinfo,                // buffer: (1,) - SVD convergence info
    float*  d_U,                      // buffer: (M, M) - left singular vectors
    float*  d_sigma,                  // buffer: (M,) - singular values
    float*  d_Vh,                     // buffer: (M, M) - right singular vectors transposed
    int*    d_illcond,                // buffer: (1,) - ill-conditioning flag (device)
    float*  d_pinv_threshold,         // buffer: (1,) - device pinv threshold (TAU_REL * sigma_max)
    float*  d_sigma_cross,            // buffer: (M,) - pseudo-inverse sigma
    const float*  d_ones,             // input: (M,) - vector of ones
    float*  d_u,                      // buffer: (M,) - intermediate: U^T * ones
    float*  d_s,                      // buffer: (M,) - intermediate: sigma_cross .* u
    int     M,                        // input: scalar - number of objectives
    int     N_nds,                    // input: scalar - N_prior + N_last
    cudaMemPool_t exec_pool,          // input: scalar - CUDA memory pool
    cudaStream_t exec_stream          // input: scalar - CUDA execution stream
);
