#pragma once

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_manager.cuh"
// ================================================================================================================= //
struct NumNDSData{
    int N_prior;     // scalar - number of individuals in prior fronts
    int N_last;      // scalar - number of individuals in last fronts
    int N_rem;       // scalar - N - N_prior
    int N_nds;       // scalar - N_prior + N_last
};
// ================================================================================================================= //

// Compute maximum domination pairs with sparse ratio based on population size
inline long long compute_max_dompairs(int size, float _sparse_ratio) {
    if (size <= 0) {
        throw std::invalid_argument("size must be positive");
    }

    long long size_ll    = static_cast<long long>(size);
    long long base_pairs = size_ll * (size_ll - 1) / 2;

    // Select sparse ratio based on population size (max N = 65536)
    double sparse_ratio;
    if (size <= 32768) {
        sparse_ratio = _sparse_ratio;
    } else if (size <= 65536) {
        sparse_ratio = _sparse_ratio;
    } else if (size <= 131072) {
        sparse_ratio = _sparse_ratio;
    } else {
        throw std::out_of_range("size too large");
    }

    double result_double = static_cast<double>(base_pairs) * sparse_ratio;

    if (result_double > std::numeric_limits<long long>::max()) {
        throw std::overflow_error("Result too large for long long");
    }

    return static_cast<long long>(result_double);
}
// =========================================================================================================================== //
int run_ndsort_init(
    cudaStreamSync cuda_streams,     // input: struct - data struct of CUDA multi-streams synchronization and events record
    float*         d_cv,             // input: (N,) - d_cv
    float*         d_fv,             // input: (M, N) - d_fv
    int*           d_fronts,         // output: (N,) - d_fronts
    int*           d_dompairs_init,  // buffer: (2 * PAIRS_CAP_INIT, ) - d_dompairs_init
    int*           d_dominatee_init, // buffer: (N,) - d_dominatee_init
    ll*            d_dom_cnt,        // buffer: (1,) - d_dom_cnt
    int*           d_asgn_cnt,       // buffer: (1,) - d_asgn_cnt
    int*           d_new_front,      // buffer: (1,) - d_new_front
    ll             PAIRS_CAP_INIT,   // input: scalar - PAIRS_CAP_INIT
    int            M,                // input: scalar - M
    int            N,                // input: scalar - target population size
    cudaMemPool_t  exec_pool         // input: scalar - CUDA memory pool for scratch allocations
);
// =========================================================================================================================== //
int run_ndsort(
    float*        d_mixcv,          // input: (N_mix,) - constraint violation values
    float*        d_mixfv,          // input: (M, N_mix) - fitness values for M objectives
    int*          d_mixfronts,      // output: (N_mix,) - assigned front index per individual
    int*          d_dompairs,       // buffer: (2 * PAIRS_CAP) - dominator-dominatee pairs in COO format
    uint32_t*     d_dominatee_mask, // buffer: (N_mix, N_tiles) - dominance bitmask in CSR format
    int*          d_dominatee,      // buffer: (N_mix,) - dominatee count per individual
    ll*           d_dom_cnt,        // buffer: (1,) - total number of dominance pairs
    int*          d_asgn_cnt,       // buffer: (1,) - number of individuals assigned to fronts
    int*          d_new_front,      // buffer: (1,) - flag indicating new front discovery
    ll            PAIRS_CAP,        // input: scalar - maximum capacity for dominance pairs
    bool&         switch2csr,       // update: bool - flag to switch COO->CSR mode
    int           n_iter,           // input: scalar - current iteration index
    int           N_iter,           // input: scalar - total number of GA iterations
    int           M,                // input: scalar - number of objectives
    int           N,                // input: scalar - target selection size
    cudaStream_t  exec_stream,      // input: scalar - CUDA execution stream
    cudaMemPool_t exec_pool         // input: scalar - CUDA memory pool for scratch allocations
);

NumNDSData get_nds_mask_size(
    int*          d_mixfronts,  // input: (N_mix,) - front index per individual
    int*          d_prior_mask, // output: (N_mix,) - bitmask for fronts < max_front
    int*          d_last_mask,  // output: (N_mix,) - bitmask for last front
    int           max_front,    // input: scalar - last front index
    int           N,            // input: scalar - target selection size
    cudaStream_t  exec_stream   // input: scalar - CUDA execution stream
);    

void extract_ndsfit_mskidx(
    float*            d_mixfv,         // input: (M, N_mix) - mixed population fitness
    int*              d_prior_mask,     // input: (N_mix,) - bitmask for prior fronts
    int*              d_last_mask,      // input: (N_mix,) - bitmask for last front
    int*              d_prior_gidx,     // output: (N_prior,) - indices of prior front individuals
    int*              d_last_gidx,      // output: (N_last,) - indices of last front individuals
    int*              d_nds_gidx,       // output: (N_nds,) - combined indices [prior; last]
    float*            d_ndsfv,         // output: (M, N_nds) - fitness values after non-dominated sorting
    const NumNDSData& num_ndsdata,      // input: struct - {N_nds, N_prior, N_last, N_rem}
    int               M,                // input: scalar - number of objectives
    int               N,                // input: scalar - target selection size 
    cudaStream_t      exec_stream       // input: scalar - CUDA execution stream
);
