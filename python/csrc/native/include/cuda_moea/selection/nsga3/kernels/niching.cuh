#pragma once

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "non_dominated_sort.cuh"

void execute_niching(
    ull               rnd_seed,                 // input:  scalar - RNG seed for all niching random draws
    ull&              glb_rnd_offset,           // update: scalar - global RNG offset (reference for proper update propagation)

    const int*        d_nds_gidx,               // input: (N_nds,) - mapping from N_nds to N_mix space
    const float*      d_asdist,                 // input: (N_nds,) - perpendicular distance per individual (N_nds space)
    const int*        d_asrpts_idx,             // input: (N_nds,) - associated ref point per individual (N_nds space)

    int*              d_rho_prior,              // buffer: (N_ref,) - niche counts from prior fronts (Phase 1 output, Phase 4/5 input for virtual_round computation)
    
    ull*              d_sorted_keys_ping,       // buffer: (N_last,) - ping buffer for radix sort keys (alternates across 3 sorts)
                                                //                         Phase 2: (ref_idx << 32) | dist_bits
                                                //                         Phase 4: (ref_idx << 32) | rank_tiebreak  
                                                //                         Phase 5: (virtual_round << 32) | random_secondary
    
    ull*              d_sorted_keys_pong,       // buffer: (N_last,) - pong buffer for radix sort keys (CUB DoubleBuffer alternate)
    
    int*              d_sorted_lfidx_ping,      // buffer: (N_last,) - ping buffer for sorted indices in last front local space [0, N_last)
                                                //                         paired with keys through all 3 sorts, final order = selection order
    
    int*              d_sorted_lfidx_pong,      // buffer: (N_last,) - pong buffer for sorted indices (CUB DoubleBuffer alternate)
    
    int*              d_seg_head_idx,           // buffer: (N_last,) - segment boundary detection workspace (reused in Phase 3 & Phase 4)
                                                //                         Pre-scan:  -1 (non-boundary) or tidx (boundary marker)
                                                //                         Post-scan: propagated segment head index for each position
    
    int*              d_sorted_rpts_idx,        // buffer: (N_last,) - reference point IDs extracted from current sorted keys (high 32 bits)
                                                //                         updated after each sort to maintain ref_idx alignment with sorted order
    
    int*              d_niche_lf_rank,          // buffer: (N_last,) - multi-purpose rank storage (semantic changes across phases)
                                                //                         Phase 3: local_idx within segment (0 = closest to ref line)
                                                //                         Phase 4+: niching_rank after rank sort (intra-niche selection order)
                                                //                         used in Phase 5 for virtual_round = ρ_prior + rank computation
    
    uint*             d_uint32_niching,         // buffer: (N_last,) - 32-bit random number workspace (regenerated twice)
                                                //                         Phase 4: random bits for rank tiebreak (rand | 1 to avoid 0)
                                                //                         Phase 5: fresh random for final sort secondary key (full 32 bits)
    
    int*              d_newpop_gidx,            // output: (N,) - selected individual indices in N_mix space
                                                //                         [0, N_prior): prior fronts (direct copy)
                                                //                         [N_prior, N): selected from last front (Phase 6 gather)

    const NumNDSData& num_ndsdata,              // input: struct - {N_nds, N_prior, N_last, N_rem} 
    int               N_ref,                    // input: scalar - number of reference points
    int               N,                        // input: scalar - target population size
    cudaMemPool_t     exec_pool,                // input: scalar - memory pool
    cudaStream_t      exec_stream,              // input: scalar - execution stream
    bool              enable_h_save
);

// __global__ void histogram_atomic_kernel(
//     const int* d_input,    // (N_target,) input reference point indices
//     int* d_output,         // (N_ref,) output histogram counts
//     int N_target,          // number of input elements
//     int N_ref              // number of reference points (histogram bins)
// );
