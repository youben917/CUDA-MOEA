#pragma once

#include "cuda_moea/core/cuda/cuda_globals.cuh"

// ======================================================================================================================================================= //
//                                              MOPType Enum - Standard Test Problems                                                                        //
// ======================================================================================================================================================= //
enum MOPType {
    DTLZ1          = 1,
    DTLZ2          = 2,
    DTLZ3          = 3,
    DTLZ4          = 4,
    DTLZ5          = 5,
    DTLZ6          = 6,
    DTLZ7          = 7,
    CONVEX_DTLZ2   = 8,
    C1_DTLZ1       = 9,
    C1_DTLZ3       = 10,
    C2_DTLZ2       = 11,
    C2_CONVEX_DTLZ2 = 12,
    C3_DTLZ1       = 13,
    C3_DTLZ4       = 14,
    CSDP            = 15
};

// ======================================================================================================================================================= //
//                                              MOPAuxData - Auxiliary Data for Standard Test Problems                                                        //
// ======================================================================================================================================================= //
struct MOPAuxData {
    MOPType mop_type;
    int N, D, M;
    float* d_fv_trans = nullptr;  ///< (N, M) transpose buffer allocated in fcal_pool

    void mallocfrompool(int _M, int _N, cudaMemPool_t pool, cudaStream_t stream);
    void free(cudaStream_t stream);
};
