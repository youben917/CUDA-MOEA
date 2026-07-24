#include "cuda_moea/problem/dtlz/evaluation_workspace.cuh"

void MOPAuxData::mallocfrompool(int _M, int _N, cudaMemPool_t pool, cudaStream_t stream)
{
    M = _M;
    N = _N;
    const size_t fv_trans_bytes = static_cast<size_t>(N) * M * FLOAT_SIZE;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_fv_trans, fv_trans_bytes, pool, stream));
}

void MOPAuxData::free(cudaStream_t stream)
{
    if (d_fv_trans) CUDA_CHECK(cudaFreeAsync(d_fv_trans, stream));
    d_fv_trans = nullptr;
}
