#include <cub/cub.cuh>

#include "cuda_moea/core/cuda/cuda_ref_segment_ops.cuh"

namespace {

constexpr int BLK_SIZE_COMMON = 256;

struct SegHeadPropagateOp {
    __host__ __device__ __forceinline__
    int operator()(int a, int b) const
    {
        return (b >= 0) ? b : a;
    }
};

__global__ void mark_boundaries_extract_ref_kernel(
    const ull* __restrict__ d_sorted_keys,      // input:  (N,) - sorted packed keys
    int*       __restrict__ d_seg_head_idx,     // output: (N,) - boundary marker / segment-head workspace
    int*       __restrict__ d_sorted_rpts_idx,  // output: (N,) - extracted reference indices
    int                     N                   // input:  scalar
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    const int curr_ref = static_cast<int>(d_sorted_keys[tidx] >> 32);
    d_sorted_rpts_idx[tidx] = curr_ref;

    const int prev_ref = (tidx > 0) ? static_cast<int>(d_sorted_keys[tidx - 1] >> 32) : -1;
    const int is_boundary = (tidx == 0) | (curr_ref != prev_ref);
    d_seg_head_idx[tidx] = is_boundary * tidx - (1 - is_boundary);
}

__global__ void compute_niche_rank_kernel(
    const int* __restrict__ d_seg_head_idx,  // input:  (N,) - propagated segment-head index
    int*       __restrict__ d_niche_rank,    // output: (N,) - local rank in segment
    int                     N                // input:  scalar
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < N) {
        d_niche_rank[tidx] = tidx - d_seg_head_idx[tidx];
    }
}

} // namespace

namespace ref_segment_ops {

void run_cub_radix_sorting(
    ull*&          d_keys_buf,
    int*&          d_locidx_buf,
    ull*           d_keys_buf_alt,
    int*           d_locidx_buf_alt,
    int            N,
    cudaMemPool_t& exec_pool,
    cudaStream_t   exec_stream,
    int            begin_bit,
    int            end_bit
)
{
    if (N <= 0) return;

    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DoubleBuffer<ull> keys_double_buf(d_keys_buf, d_keys_buf_alt);
    cub::DoubleBuffer<int> idx_double_buf(d_locidx_buf, d_locidx_buf_alt);

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, keys_double_buf, idx_double_buf, N, begin_bit, end_bit, exec_stream
    );

    CUDA_CHECK(cudaMallocFromPoolAsync(&d_temp_storage, temp_storage_bytes, exec_pool, exec_stream));
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, keys_double_buf, idx_double_buf, N, begin_bit, end_bit, exec_stream
    );
    CUDA_CHECK(cudaFreeAsync(d_temp_storage, exec_stream));

    d_keys_buf   = keys_double_buf.Current();
    d_locidx_buf = idx_double_buf.Current();
}

void mark_boundaries_extract_ref(
    const ull*   d_sorted_keys,
    int*         d_seg_head_idx,
    int*         d_sorted_rpts_idx,
    int          N,
    cudaStream_t exec_stream
)
{
    if (N <= 0) return;

    const int grid_size = (N + BLK_SIZE_COMMON - 1) / BLK_SIZE_COMMON;
    mark_boundaries_extract_ref_kernel<<<grid_size, BLK_SIZE_COMMON, 0, exec_stream>>>(
        d_sorted_keys, d_seg_head_idx, d_sorted_rpts_idx, N
    );
    CUDA_CHECK(cudaGetLastError());
}

void run_segment_head_scan(
    int*            d_seg_head_idx,
    int             N,
    cudaMemPool_t&  exec_pool,
    cudaStream_t    exec_stream
)
{
    if (N <= 0) return;

    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    SegHeadPropagateOp op;
    cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_seg_head_idx, d_seg_head_idx, op, N, exec_stream
    );

    CUDA_CHECK(cudaMallocFromPoolAsync(&d_temp_storage, temp_storage_bytes, exec_pool, exec_stream));
    cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_seg_head_idx, d_seg_head_idx, op, N, exec_stream
    );
    CUDA_CHECK(cudaFreeAsync(d_temp_storage, exec_stream));
}

void compute_niche_rank(
    const int*     d_seg_head_idx,
    int*           d_niche_rank,
    int            N,
    cudaStream_t   exec_stream
)
{
    if (N <= 0) return;

    const int grid_size = (N + BLK_SIZE_COMMON - 1) / BLK_SIZE_COMMON;
    compute_niche_rank_kernel<<<grid_size, BLK_SIZE_COMMON, 0, exec_stream>>>(d_seg_head_idx, d_niche_rank, N);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ref_segment_ops
