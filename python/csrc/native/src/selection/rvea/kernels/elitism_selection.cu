#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/selection/rvea/kernels/elitism_selection.cuh"
#include "cuda_moea/core/cuda/cuda_ref_segment_ops.cuh"

namespace {

constexpr float CV_EPS     = 1e-8f;   // feasibility tolerance for CV boundary
constexpr float METRIC_MAX = 1e30f;   // bounded sentinel for invalid metric values

__device__ __forceinline__ int bool_to_mask(const bool pred)
{
    return -static_cast<int>(pred);
}

__device__ __forceinline__ float bitselect_f32(
    const int   mask,                  // input: 0xFFFFFFFF -> take_true, 0x00000000 -> take_false
    const float take_true,
    const float take_false
)
{
    const unsigned int umask = static_cast<unsigned int>(mask);
    const unsigned int a     = __float_as_uint(take_true);
    const unsigned int b     = __float_as_uint(take_false);
    return __uint_as_float((a & umask) | (b & ~umask));
}

// __device__ __forceinline__ int bitselect_i32(
//     const int mask,                     // input: 0xFFFFFFFF -> take_true, 0x00000000 -> take_false
//     const int take_true,
//     const int take_false
// )
// {
//     const unsigned int umask = static_cast<unsigned int>(mask);
//     const unsigned int a     = static_cast<unsigned int>(take_true);
//     const unsigned int b     = static_cast<unsigned int>(take_false);
//     return static_cast<int>((a & umask) | (b & ~umask));
// }

__device__ __forceinline__ float sanitize_nonneg_metric(const float metric)
{
    const int valid_mask = bool_to_mask(isfinite(metric) && (metric >= 0.f));
    return bitselect_f32(valid_mask, metric, METRIC_MAX);
}

} // namespace

// ========================================================================================================== //
// KERNEL: Build Constrained Key
// ========================================================================================================== //
// PURPOSE:
//   Build packed constrained sorting key for each mixed-population individual:
//     key = (ref_idx << 32) | (flag << 31) | metric_bits
//
// KEY LAYOUT:
//   [63:32] ref_idx
//   [31]    feasibility flag (0 = feasible, 1 = infeasible)
//   [30:0]  metric bits (__float_as_uint(metric) & 0x7FFFFFFF)
//
// METRIC RULE:
//   feasible (CV <= CV_EPS): metric = APD
//   infeasible:              metric = CV
// ========================================================================================================== //
__global__ void build_constrained_key_kernel(
    const int*   __restrict__ d_refvec_idx,         // input:  (N_mix,) - assigned reference index
    const float* __restrict__ d_mixcv,              // input:  (N_mix,) - merged CV
    const float* __restrict__ d_apd,                // input:  (N_mix,) - APD metric
    ull*         __restrict__ d_sort_keys,          // output: (N_mix,) - packed constrained sort key
    int N_mix                                       // input:  scalar - merged population size
)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N_mix;
         tidx += gridDim.x * blockDim.x) {

        const int   ref_idx   = d_refvec_idx[tidx];
        const float cv        = d_mixcv[tidx];
        const float apd       = d_apd[tidx];
        const int   feas_mask = bool_to_mask(cv <= CV_EPS);

        const float metric_raw  = bitselect_f32(feas_mask, apd, cv);
        const float metric_safe = sanitize_nonneg_metric(metric_raw);

        const uint metric_bits = __float_as_uint(metric_safe) & 0x7FFFFFFFu;
        const uint flag        = 1u ^ (static_cast<uint>(feas_mask) & 1u); // feasible->0, infeasible->1

        const ull ref_part    = static_cast<ull>(static_cast<uint>(ref_idx)) << 32;
        const ull flag_part   = static_cast<ull>(flag) << 31;
        const ull metric_part = static_cast<ull>(metric_bits);
        d_sort_keys[tidx] = ref_part | flag_part | metric_part;
    }
}

// ========================================================================================================== //
// KERNEL: Extract Segment Winners
// ========================================================================================================== //
// PURPOSE:
//   Select one winner per non-empty reference-vector segment after constrained sort.
//   Segment winner is identified by local rank == 0 in sorted order.
// ========================================================================================================== //
__global__ void extract_segment_winners_kernel(
    const int* __restrict__ d_niche_rank,           // input:  (N_mix,) - local rank in segment
    const int* __restrict__ d_sorted_mix_idx,       // input:  (N_mix,) - sorted original indices
    const int* __restrict__ d_sorted_rpts_idx,      // input:  (N_mix,) - sorted reference-vector index
    int*       __restrict__ d_winner_gidx,          // output: (N_mix,) - winner global indices
    int*       __restrict__ d_winner_count,         // output: (1,) - atomic winner count
    int N_mix,                                      // input:  scalar - merged population size
    int N_ref                                       // input:  scalar - number of valid reference vectors
)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N_mix;
         tidx += gridDim.x * blockDim.x) {

        const int ref_idx = d_sorted_rpts_idx[tidx];
        if (d_niche_rank[tidx] == 0 && ref_idx >= 0 && ref_idx < N_ref) {
            const int pos = atomicAdd(d_winner_count, 1);
            d_winner_gidx[pos] = d_sorted_mix_idx[tidx];
        }
    }
}


// ========================================================================================================== //
// KERNEL: Build Winner-Only Truncation Keys
// ========================================================================================================== //
// PURPOSE:
//   Build winner-quality sort key without ref_idx prefix:
//     trunc_key = (flag << 63) | (metric_bits << 32) | tie32
//
// Semantics:
//   flag       = 0 if feasible else 1
//   metric     = feasible ? APD : CV
//   tie32      = global index (deterministic tie-break)
// ========================================================================================================== //
__global__ void build_winner_trunc_key_kernel(
    const int*   __restrict__ d_winner_gidx,        // input:  (N_winners,) winner indices
    const float* __restrict__ d_mixcv,              // input:  (N_mix,) merged CV
    const float* __restrict__ d_apd,                // input:  (N_mix,) APD metric
    ull*         __restrict__ d_trunc_keys,         // output: (N_winners,) packed trunc keys
    int N_winners                                   // input:  scalar - winner count
)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N_winners;
         tidx += gridDim.x * blockDim.x) {

        const int gidx      = d_winner_gidx[tidx];
        const float cv      = d_mixcv[gidx];
        const float apd     = d_apd[gidx];
        const int feas_mask = bool_to_mask(cv <= CV_EPS);

        const float metric_raw  = bitselect_f32(feas_mask, apd, cv);
        const float metric_safe = sanitize_nonneg_metric(metric_raw);

        const uint metric_bits = __float_as_uint(metric_safe) & 0x7FFFFFFFu;
        const uint flag        = 1u ^ (static_cast<uint>(feas_mask) & 1u);
        const uint tie32       = static_cast<uint>(gidx);

        const ull flag_part   = static_cast<ull>(flag) << 63;
        const ull metric_part = static_cast<ull>(metric_bits) << 32;
        const ull tie_part    = static_cast<ull>(tie32);
        d_trunc_keys[tidx] = flag_part | metric_part | tie_part;
    }
}

// ========================================================================================================== //
// KERNEL: Gather Top-N Winners After Winner-Only Sort
// ========================================================================================================== //
__global__ void gather_topn_winners_kernel(
    const int* __restrict__ d_sorted_winner_locidx, // input:  (N_winners,) sorted winner-local indices
    const int* __restrict__ d_winner_gidx,          // input:  (N_winners,) winner global indices
    int*       __restrict__ d_newpop_gidx,          // output: (N,) final selected indices
    int N                                           // input:  scalar - output size
)
{
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N;
         tidx += gridDim.x * blockDim.x) {

        const int wloc = d_sorted_winner_locidx[tidx];
        d_newpop_gidx[tidx] = d_winner_gidx[wloc];
    }
}


// ========================================================================================================== //
// API: Run Full RVEA Elitism Selection Pipeline
// ========================================================================================================== //
void rvea::run_elitism_selection(
    const int*           d_refvec_idx,      // input:  (N_mix,) - assigned reference-vector index
    const float*         d_mixcv,           // input:  (N_mix,) - merged constraint value
    const float*         d_apd,             // input:  (N_mix,) - APD metric
    ull*                 d_sort_keys_ping,  // buffer: (N_mix,) - radix-sort key ping buffer
    ull*                 d_sort_keys_pong,  // buffer: (N_mix,) - radix-sort key pong buffer
    int*                 d_sort_idx_ping,   // buffer: (N_mix,) - radix-sort index ping buffer
    int*                 d_sort_idx_pong,   // buffer: (N_mix,) - radix-sort index pong buffer
    int*                 d_seg_head_idx,    // buffer: (N_mix,) - segment-head workspace
    int*                 d_sorted_rpts_idx, // buffer: (N_mix,) - extracted reference index in sorted order
    int*                 d_niche_rank,      // buffer: (N_mix,) - local rank inside each reference segment
    int*                 d_winner_gidx,     // buffer: (N_mix,) - winner indices in merged-population space
    int*                 d_winner_count,    // buffer: (1,)     - number of segment winners
    int*                 d_newpop_gidx,     // output: (N,)     - final selected indices for next generation
    int                  N,                 // input:  scalar   - next-generation population size
    int                  N_mix,             // input:  scalar   - merged-population size
    int                  N_ref,             // input:  scalar   - number of valid reference vectors
    int&                 N_active,          // update: scalar   - active population count after selection
    cudaMemPool_t&       exec_pool,         // input:  scalar   - memory pool for temporary storage
    cudaStream_t         exec_stream        // input:  scalar   - execution stream
)
{
    if (N <= 0 || N_mix <= 0) {
        N_active = 0;
        return;
    }

    // Phase 1: build constrained key, then initialize sort indices.
    build_constrained_key(
        d_refvec_idx, d_mixcv, d_apd,
        reinterpret_cast<unsigned long long*>(d_sort_keys_ping),
        N_mix,
        exec_stream
    );
    thrust::sequence(thrust::cuda::par.on(exec_stream), d_sort_idx_ping, d_sort_idx_ping + N_mix, 0);

    // Phase 2: sort by constrained key (ref group + feasibility + metric).
    ull* d_sorted_keys = d_sort_keys_ping;
    int* d_sorted_idx  = d_sort_idx_ping;
    ref_segment_ops::run_cub_radix_sorting(
        d_sorted_keys, d_sorted_idx,
        d_sort_keys_pong, d_sort_idx_pong,
        N_mix, exec_pool, exec_stream,
        0, 64
    );

    // Phase 3: detect reference segments and compute local rank in each segment.
    ref_segment_ops::mark_boundaries_extract_ref(
        d_sorted_keys, d_seg_head_idx, d_sorted_rpts_idx, N_mix, exec_stream
    );
    ref_segment_ops::run_segment_head_scan(d_seg_head_idx, N_mix, exec_pool, exec_stream);
    ref_segment_ops::compute_niche_rank(d_seg_head_idx, d_niche_rank, N_mix, exec_stream);

    // Phase 4: keep one winner per segment, then fill/truncate to exact N.
    extract_segment_winners(
        d_niche_rank, d_sorted_idx, d_sorted_rpts_idx,
        d_winner_gidx, d_winner_count,
        N_mix, N_ref,
        exec_stream
    );

    int h_winner_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_winner_count, d_winner_count, INT_SIZE, cudaMemcpyDeviceToHost, exec_stream));
    CUDA_CHECK(cudaStreamSynchronize(exec_stream));

    generate_newpop_gidx(
        d_winner_gidx, h_winner_count,
        reinterpret_cast<const unsigned long long*>(d_sorted_keys),
        d_sorted_idx,
        d_mixcv, d_apd,
        d_newpop_gidx,
        N, N_mix,
        exec_pool,
        exec_stream
    );

    N_active = (h_winner_count > 0) ? std::min(h_winner_count, N) : N;
}

// ========================================================================================================== //
// API: Build Constrained Key
// ========================================================================================================== //
void rvea::build_constrained_key(
    const int*           d_refvec_idx,            // input:  (N_mix,)
    const float*         d_mixcv,                 // input:  (N_mix,)
    const float*         d_apd,                   // input:  (N_mix,)
    unsigned long long*  d_sort_keys,             // output: (N_mix,)
    int                  N_mix,                   // input:  scalar
    cudaStream_t         exec_stream              // input:  scalar
)
{
    if (N_mix <= 0) return;

    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE = (N_mix + BLK_SIZE - 1) / BLK_SIZE;

    build_constrained_key_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_refvec_idx, d_mixcv, d_apd, reinterpret_cast<ull*>(d_sort_keys), N_mix
    );
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================================== //
// API: Extract Segment Winners
// ========================================================================================================== //
void rvea::extract_segment_winners(
    const int*   d_niche_rank,              // input:  (N_mix,)
    const int*   d_sorted_mix_idx,          // input:  (N_mix,)
    const int*   d_sorted_rpts_idx,         // input:  (N_mix,)
    int*         d_winner_gidx,             // output: (N_mix,)
    int*         d_winner_count,            // output: (1,)
    int          N_mix,                     // input:  scalar
    int          N_ref,                     // input:  scalar
    cudaStream_t exec_stream                // input:  scalar
)
{
    CUDA_CHECK(cudaMemsetAsync(d_winner_count, 0, INT_SIZE, exec_stream));
    if (N_mix <= 0) return;

    constexpr int BLK_SIZE = 256;
    const int GRID_SIZE = (N_mix + BLK_SIZE - 1) / BLK_SIZE;

    extract_segment_winners_kernel<<<GRID_SIZE, BLK_SIZE, 0, exec_stream>>>(
        d_niche_rank, d_sorted_mix_idx, d_sorted_rpts_idx, d_winner_gidx, d_winner_count, N_mix, N_ref
    );
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================================== //
// API: Generate Exact-Size d_newpop_gidx
// ========================================================================================================== //
void rvea::generate_newpop_gidx(
    const int*            d_winner_gidx,            // input:  (N_winners,)
    int                   N_winners,                // input:  scalar
    const unsigned long long* d_sort_keys,          // input:  (N_mix,) - reserved for compatibility
    const int*            d_sorted_idx,             // input:  (N_mix,)
    const float*          d_mixcv,                  // input:  (N_mix,)
    const float*          d_apd,                    // input:  (N_mix,)
    int*                  d_newpop_gidx,            // output: (N,)
    int                   N,                        // input:  scalar
    int                   N_mix,                    // input:  scalar
    cudaMemPool_t&        exec_pool,                // input:  scalar
    cudaStream_t          exec_stream               // input:  scalar
)
{
    (void)d_sort_keys;

    if (N <= 0 || N_mix <= 0) return;
    if (N_winners < 0) N_winners = 0;

    // ------------------------------------------------------------------------------------------------------
    // Case 1: No winner extracted (fallback) -> take first N from globally sorted list.
    // ------------------------------------------------------------------------------------------------------
    if (N_winners == 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_newpop_gidx, d_sorted_idx, static_cast<size_t>(N) * INT_SIZE,
            cudaMemcpyDeviceToDevice, exec_stream
        ));
        return;
    }

    // ------------------------------------------------------------------------------------------------------
    // Case 2: Exactly N winners -> direct copy.
    // ------------------------------------------------------------------------------------------------------
    if (N_winners == N) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_newpop_gidx, d_winner_gidx, static_cast<size_t>(N) * INT_SIZE,
            cudaMemcpyDeviceToDevice, exec_stream
        ));
        return;
    }

    // ------------------------------------------------------------------------------------------------------
    // Case 3: N_winners < N -> copy winners only (paper-faithful: |P_{t+1}| <= N)
    // ------------------------------------------------------------------------------------------------------
    if (N_winners < N) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_newpop_gidx, d_winner_gidx, static_cast<size_t>(N_winners) * INT_SIZE,
            cudaMemcpyDeviceToDevice, exec_stream
        ));
        return;
    }

    // ------------------------------------------------------------------------------------------------------
    // Case 4: N_winners > N — effectively unreachable when N == N_ref under normal
    // operation (at most N_ref unique ref segments → at most N_ref winners).
    // Retained as defensive fallback for unforeseen edge cases.
    // ------------------------------------------------------------------------------------------------------
    ull* d_trunc_keys_ping = nullptr;
    ull* d_trunc_keys_pong = nullptr;
    int* d_trunc_idx_ping  = nullptr;
    int* d_trunc_idx_pong  = nullptr;

    CUDA_CHECK(cudaMallocFromPoolAsync(&d_trunc_keys_ping, static_cast<size_t>(N_winners) * ULL_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_trunc_keys_pong, static_cast<size_t>(N_winners) * ULL_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_trunc_idx_ping,  static_cast<size_t>(N_winners) * INT_SIZE, exec_pool, exec_stream));
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_trunc_idx_pong,  static_cast<size_t>(N_winners) * INT_SIZE, exec_pool, exec_stream));

    constexpr int BLK_SIZE = 256;
    const int GRID_WIN = (N_winners + BLK_SIZE - 1) / BLK_SIZE;
    const int GRID_OUT = (N + BLK_SIZE - 1) / BLK_SIZE;

    build_winner_trunc_key_kernel<<<GRID_WIN, BLK_SIZE, 0, exec_stream>>>(
        d_winner_gidx, d_mixcv, d_apd, d_trunc_keys_ping, N_winners
    );
    CUDA_CHECK(cudaGetLastError());

    thrust::sequence(thrust::cuda::par.on(exec_stream), d_trunc_idx_ping, d_trunc_idx_ping + N_winners);

    ull* d_keys_active = d_trunc_keys_ping;
    int* d_idx_active  = d_trunc_idx_ping;
    ref_segment_ops::run_cub_radix_sorting(
        d_keys_active, d_idx_active, d_trunc_keys_pong, d_trunc_idx_pong,
        N_winners, exec_pool, exec_stream, 0, 64
    );

    gather_topn_winners_kernel<<<GRID_OUT, BLK_SIZE, 0, exec_stream>>>(
        d_idx_active, d_winner_gidx, d_newpop_gidx, N
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_trunc_idx_pong, exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_trunc_idx_ping, exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_trunc_keys_pong, exec_stream));
    CUDA_CHECK(cudaFreeAsync(d_trunc_keys_ping, exec_stream));
}
