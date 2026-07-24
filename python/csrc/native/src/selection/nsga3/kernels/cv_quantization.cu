#include <cmath>

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/selection/nsga3/kernels/cv_quantization.cuh"

namespace {

enum class CVQuantSpace : int { Linear = 0, Log = 1 };

template <CVQuantSpace SPACE>
__device__ __forceinline__ float quantize_cv_impl(
    float cv_raw,
    float eps_feas,
    float clip_upper,
    float inv_clip_log,
    float log_alpha,
    float inv_clip_upper,
    int   B
)
{
    const float cv_pos = fmaxf(cv_raw, 0.0f);
    const float u = fminf(cv_pos, clip_upper);

    float x = 0.0f;
    if constexpr (SPACE == CVQuantSpace::Log) {
        x = log1pf(log_alpha * u) * inv_clip_log;
    } else {
        x = u * inv_clip_upper;
    }

    int bin = 1 + __float2int_rd(x * __int2float_rn(B));
    bin = max(1, min(B, bin));

    // Phase 1 compatibility:
    // - eps_feas == 0.0f  -> preserve existing strict FEASIBLE_CV == 0.0f semantics
    // - eps_feas >  0.0f  -> future mode (warned by caller), absorb cv <= eps_feas to feasible
    const int infeas = (eps_feas == 0.0f)
        ? static_cast<int>(cv_raw != FEASIBLE_CV)
        : static_cast<int>(cv_raw > eps_feas);
    const int mask = -infeas;
    return __int_as_float(__float_as_int(__int2float_rn(bin)) & mask);
}

template <CVQuantSpace SPACE>
__global__ void cv_quantize_kernel(
    const float* __restrict__ d_cv_raw,
    float*       __restrict__ d_cv_quant,
    const float eps_feas,
    const float clip_upper,
    const float inv_clip_log,
    const float log_alpha,
    const float inv_clip_upper,
    const int   B,
    const int   N
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_cv_quant[idx] = quantize_cv_impl<SPACE>(
        d_cv_raw[idx], eps_feas, clip_upper, inv_clip_log, log_alpha, inv_clip_upper, B);
}

} // namespace

void nsga3::quantize_cv(
    const float* d_cv_raw,
    float*       d_cv_quant,
    int          N,
    float        eps_feas,
    float        clip_upper,
    float        log_alpha,
    int          B,
    cudaStream_t stream
)
{
    if (d_cv_raw == nullptr || d_cv_quant == nullptr || N <= 0 || B <= 0) return;

    constexpr int WPB = 8;
    constexpr int BLK = WPB * 32;
    const int NBLK = (N + BLK - 1) / BLK;

    const float inv_clip_upper = 1.0f / clip_upper;
    const float inv_clip_log = (log_alpha > 0.0f)
        ? (1.0f / log1pf(log_alpha * clip_upper))
        : 0.0f;

    if (log_alpha > 0.0f) {
        cv_quantize_kernel<CVQuantSpace::Log><<<NBLK, BLK, 0, stream>>>(
            d_cv_raw, d_cv_quant, eps_feas, clip_upper, inv_clip_log, log_alpha, inv_clip_upper, B, N);
    } else {
        cv_quantize_kernel<CVQuantSpace::Linear><<<NBLK, BLK, 0, stream>>>(
            d_cv_raw, d_cv_quant, eps_feas, clip_upper, inv_clip_log, log_alpha, inv_clip_upper, B, N);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool nsga3::validate_cv_quant_params(
    int   B,
    float clip_upper,
    float log_alpha,
    float eps_feas
)
{
    if (B <= 0) return false;
    if (clip_upper <= 0.0f || !std::isfinite(clip_upper)) return false;
    if (log_alpha < 0.0f  || !std::isfinite(log_alpha)) return false;
    if (eps_feas < 0.0f   || !std::isfinite(eps_feas)) return false;
    if (log_alpha > 0.0f) {
        const float clip_log = log1pf(log_alpha * clip_upper);
        if (!(clip_log > 0.0f) || !std::isfinite(clip_log)) return false;
    }
    return true;
}
