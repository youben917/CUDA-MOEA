#pragma once

#include "cuda_runtime.h"

namespace nsga3 {

void quantize_cv(
    const float* d_cv_raw,     // input:  (N,) - raw CV values
    float*       d_cv_quant,   // output: (N,) - quantized CV values for ndsort
    int          N,            // input:  scalar - number of individuals
    float        eps_feas,     // input:  scalar - feasibility epsilon (Phase 1 uses strict-zero when == 0)
    float        clip_upper,   // input:  scalar - positive clip upper bound
    float        log_alpha,    // input:  scalar - log mapping alpha (0 = linear)
    int          B,            // input:  scalar - quantization bin count (> 0)
    cudaStream_t stream        // input:  scalar - CUDA stream
);

bool validate_cv_quant_params(
    int   B,
    float clip_upper,
    float log_alpha,
    float eps_feas
);

} // namespace nsga3
