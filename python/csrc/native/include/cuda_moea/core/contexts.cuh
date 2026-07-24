#pragma once

#include "cuda_moea/core/cuda_context.cuh"

namespace cuda_moea {

struct GenerationContext {
    int generation = 0;
    int max_generations = 0;
    CudaContext& cuda;
};

struct EvaluationContext {
    int generation = 0;
    int max_generations = 0;
    CudaContext& cuda;
};

} // namespace cuda_moea
