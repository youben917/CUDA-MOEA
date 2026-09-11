#include "problem.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
__global__ void bi_sphere_kernel(const float* x, float* f, float* cv,
                                 int n, int d, float offset, float radius,
                                 bool constrained) {
    // One block per individual; threads cooperate on the decision dimensions.
    const int i = blockIdx.x;
    const int lane = threadIdx.x;
    float a = 0.0f, b = 0.0f;
    for (int j = lane; j < d; j += blockDim.x) {
        float value = x[static_cast<size_t>(i) * d + j];
        a += value * value;
        float delta = value - offset;
        b += delta * delta;
    }
    __shared__ float first[128], second[128];
    first[lane] = a; second[lane] = b;
    __syncthreads();
    for (int step = 64; step; step /= 2) {
        if (lane < step) {
            first[lane] += first[lane + step];
            second[lane] += second[lane + step];
        }
        __syncthreads();
    }
    if (!lane) {
        f[i] = first[0];
        f[n + i] = second[0];
        cv[i] = constrained ? fmaxf(first[0] - radius * radius, 0.0f) : 0.0f;
    }
}
}

BiSphere::BiSphere(const cuda_moea::native::ProblemConfig& config)
    : offset_(static_cast<float>(config.get<double>("offset"))),
      radius_(static_cast<float>(config.get<double>("radius"))),
      constrained_(config.get<bool>("constrained")) {
    if (config.objectives != 2)
        throw std::invalid_argument("BiSphere requires objectives=2");
    if (!std::isfinite(offset_) || !std::isfinite(radius_) || radius_ < 0 ||
        radius_ > std::sqrt(std::numeric_limits<float>::max()))
        throw std::invalid_argument("BiSphere requires a finite float32 offset and nonnegative finite squared radius");
    info_.name = "BiSphere";
    info_.dimension = config.dimension;
    info_.objective_count = 2;
    info_.constraint_count = constrained_ ? 1 : 0;
    info_.lower_bounds = config.lower_bounds;
    info_.upper_bounds = config.upper_bounds;
}

void BiSphere::evaluate(cuda_moea::PopulationView p,
                        const cuda_moea::EvaluationContext&, cudaStream_t stream) {
    if (p.size == 0) return;
    bi_sphere_kernel<<<p.size, 128, 0, stream>>>(p.variables, p.objectives,
        p.constraints, p.size, p.dimension, offset_, radius_, constrained_);
}
