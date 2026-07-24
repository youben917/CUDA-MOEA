#include "cuda_moea/reproduction/mutation_operator.cuh"

#include <algorithm>
#include <stdexcept>

#include "cuda_moea/core/cuda/cuda_random.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"

namespace cuda_moea {
namespace {

float annealed(float begin, float end, int generation, int total)
{
    const float progress = total > 0
        ? std::clamp(static_cast<float>(generation) / total, 0.0f, 1.0f)
        : 1.0f;
    return begin + (end - begin) * progress;
}

__global__ void polynomial_mutation_kernel(
    const float2* random_values,
    float* offspring,
    const float* bounds,
    float eta,
    float probability,
    int count,
    int dimension)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const int d = index % dimension;
    const float2 random = random_values[index];
    if (random.x >= probability / static_cast<float>(dimension)) return;

    const float lower = bounds[2 * d];
    const float upper = bounds[2 * d + 1];
    const float range = upper - lower;
    const float value = offspring[index];
    const float normalized = (value - lower) / range;
    const float exponent = eta + 1.0f;

    float delta;
    if (random.y <= 0.5f) {
        const float xy = 1.0f - normalized;
        const float value_mut =
            2.0f * random.y +
            (1.0f - 2.0f * random.y) * powf(xy, exponent);
        delta = powf(value_mut, 1.0f / exponent) - 1.0f;
    } else {
        const float xy = normalized;
        const float value_mut =
            2.0f * (1.0f - random.y) +
            2.0f * (random.y - 0.5f) * powf(xy, exponent);
        delta = 1.0f - powf(value_mut, 1.0f / exponent);
    }

    offspring[index] =
        fminf(fmaxf(value + delta * range, lower), upper);
}

} // namespace

PolynomialMutation::PolynomialMutation(PolynomialMutationConfig config)
    : config_(config)
{
    if (config_.eta_initial <= 0.0f ||
        config_.eta_final <= 0.0f ||
        config_.probability < 0.0f ||
        config_.probability > 1.0f) {
        throw std::invalid_argument(
            "Invalid polynomial-mutation configuration");
    }
}

PolynomialMutation::PolynomialMutation(
    float distribution_index,
    float probability)
    : PolynomialMutation(PolynomialMutationConfig{
          distribution_index,
          distribution_index,
          probability})
{}

void PolynomialMutation::initialize(
    const AlgorithmInfo& info,
    CudaContext& cuda)
{
    random_values_.allocate(
        static_cast<std::size_t>(info.population_size) * info.dimension,
        cuda.execution_pool(),
        cuda.execution_stream());
}

void PolynomialMutation::mutate(
    PopulationView offspring,
    const GenerationContext& context,
    cudaStream_t stream)
{
    const int count = offspring.size * offspring.dimension;
    launch_random_float2_kernel(
        random_values_.data(),
        count,
        context.cuda.random_seed(),
        context.cuda.random_offset(),
        128,
        stream);

    const float eta = annealed(
        config_.eta_initial,
        config_.eta_final,
        context.generation,
        context.max_generations);
    const int blocks = (count + 255) / 256;
    polynomial_mutation_kernel<<<blocks, 256, 0, stream>>>(
        random_values_.data(),
        offspring.variables,
        offspring.bounds,
        eta,
        config_.probability,
        count,
        offspring.dimension);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_moea
