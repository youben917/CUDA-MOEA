#include "cuda_moea/reproduction/crossover_operator.cuh"

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

__global__ void sbx_kernel(
    const float* parents,
    const int* parent_indices,
    const float* bounds,
    const float2* random_values,
    const int* random_signs,
    const float* copy_random,
    float* offspring,
    float eta,
    float probability,
    float variable_copy_probability,
    int pair_count,
    int dimension)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = pair_count * dimension;
    if (index >= count) return;

    const int pair = index / dimension;
    const int d = index % dimension;
    const int child_a = 2 * pair;
    const int child_b = child_a + 1;
    const int parent_a = parent_indices[child_a];
    const int parent_b = parent_indices[child_b];

    const float x1 = parents[parent_a * dimension + d];
    const float x2 = parents[parent_b * dimension + d];
    const float2 random = random_values[index];

    if (random.y >= probability ||
        copy_random[index] < variable_copy_probability) {
        offspring[child_a * dimension + d] = x1;
        offspring[child_b * dimension + d] = x2;
        return;
    }

    const float power = 1.0f / (1.0f + eta);
    float beta = random.x < 0.5f
        ? powf(2.0f * random.x, power)
        : powf(1.0f / (2.0f - 2.0f * random.x), power);
    beta *= 2 * random_signs[index] - 1;

    const float midpoint = 0.5f * (x1 + x2);
    const float delta = 0.5f * beta * (x1 - x2);
    const float lower = bounds[2 * d];
    const float upper = bounds[2 * d + 1];
    offspring[child_a * dimension + d] =
        fminf(fmaxf(midpoint + delta, lower), upper);
    offspring[child_b * dimension + d] =
        fminf(fmaxf(midpoint - delta, lower), upper);
}

__global__ void copy_last_parent_kernel(
    const float* parents,
    const int* parent_indices,
    float* offspring,
    int child,
    int dimension)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dimension) return;
    offspring[child * dimension + d] =
        parents[parent_indices[child] * dimension + d];
}

} // namespace

SimulatedBinaryCrossover::SimulatedBinaryCrossover(SBXConfig config)
    : config_(config)
{
    if (config_.eta_initial <= 0.0f ||
        config_.eta_final <= 0.0f ||
        config_.probability < 0.0f ||
        config_.probability > 1.0f ||
        config_.variable_copy_probability < 0.0f ||
        config_.variable_copy_probability > 1.0f) {
        throw std::invalid_argument("Invalid SBX configuration");
    }
}

SimulatedBinaryCrossover::SimulatedBinaryCrossover(
    float distribution_index,
    float probability)
    : SimulatedBinaryCrossover(SBXConfig{
          distribution_index,
          distribution_index,
          probability})
{}

void SimulatedBinaryCrossover::initialize(
    const AlgorithmInfo& info,
    CudaContext& cuda)
{
    const std::size_t count =
        static_cast<std::size_t>(info.population_size / 2) * info.dimension;
    random_values_.allocate(
        count, cuda.execution_pool(), cuda.execution_stream());
    random_signs_.allocate(
        count, cuda.execution_pool(), cuda.execution_stream());
    copy_random_.allocate(
        count, cuda.execution_pool(), cuda.execution_stream());
}

void SimulatedBinaryCrossover::apply(
    ConstPopulationView parents,
    DeviceSpan<const int> parent_indices,
    PopulationView offspring,
    const GenerationContext& context,
    cudaStream_t stream)
{
    if (parent_indices.size < static_cast<std::size_t>(offspring.size)) {
        throw std::invalid_argument("SBX parent-index array is too small");
    }

    const int pair_count = offspring.size / 2;
    const int count = pair_count * offspring.dimension;
    launch_random_float2_kernel(
        random_values_.data(),
        count,
        context.cuda.random_seed(),
        context.cuda.random_offset(),
        128,
        stream);
    launch_random_boolean_kernel(
        random_signs_.data(),
        count,
        context.cuda.random_seed(),
        context.cuda.random_offset(),
        256,
        stream);
    launch_random_float_kernel(
        copy_random_.data(),
        count,
        context.cuda.random_seed(),
        context.cuda.random_offset(),
        256,
        stream);

    const float eta = annealed(
        config_.eta_initial,
        config_.eta_final,
        context.generation,
        context.max_generations);
    const int blocks = (count + 255) / 256;
    sbx_kernel<<<blocks, 256, 0, stream>>>(
        parents.variables,
        parent_indices.data,
        parents.bounds,
        random_values_.data(),
        random_signs_.data(),
        copy_random_.data(),
        offspring.variables,
        eta,
        config_.probability,
        config_.variable_copy_probability,
        pair_count,
        offspring.dimension);
    CUDA_CHECK(cudaGetLastError());

    if (offspring.size % 2 != 0) {
        const int child = offspring.size - 1;
        const int copy_blocks = (offspring.dimension + 255) / 256;
        copy_last_parent_kernel<<<copy_blocks, 256, 0, stream>>>(
            parents.variables,
            parent_indices.data,
            offspring.variables,
            child,
            offspring.dimension);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace cuda_moea
