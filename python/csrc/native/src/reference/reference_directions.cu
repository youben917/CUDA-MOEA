#include "cuda_moea/reference/reference_direction_provider.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cuda_moea/reference/nsga3_reference_generator.cuh"
#include "cuda_moea/reference/rvea_reference_adaptation.cuh"
#include "cuda_moea/reference/rvea_reference_generator.cuh"
#include "cuda_moea/selection/rvea/kernels/angle_penalized_distance.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"

namespace cuda_moea {

DasDennisDirections::DasDennisDirections(DasDennisConfig config)
    : config_(config)
{
    if (config_.partitions < 0) {
        throw std::invalid_argument(
            "Das-Dennis partitions cannot be negative");
    }
}

void DasDennisDirections::initialize(
    int requested_count,
    int objective_count,
    CudaContext& cuda)
{
    auto generated = config_.partitions > 0
        ? initialize_refpts_by_partitions(
              config_.partitions,
              objective_count)
        : initialize_refpts(requested_count, objective_count);
    auto& [raw, normalized, count] = generated;
    (void)raw;
    objective_count_ = objective_count;
    count_ = count;
    directions_.allocate(
        normalized.size(),
        cuda.execution_pool(),
        cuda.execution_stream());
    CUDA_CHECK(cudaMemcpyAsync(
        directions_.data(),
        normalized.data(),
        normalized.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        cuda.execution_stream()));
}

ReferenceDirectionView DasDennisDirections::view() noexcept {
    return {
        directions_.data(),
        directions_.data(),
        nullptr,
        objective_count_,
        count_
    };
}

AdaptiveRVEADirections::AdaptiveRVEADirections(
    AdaptiveDirectionConfig config)
    : config_(config) {}

void AdaptiveRVEADirections::initialize(
    int requested_count,
    int objective_count,
    CudaContext& cuda)
{
    if (!(config_.frequency > 0.0f && config_.frequency <= 1.0f)) {
        throw std::invalid_argument(
            "RVEA reference adaptation frequency must be in (0, 1]");
    }

    auto [raw, normalized, count] =
        rvea::initialize_refpts(requested_count, objective_count);
    (void)raw;
    objective_count_ = objective_count;
    count_ = count;
    directions_.allocate(
        normalized.size(),
        cuda.execution_pool(),
        cuda.execution_stream());
    initial_directions_.allocate(
        normalized.size(),
        cuda.execution_pool(),
        cuda.execution_stream());
    zmin_.allocate(
        objective_count,
        cuda.execution_pool(),
        cuda.execution_stream());
    zmax_.allocate(
        objective_count,
        cuda.execution_pool(),
        cuda.execution_stream());
    const int gamma_tiles = (count_ + 255) / 256;
    gamma_block_max_.allocate(
        static_cast<std::size_t>(gamma_tiles) * count_,
        cuda.execution_pool(),
        cuda.execution_stream());
    gamma_.allocate(
        count_, cuda.execution_pool(), cuda.execution_stream());

    const std::size_t bytes = normalized.size() * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(
        directions_.data(),
        normalized.data(),
        bytes,
        cudaMemcpyHostToDevice,
        cuda.execution_stream()));
    CUDA_CHECK(cudaMemcpyAsync(
        initial_directions_.data(),
        normalized.data(),
        bytes,
        cudaMemcpyHostToDevice,
        cuda.execution_stream()));
    rvea::compute_gamma(
        directions_.data(), gamma_block_max_.data(), gamma_.data(),
        objective_count_, count_, cuda.execution_stream());
}

ReferenceDirectionView AdaptiveRVEADirections::view() noexcept {
    return {
        directions_.data(),
        initial_directions_.data(),
        gamma_.data(),
        objective_count_,
        count_
    };
}

void AdaptiveRVEADirections::update(
    ConstPopulationView population,
    int active_count,
    const GenerationContext& context,
    cudaStream_t stream)
{
    const int update_interval = std::max(
        1, static_cast<int>(std::lround(config_.frequency * context.max_generations)));
    const int completed_generation = context.generation + 1;
    if (completed_generation % update_interval != 0) return;

    float immediate_trigger = 0.0f;
    rvea::adapt_refvecs(
        initial_directions_.data(),
        population.objectives,
        directions_.data(),
        zmin_.data(),
        zmax_.data(),
        context.generation,
        context.max_generations,
        1.0f,
        immediate_trigger,
        objective_count_,
        count_,
        active_count,
        population.size,
        stream);
    rvea::compute_gamma(
        directions_.data(), gamma_block_max_.data(), gamma_.data(),
        objective_count_, count_, stream);
}

void AdaptiveRVEADirections::reset(cudaStream_t stream)
{
    if (!directions_.empty() && !initial_directions_.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(
            directions_.data(),
            initial_directions_.data(),
            directions_.size() * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
    }
    if (!gamma_.empty()) {
        rvea::compute_gamma(
            directions_.data(), gamma_block_max_.data(), gamma_.data(),
            objective_count_, count_, stream);
    }
}

UserDefinedDirections::UserDefinedDirections(
    std::vector<float> objective_major_values,
    int objective_count,
    bool normalize)
    : host_values_(std::move(objective_major_values)),
      objective_count_(objective_count),
      normalize_(normalize)
{
    if (objective_count_ <= 0 ||
        host_values_.empty() ||
        host_values_.size() % objective_count_ != 0) {
        throw std::invalid_argument(
            "User directions must have shape (objective_count, count)");
    }
    count_ = static_cast<int>(host_values_.size() / objective_count_);
}

void UserDefinedDirections::initialize(
    int,
    int objective_count,
    CudaContext& cuda)
{
    if (objective_count != objective_count_) {
        throw std::invalid_argument(
            "User direction objective count does not match the problem");
    }

    if (normalize_) {
        for (int k = 0; k < count_; ++k) {
            float norm2 = 0.0f;
            for (int m = 0; m < objective_count_; ++m) {
                const float value = host_values_[m * count_ + k];
                norm2 += value * value;
            }
            if (norm2 <= 0.0f) {
                throw std::invalid_argument(
                    "Reference directions must be non-zero");
            }
            const float inverse = 1.0f / std::sqrt(norm2);
            for (int m = 0; m < objective_count_; ++m) {
                host_values_[m * count_ + k] *= inverse;
            }
        }
    }

    directions_.allocate(
        host_values_.size(),
        cuda.execution_pool(),
        cuda.execution_stream());
    CUDA_CHECK(cudaMemcpyAsync(
        directions_.data(),
        host_values_.data(),
        host_values_.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        cuda.execution_stream()));
}

ReferenceDirectionView UserDefinedDirections::view() noexcept {
    return {
        directions_.data(),
        directions_.data(),
        nullptr,
        objective_count_,
        count_
    };
}

} // namespace cuda_moea
