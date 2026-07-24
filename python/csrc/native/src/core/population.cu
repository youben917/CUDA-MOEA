#include "cuda_moea/core/population.cuh"

#include <stdexcept>
#include <vector>

#include "cuda_moea/core/cuda/cuda_random.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"

namespace cuda_moea {
namespace {

__global__ void map_unit_to_bounds_kernel(
    float* variables,
    const float* bounds,
    int count,
    int dimension)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int d = index % dimension;
    const float lower = bounds[2 * d];
    const float upper = bounds[2 * d + 1];
    variables[index] = lower + variables[index] * (upper - lower);
}

} // namespace

Population::Population(
    int size,
    int dimension,
    int objective_count,
    CudaContext& cuda)
{
    allocate(size, dimension, objective_count, cuda);
}

void Population::allocate(
    int size,
    int dimension,
    int objective_count,
    CudaContext& cuda)
{
    if (size <= 0 || dimension <= 0 || objective_count <= 0) {
        throw std::invalid_argument(
            "Population requires positive size, dimension, and objective count");
    }

    size_ = size;
    dimension_ = dimension;
    objective_count_ = objective_count;

    variables_.allocate(
        static_cast<std::size_t>(size) * dimension,
        cuda.execution_pool(),
        cuda.execution_stream());
    bounds_.allocate(
        static_cast<std::size_t>(2) * dimension,
        cuda.execution_pool(),
        cuda.execution_stream());
    objectives_.allocate(
        static_cast<std::size_t>(objective_count) * size,
        cuda.evaluation_pool(),
        cuda.evaluation_stream());
    constraints_.allocate(
        size,
        cuda.evaluation_pool(),
        cuda.evaluation_stream());
}

void Population::set_bounds(
    const std::vector<float>& lower,
    const std::vector<float>& upper,
    cudaStream_t stream)
{
    if (static_cast<int>(lower.size()) != dimension_ ||
        static_cast<int>(upper.size()) != dimension_) {
        throw std::invalid_argument("Problem bounds do not match population dimension");
    }

    std::vector<float> interleaved(static_cast<std::size_t>(2) * dimension_);
    for (int d = 0; d < dimension_; ++d) {
        if (!(lower[d] < upper[d])) {
            throw std::invalid_argument("Each lower bound must be smaller than upper bound");
        }
        interleaved[2 * d] = lower[d];
        interleaved[2 * d + 1] = upper[d];
    }

    CUDA_CHECK(cudaMemcpyAsync(
        bounds_.data(),
        interleaved.data(),
        interleaved.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
}

void Population::initialize_random(CudaContext& cuda)
{
    const int count = size_ * dimension_;
    launch_random_uniform_kernel(
        variables_.data(),
        count,
        cuda.random_seed(),
        cuda.random_offset(),
        256,
        cuda.execution_stream());

    const int blocks = (count + 255) / 256;
    map_unit_to_bounds_kernel<<<blocks, 256, 0, cuda.execution_stream()>>>(
        variables_.data(),
        bounds_.data(),
        count,
        dimension_);
    CUDA_CHECK(cudaGetLastError());
}

void Population::initialize_values(
    const std::vector<float>& values,
    cudaStream_t stream)
{
    const auto expected = static_cast<std::size_t>(size_) * dimension_;
    if (values.size() != expected) {
        throw std::invalid_argument(
            "Initial population size does not match population_size * dimension");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        variables_.data(),
        values.data(),
        expected * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
}

void Population::clear_evaluation(cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(
        objectives_.data(),
        0,
        objectives_.size() * sizeof(float),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        constraints_.data(),
        0,
        constraints_.size() * sizeof(float),
        stream));
}

PopulationView Population::view() noexcept
{
    return {
        variables_.data(),
        objectives_.data(),
        constraints_.data(),
        bounds_.data(),
        size_,
        dimension_,
        objective_count_
    };
}

ConstPopulationView Population::view() const noexcept
{
    return {
        variables_.data(),
        objectives_.data(),
        constraints_.data(),
        bounds_.data(),
        size_,
        dimension_,
        objective_count_
    };
}

} // namespace cuda_moea
