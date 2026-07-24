#pragma once

#include <vector>

#include "cuda_moea/core/cuda_context.cuh"
#include "cuda_moea/core/device_buffer.cuh"
#include "cuda_moea/core/types.cuh"

namespace cuda_moea {

class Population {
public:
    Population() = default;
    Population(
        int size,
        int dimension,
        int objective_count,
        CudaContext& cuda);

    Population(const Population&) = delete;
    Population& operator=(const Population&) = delete;
    Population(Population&&) noexcept = default;
    Population& operator=(Population&&) noexcept = default;

    void allocate(
        int size,
        int dimension,
        int objective_count,
        CudaContext& cuda);

    void set_bounds(
        const std::vector<float>& lower,
        const std::vector<float>& upper,
        cudaStream_t stream);

    void initialize_random(CudaContext& cuda);
    void initialize_values(
        const std::vector<float>& values,
        cudaStream_t stream);
    void clear_evaluation(cudaStream_t stream);

    PopulationView view() noexcept;
    ConstPopulationView view() const noexcept;

    int size() const noexcept { return size_; }
    int dimension() const noexcept { return dimension_; }
    int objective_count() const noexcept { return objective_count_; }

private:
    DeviceBuffer<float> variables_;
    DeviceBuffer<float> objectives_;
    DeviceBuffer<float> constraints_;
    DeviceBuffer<float> bounds_;
    int size_ = 0;
    int dimension_ = 0;
    int objective_count_ = 0;
};

} // namespace cuda_moea
