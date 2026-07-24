#pragma once

#include <vector>

#include "cuda_moea/core/contexts.cuh"
#include "cuda_moea/core/device_buffer.cuh"
#include "cuda_moea/core/types.cuh"

namespace cuda_moea {

class IReferenceDirectionProvider {
public:
    virtual ~IReferenceDirectionProvider() = default;
    virtual void initialize(
        int requested_count,
        int objective_count,
        CudaContext& cuda) = 0;
    virtual ReferenceDirectionView view() noexcept = 0;
    virtual void update(
        ConstPopulationView,
        int,
        const GenerationContext&,
        cudaStream_t)
    {}
    virtual void reset(cudaStream_t) {}
};

struct DasDennisConfig {
    // Zero selects the largest one/two-layer set not exceeding population size.
    // A positive value generates one exact Das-Dennis layer with H partitions.
    int partitions = 0;
};

class DasDennisDirections final : public IReferenceDirectionProvider {
public:
    explicit DasDennisDirections(DasDennisConfig config = {});
    void initialize(
        int requested_count,
        int objective_count,
        CudaContext& cuda) override;
    ReferenceDirectionView view() noexcept override;

private:
    DasDennisConfig config_;
    DeviceBuffer<float> directions_;
    int objective_count_ = 0;
    int count_ = 0;
};

struct AdaptiveDirectionConfig {
    float frequency = 0.1f;
};

class AdaptiveRVEADirections final : public IReferenceDirectionProvider {
public:
    explicit AdaptiveRVEADirections(AdaptiveDirectionConfig config = {});
    void initialize(
        int requested_count,
        int objective_count,
        CudaContext& cuda) override;
    ReferenceDirectionView view() noexcept override;
    void update(
        ConstPopulationView population,
        int active_count,
        const GenerationContext& context,
        cudaStream_t stream) override;
    void reset(cudaStream_t stream) override;

private:
    AdaptiveDirectionConfig config_;
    DeviceBuffer<float> directions_;
    DeviceBuffer<float> initial_directions_;
    DeviceBuffer<float> zmin_;
    DeviceBuffer<float> zmax_;
    DeviceBuffer<float> gamma_block_max_;
    DeviceBuffer<float> gamma_;
    int objective_count_ = 0;
    int count_ = 0;
};

class UserDefinedDirections final : public IReferenceDirectionProvider {
public:
    UserDefinedDirections(
        std::vector<float> objective_major_values,
        int objective_count,
        bool normalize = true);

    void initialize(
        int requested_count,
        int objective_count,
        CudaContext& cuda) override;
    ReferenceDirectionView view() noexcept override;

private:
    std::vector<float> host_values_;
    int objective_count_ = 0;
    bool normalize_ = true;
    DeviceBuffer<float> directions_;
    int count_ = 0;
};

} // namespace cuda_moea
