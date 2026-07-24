#pragma once

#include "cuda_moea/core/contexts.cuh"
#include "cuda_moea/core/device_buffer.cuh"
#include "cuda_moea/core/types.cuh"

namespace cuda_moea {

class ICrossoverOperator {
public:
    virtual ~ICrossoverOperator() = default;
    virtual void initialize(const AlgorithmInfo&, CudaContext&) {}
    virtual void apply(
        ConstPopulationView parents,
        DeviceSpan<const int> parent_indices,
        PopulationView offspring,
        const GenerationContext& context,
        cudaStream_t stream) = 0;
};

struct SBXConfig {
    float eta_initial = 30.0f;
    float eta_final = 30.0f;
    float probability = 1.0f;
    // Per-variable probability of retaining the two parental values after
    // the pair has been selected for crossover.  EvoX's SBX uses 0.5.
    float variable_copy_probability = 0.0f;
};

class SimulatedBinaryCrossover final : public ICrossoverOperator {
public:
    explicit SimulatedBinaryCrossover(SBXConfig config = {});
    explicit SimulatedBinaryCrossover(
        float distribution_index,
        float probability = 1.0f);
    void initialize(const AlgorithmInfo& info, CudaContext& cuda) override;
    void apply(
        ConstPopulationView parents,
        DeviceSpan<const int> parent_indices,
        PopulationView offspring,
        const GenerationContext& context,
        cudaStream_t stream) override;

private:
    SBXConfig config_;
    DeviceBuffer<float2> random_values_;
    DeviceBuffer<int> random_signs_;
    DeviceBuffer<float> copy_random_;
};

using SBXCrossover = SimulatedBinaryCrossover;

} // namespace cuda_moea
