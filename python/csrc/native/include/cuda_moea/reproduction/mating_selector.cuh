#pragma once

#include "cuda_moea/core/contexts.cuh"
#include "cuda_moea/core/device_buffer.cuh"
#include "cuda_moea/core/types.cuh"

namespace cuda_moea {

class IMatingSelector {
public:
    virtual ~IMatingSelector() = default;
    virtual void initialize(const AlgorithmInfo&, CudaContext&) {}
    virtual void select(
        ConstPopulationView parents,
        MatingStateView state,
        DeviceSpan<int> parent_indices,
        const GenerationContext& context,
        cudaStream_t stream) = 0;
};

class RandomMating final : public IMatingSelector {
public:
    void select(
        ConstPopulationView parents,
        MatingStateView state,
        DeviceSpan<int> parent_indices,
        const GenerationContext& context,
        cudaStream_t stream) override;
};

class TournamentMating final : public IMatingSelector {
public:
    void initialize(const AlgorithmInfo& info, CudaContext& cuda) override;
    void select(
        ConstPopulationView parents,
        MatingStateView state,
        DeviceSpan<int> parent_indices,
        const GenerationContext& context,
        cudaStream_t stream) override;

private:
    DeviceBuffer<int2> candidates_;
};

} // namespace cuda_moea
