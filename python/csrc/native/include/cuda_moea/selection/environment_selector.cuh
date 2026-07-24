#pragma once

#include "cuda_moea/core/contexts.cuh"
#include "cuda_moea/core/types.cuh"
#include "cuda_moea/reference/reference_direction_provider.cuh"

namespace cuda_moea {

class IEnvironmentSelector {
public:
    virtual ~IEnvironmentSelector() = default;

    virtual void initialize(
        const AlgorithmInfo& info,
        IReferenceDirectionProvider* references,
        CudaContext& cuda) = 0;

    virtual void prepare(
        ConstPopulationView population,
        const GenerationContext& context) = 0;

    virtual void select(
        ConstPopulationView parents,
        ConstPopulationView offspring,
        PopulationView next,
        const GenerationContext& context) = 0;

    virtual MatingStateView mating_state() const noexcept { return {}; }
    virtual int active_count() const noexcept = 0;
    virtual DeviceSpan<const int> result_indices() const noexcept { return {}; }
    virtual void finalize(ConstPopulationView, const GenerationContext&) {}
    virtual void reset() = 0;
};

} // namespace cuda_moea
