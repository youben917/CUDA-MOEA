#pragma once

#include <memory>

#include "cuda_moea/builder/basic_algorithm_builder.cuh"

namespace cuda_moea {

struct RVEASelectionConfig {
    float alpha = 2.0f;
};

class RVEAEnvironmentSelector final : public IEnvironmentSelector {
public:
    explicit RVEAEnvironmentSelector(RVEASelectionConfig config = {});
    ~RVEAEnvironmentSelector() override;

    void initialize(
        const AlgorithmInfo& info,
        IReferenceDirectionProvider* references,
        CudaContext& cuda) override;
    void prepare(
        ConstPopulationView population,
        const GenerationContext& context) override;
    void select(
        ConstPopulationView parents,
        ConstPopulationView offspring,
        PopulationView next,
        const GenerationContext& context) override;
    MatingStateView mating_state() const noexcept override;
    int active_count() const noexcept override;
    DeviceSpan<const int> result_indices() const noexcept override;
    void finalize(
        ConstPopulationView population,
        const GenerationContext& context) override;
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RVEA {
public:
    class Builder final
        : public BasicAlgorithmBuilder<Builder> {
    public:
        Builder();

        Builder& selection(RVEASelectionConfig config);
        Builder& selectionAlpha(float value);
        Builder& referenceAdaptation(AdaptiveDirectionConfig config);
        Builder& referenceAdaptationFrequency(float value);

        Algorithm build();

    private:
        RVEASelectionConfig selection_config_;
        AdaptiveDirectionConfig reference_config_;
    };
};

} // namespace cuda_moea
