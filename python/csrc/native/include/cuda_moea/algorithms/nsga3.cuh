#pragma once

#include <memory>

#include "cuda_moea/builder/basic_algorithm_builder.cuh"

namespace cuda_moea {

struct CVQuantizationConfig {
    int bins = 0;
    float clip_upper = 0.0f;
    float log_alpha = 0.0f;
    float feasibility_epsilon = 0.0f;
};

struct NSGA3SelectionConfig {
    float sparse_ratio = 0.5f;
    CVQuantizationConfig cv_quantization;
};

class NSGA3EnvironmentSelector final : public IEnvironmentSelector {
public:
    explicit NSGA3EnvironmentSelector(NSGA3SelectionConfig config = {});
    ~NSGA3EnvironmentSelector() override;

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
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NSGA3 {
public:
    class Builder final
        : public BasicAlgorithmBuilder<Builder> {
    public:
        Builder();

        Builder& selection(NSGA3SelectionConfig config);
        Builder& sparseRatio(float value);
        Builder& referencePointPartition(int partitions);

        Algorithm build();

    private:
        NSGA3SelectionConfig selection_config_;
    };
};

} // namespace cuda_moea
