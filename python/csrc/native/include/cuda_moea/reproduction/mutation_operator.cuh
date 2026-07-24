#pragma once

#include "cuda_moea/core/contexts.cuh"
#include "cuda_moea/core/device_buffer.cuh"
#include "cuda_moea/core/types.cuh"

namespace cuda_moea {

class IMutationOperator {
public:
    virtual ~IMutationOperator() = default;
    virtual void initialize(const AlgorithmInfo&, CudaContext&) {}
    virtual void mutate(
        PopulationView offspring,
        const GenerationContext& context,
        cudaStream_t stream) = 0;
};

struct PolynomialMutationConfig {
    float eta_initial = 20.0f;
    float eta_final = 20.0f;
    float probability = 1.0f;
};

class PolynomialMutation final : public IMutationOperator {
public:
    explicit PolynomialMutation(PolynomialMutationConfig config = {});
    explicit PolynomialMutation(
        float distribution_index,
        float probability = 1.0f);
    void initialize(const AlgorithmInfo& info, CudaContext& cuda) override;
    void mutate(
        PopulationView offspring,
        const GenerationContext& context,
        cudaStream_t stream) override;

private:
    PolynomialMutationConfig config_;
    DeviceBuffer<float2> random_values_;
};

class NoMutation final : public IMutationOperator {
public:
    void mutate(
        PopulationView,
        const GenerationContext&,
        cudaStream_t) override {}
};

} // namespace cuda_moea
