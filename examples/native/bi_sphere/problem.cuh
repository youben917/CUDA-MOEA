#pragma once
#include "cuda_moea/native/plugin.cuh"

class BiSphere final : public cuda_moea::IProblemEvaluator {
public:
    explicit BiSphere(const cuda_moea::native::ProblemConfig& config);
    const cuda_moea::ProblemInfo& info() const noexcept override { return info_; }
    void evaluate(cuda_moea::PopulationView population,
                  const cuda_moea::EvaluationContext& context,
                  cudaStream_t stream) override;
private:
    cuda_moea::ProblemInfo info_;
    float offset_, radius_;
    bool constrained_;
};
