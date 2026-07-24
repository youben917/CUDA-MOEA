#pragma once

#include <memory>

#include "cuda_moea/core/contexts.cuh"
#include "cuda_moea/core/types.cuh"

namespace cuda_moea {

class IProblemEvaluator {
public:
    virtual ~IProblemEvaluator() = default;

    virtual const ProblemInfo& info() const noexcept = 0;
    virtual void initialize(CudaContext&) {}

    virtual void evaluate(
        PopulationView population,
        const EvaluationContext& context,
        cudaStream_t stream) = 0;

    virtual bool prepare_parent(
        PopulationView,
        const EvaluationContext&,
        cudaStream_t)
    {
        return false;
    }

    virtual void reset() {}
};

enum class DTLZProblemType {
    DTLZ1 = 1,
    DTLZ2 = 2,
    DTLZ3 = 3,
    DTLZ4 = 4,
    DTLZ5 = 5,
    DTLZ6 = 6,
    DTLZ7 = 7,
    ConvexDTLZ2 = 8,
    C1DTLZ1 = 9,
    C1DTLZ3 = 10,
    C2DTLZ2 = 11,
    C2ConvexDTLZ2 = 12,
    C3DTLZ1 = 13,
    C3DTLZ4 = 14,
    CSDP = 15
};

struct DTLZProblemConfig {
    DTLZProblemType type = DTLZProblemType::DTLZ2;
    int dimension = 12;
    int objective_count = 3;
    float constraint_activation_ratio = 0.0f;
};

class DTLZProblem final : public IProblemEvaluator {
public:
    explicit DTLZProblem(DTLZProblemConfig config);
    ~DTLZProblem() override;

    DTLZProblem(const DTLZProblem&) = delete;
    DTLZProblem& operator=(const DTLZProblem&) = delete;

    const ProblemInfo& info() const noexcept override;
    void initialize(CudaContext& cuda) override;
    void evaluate(
        PopulationView population,
        const EvaluationContext& context,
        cudaStream_t stream) override;
    bool prepare_parent(
        PopulationView population,
        const EvaluationContext& context,
        cudaStream_t stream) override;
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuda_moea
