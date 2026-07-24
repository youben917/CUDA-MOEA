#pragma once

#include <memory>

#include "cuda_moea/factory/kinds.cuh"
#include "cuda_moea/problem/problem_evaluator.cuh"

namespace cuda_moea {

class ProblemFactory {
public:
    static std::unique_ptr<IProblemEvaluator> create(
        ProblemKind kind,
        int dimension,
        int objective_count,
        float constraint_activation_ratio = 0.0f);

    static std::unique_ptr<IProblemEvaluator> create(
        DTLZProblemConfig config);
};

} // namespace cuda_moea
