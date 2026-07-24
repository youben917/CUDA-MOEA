#pragma once

#include <memory>

#include "cuda_moea/factory/kinds.cuh"
#include "cuda_moea/reference/reference_direction_provider.cuh"
#include "cuda_moea/reproduction/crossover_operator.cuh"
#include "cuda_moea/reproduction/mating_selector.cuh"
#include "cuda_moea/reproduction/mutation_operator.cuh"
#include "cuda_moea/selection/environment_selector.cuh"

namespace cuda_moea {

class OperatorFactory {
public:
    static std::unique_ptr<IMatingSelector> mating(MatingKind kind);

    static std::unique_ptr<ICrossoverOperator> crossover(
        CrossoverKind kind);
    static std::unique_ptr<ICrossoverOperator> crossover(
        CrossoverKind kind,
        float distribution_index,
        float probability = 1.0f);
    static std::unique_ptr<ICrossoverOperator> crossover(
        CrossoverKind kind,
        SBXConfig config);

    static std::unique_ptr<IMutationOperator> mutation(
        MutationKind kind);
    static std::unique_ptr<IMutationOperator> mutation(
        MutationKind kind,
        float distribution_index,
        float probability);
    static std::unique_ptr<IMutationOperator> mutation(
        MutationKind kind,
        PolynomialMutationConfig config);

    static std::unique_ptr<IReferenceDirectionProvider>
    reference_directions(ReferenceDirectionKind kind);

    static std::unique_ptr<IEnvironmentSelector>
    environment_selector(EnvironmentSelectorKind kind);
};

} // namespace cuda_moea
