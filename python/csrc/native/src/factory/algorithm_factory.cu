#include "cuda_moea/factory/algorithm_factory.cuh"

#include <stdexcept>
#include <utility>

#include "cuda_moea/algorithms/nsga3.cuh"
#include "cuda_moea/algorithms/rvea.cuh"
#include "cuda_moea/factory/operator_factory.cuh"

namespace cuda_moea {
namespace {

const char* algorithm_name(AlgorithmKind kind)
{
    switch (kind) {
    case AlgorithmKind::NSGA3: return "NSGA-III";
    case AlgorithmKind::RVEA: return "RVEA";
    }
    return "Unknown";
}

} // namespace

detail::AlgorithmBuildState
AlgorithmFactory::defaults(AlgorithmKind kind)
{
    detail::AlgorithmBuildState state{
        kind,
        AlgorithmConfig{},
        AlgorithmStrategies{}
    };

    switch (kind) {
    case AlgorithmKind::NSGA3:
        state.strategies.mating =
            OperatorFactory::mating(MatingKind::Tournament);
        state.strategies.crossover =
            OperatorFactory::crossover(CrossoverKind::SBX);
        state.strategies.mutation =
            OperatorFactory::mutation(MutationKind::Polynomial);
        state.strategies.reference_directions =
            OperatorFactory::reference_directions(
                ReferenceDirectionKind::DasDennis);
        state.strategies.environment_selector =
            std::make_unique<NSGA3EnvironmentSelector>();
        break;

    case AlgorithmKind::RVEA:
        state.strategies.mating =
            OperatorFactory::mating(MatingKind::Random);
        state.strategies.crossover =
            OperatorFactory::crossover(CrossoverKind::SBX);
        state.strategies.mutation =
            OperatorFactory::mutation(MutationKind::Polynomial);
        state.strategies.reference_directions =
            OperatorFactory::reference_directions(
                ReferenceDirectionKind::AdaptiveRVEA);
        state.strategies.environment_selector =
            std::make_unique<RVEAEnvironmentSelector>();
        break;
    }

    return state;
}

Algorithm AlgorithmFactory::create(detail::AlgorithmBuildState state)
{
    const auto& config = state.config;
    auto& strategies = state.strategies;

    if (config.population_size <= 0) {
        throw std::invalid_argument(
            "populationSize must be positive");
    }
    if (config.max_generations <= 0) {
        throw std::invalid_argument(
            "maxGenerations must be positive");
    }
    if (config.data_save.enabled &&
        config.data_save.output_directory.empty()) {
        throw std::invalid_argument(
            "saveData output directory cannot be empty");
    }
    if (config.data_save.generation_interval <= 0) {
        throw std::invalid_argument(
            "saveData generation interval must be positive");
    }
    if (!strategies.problem) {
        throw std::invalid_argument(
            "A problem must be configured before build()");
    }
    if (!strategies.mating ||
        !strategies.crossover ||
        !strategies.mutation ||
        !strategies.environment_selector) {
        throw std::invalid_argument(
            "Mating, crossover, mutation, and environment selection "
            "strategies are required");
    }
    if (!strategies.reference_directions) {
        throw std::invalid_argument(
            "The built-in NSGA-III and RVEA selectors require "
            "reference directions");
    }
    if (state.kind == AlgorithmKind::NSGA3 &&
        config.population_size % 2 != 0) {
        throw std::invalid_argument(
            "The current NSGA-III CUDA backend requires an even "
            "population size");
    }

    return Algorithm(std::make_unique<EvolutionaryAlgorithm>(
        state.config,
        std::move(strategies),
        algorithm_name(state.kind)));
}

} // namespace cuda_moea
