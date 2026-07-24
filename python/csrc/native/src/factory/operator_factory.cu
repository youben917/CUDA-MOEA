#include "cuda_moea/factory/operator_factory.cuh"

#include <stdexcept>

#include "cuda_moea/algorithms/nsga3.cuh"
#include "cuda_moea/algorithms/rvea.cuh"

namespace cuda_moea {

std::unique_ptr<IMatingSelector>
OperatorFactory::mating(MatingKind kind)
{
    switch (kind) {
    case MatingKind::Random:
        return std::make_unique<RandomMating>();
    case MatingKind::Tournament:
        return std::make_unique<TournamentMating>();
    }
    throw std::invalid_argument("Unsupported mating kind");
}

std::unique_ptr<ICrossoverOperator>
OperatorFactory::crossover(CrossoverKind kind)
{
    return crossover(kind, SBXConfig{});
}

std::unique_ptr<ICrossoverOperator>
OperatorFactory::crossover(
    CrossoverKind kind,
    float distribution_index,
    float probability)
{
    return crossover(kind, SBXConfig{
        distribution_index,
        distribution_index,
        probability
    });
}

std::unique_ptr<ICrossoverOperator>
OperatorFactory::crossover(
    CrossoverKind kind,
    SBXConfig config)
{
    switch (kind) {
    case CrossoverKind::SBX:
        return std::make_unique<SBXCrossover>(config);
    }
    throw std::invalid_argument("Unsupported crossover kind");
}

std::unique_ptr<IMutationOperator>
OperatorFactory::mutation(MutationKind kind)
{
    switch (kind) {
    case MutationKind::Polynomial:
        return std::make_unique<PolynomialMutation>();
    case MutationKind::None:
        return std::make_unique<NoMutation>();
    }
    throw std::invalid_argument("Unsupported mutation kind");
}

std::unique_ptr<IMutationOperator>
OperatorFactory::mutation(
    MutationKind kind,
    float distribution_index,
    float probability)
{
    return mutation(kind, PolynomialMutationConfig{
        distribution_index,
        distribution_index,
        probability
    });
}

std::unique_ptr<IMutationOperator>
OperatorFactory::mutation(
    MutationKind kind,
    PolynomialMutationConfig config)
{
    switch (kind) {
    case MutationKind::Polynomial:
        return std::make_unique<PolynomialMutation>(config);
    case MutationKind::None:
        throw std::invalid_argument(
            "NoMutation does not accept polynomial-mutation configuration");
    }
    throw std::invalid_argument("Unsupported mutation kind");
}

std::unique_ptr<IReferenceDirectionProvider>
OperatorFactory::reference_directions(ReferenceDirectionKind kind)
{
    switch (kind) {
    case ReferenceDirectionKind::DasDennis:
        return std::make_unique<DasDennisDirections>();
    case ReferenceDirectionKind::AdaptiveRVEA:
        return std::make_unique<AdaptiveRVEADirections>();
    }
    throw std::invalid_argument(
        "Unsupported reference-direction kind");
}

std::unique_ptr<IEnvironmentSelector>
OperatorFactory::environment_selector(EnvironmentSelectorKind kind)
{
    switch (kind) {
    case EnvironmentSelectorKind::NSGA3:
        return std::make_unique<NSGA3EnvironmentSelector>();
    case EnvironmentSelectorKind::RVEA:
        return std::make_unique<RVEAEnvironmentSelector>();
    }
    throw std::invalid_argument(
        "Unsupported environment-selector kind");
}

} // namespace cuda_moea
