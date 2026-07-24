#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "cuda_moea/factory/algorithm_factory.cuh"
#include "cuda_moea/factory/operator_factory.cuh"
#include "cuda_moea/factory/problem_factory.cuh"

namespace cuda_moea {

template <typename Derived>
class BasicAlgorithmBuilder {
public:
    Derived& populationSize(int value) {
        state_.config.population_size = value;
        return self();
    }

    Derived& maxGenerations(int value) {
        state_.config.max_generations = value;
        return self();
    }

    Derived& progressInterval(int value) {
        state_.config.progress_interval = value;
        return self();
    }

    Derived& printProgress(bool value) {
        state_.config.print_progress = value;
        return self();
    }

    Derived& cudaDevice(int device_id) {
        state_.config.cuda.device_id = device_id;
        return self();
    }

    Derived& cudaWarmup(bool enabled = true) {
        state_.config.cuda.enable_warmup = enabled;
        return self();
    }

    Derived& randomSeed(unsigned long long seed) {
        state_.config.cuda.seed = seed;
        return self();
    }

    Derived& saveData(
        std::string output_directory,
        int generation_interval = 1) {
        if (output_directory.empty()) {
            throw std::invalid_argument(
                "saveData output directory cannot be empty");
        }
        if (generation_interval <= 0) {
            throw std::invalid_argument(
                "saveData generation interval must be positive");
        }
        state_.config.data_save.enabled = true;
        state_.config.data_save.output_directory =
            std::move(output_directory);
        state_.config.data_save.generation_interval =
            generation_interval;
        return self();
    }

    Derived& cudaConfig(CudaConfig config) {
        state_.config.cuda = config;
        return self();
    }

    Derived& problem(
        ProblemKind kind,
        int dimension,
        int objective_count,
        float constraint_activation_ratio = 0.0f) {
        return problem(ProblemFactory::create(
            kind,
            dimension,
            objective_count,
            constraint_activation_ratio));
    }

    Derived& problem(DTLZProblemConfig config) {
        return problem(ProblemFactory::create(config));
    }

    Derived& problem(std::unique_ptr<IProblemEvaluator> value) {
        require_non_null(value, "problem");
        state_.strategies.problem = std::move(value);
        return self();
    }

    Derived& mating(MatingKind kind) {
        return mating(OperatorFactory::mating(kind));
    }

    Derived& mating(std::unique_ptr<IMatingSelector> value) {
        require_non_null(value, "mating selector");
        state_.strategies.mating = std::move(value);
        return self();
    }

    Derived& crossover(CrossoverKind kind) {
        return crossover(OperatorFactory::crossover(kind));
    }

    Derived& crossover(
        CrossoverKind kind,
        float distribution_index,
        float probability = 1.0f) {
        return crossover(OperatorFactory::crossover(
            kind,
            distribution_index,
            probability));
    }

    Derived& crossover(
        CrossoverKind kind,
        SBXConfig config) {
        return crossover(OperatorFactory::crossover(kind, config));
    }

    Derived& crossover(std::unique_ptr<ICrossoverOperator> value) {
        require_non_null(value, "crossover operator");
        state_.strategies.crossover = std::move(value);
        return self();
    }

    Derived& mutation(MutationKind kind) {
        return mutation(OperatorFactory::mutation(kind));
    }

    Derived& mutation(
        MutationKind kind,
        float distribution_index,
        float probability = 1.0f) {
        return mutation(OperatorFactory::mutation(
            kind,
            distribution_index,
            probability));
    }

    Derived& mutation(
        MutationKind kind,
        PolynomialMutationConfig config) {
        return mutation(OperatorFactory::mutation(kind, config));
    }

    Derived& mutation(std::unique_ptr<IMutationOperator> value) {
        require_non_null(value, "mutation operator");
        state_.strategies.mutation = std::move(value);
        return self();
    }

    Derived& referenceDirections(ReferenceDirectionKind kind) {
        return referenceDirections(
            OperatorFactory::reference_directions(kind));
    }

    Derived& referenceDirections(
        std::unique_ptr<IReferenceDirectionProvider> value) {
        require_non_null(value, "reference-direction provider");
        state_.strategies.reference_directions = std::move(value);
        return self();
    }

    Derived& environmentSelector(EnvironmentSelectorKind kind) {
        return environmentSelector(
            OperatorFactory::environment_selector(kind));
    }

    Derived& environmentSelector(
        std::unique_ptr<IEnvironmentSelector> value) {
        require_non_null(value, "environment selector");
        state_.strategies.environment_selector = std::move(value);
        return self();
    }

protected:
    explicit BasicAlgorithmBuilder(AlgorithmKind kind)
        : state_(AlgorithmFactory::defaults(kind)) {}

    Algorithm buildConfigured() {
        return AlgorithmFactory::create(std::move(state_));
    }

    detail::AlgorithmBuildState state_;

private:
    Derived& self() noexcept {
        return static_cast<Derived&>(*this);
    }

    template <typename T>
    static void require_non_null(
        const std::unique_ptr<T>& value,
        const char* name) {
        if (!value) {
            throw std::invalid_argument(
                std::string(name) + " cannot be null");
        }
    }
};

} // namespace cuda_moea
