#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cuda_moea/core/population.cuh"
#include "cuda_moea/problem/problem_evaluator.cuh"
#include "cuda_moea/reference/reference_direction_provider.cuh"
#include "cuda_moea/reproduction/crossover_operator.cuh"
#include "cuda_moea/reproduction/mating_selector.cuh"
#include "cuda_moea/reproduction/mutation_operator.cuh"
#include "cuda_moea/selection/environment_selector.cuh"

namespace cuda_moea {

struct DataSaveConfig {
    bool enabled = false;
    std::string output_directory = "cuda_moea_data";
    int generation_interval = 1;
};

struct AlgorithmConfig {
    int population_size = 1024;
    int max_generations = 100;
    int progress_interval = 100;
    bool print_progress = true;
    CudaConfig cuda;
    DataSaveConfig data_save;
    // Optional row-major (population_size, dimension) initial decision matrix.
    // An empty vector retains the native random initialization path.
    std::vector<float> initial_population;
};

struct AlgorithmStrategies {
    std::unique_ptr<IProblemEvaluator> problem;
    std::unique_ptr<IMatingSelector> mating;
    std::unique_ptr<ICrossoverOperator> crossover;
    std::unique_ptr<IMutationOperator> mutation;
    std::unique_ptr<IReferenceDirectionProvider> reference_directions;
    std::unique_ptr<IEnvironmentSelector> environment_selector;
};

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    virtual void initialize() = 0;
    virtual void step() = 0;
    virtual RunResult run() = 0;
    virtual DeviceRunResult run_device() = 0;
    virtual RunResult result() = 0;
    virtual DeviceRunResult device_result() = 0;
    virtual void synchronize() = 0;
    virtual void reset() = 0;

    virtual bool initialized() const noexcept = 0;
    virtual bool finished() const noexcept = 0;
    virtual int generation() const noexcept = 0;

    virtual const Population& population() const noexcept = 0;
    virtual Population& population() noexcept = 0;
};

// Template-method implementation for generational evolutionary algorithms.
// The public lifecycle is fixed; derived algorithms may customize protected hooks.
class EvolutionaryAlgorithm : public IAlgorithm {
public:
    EvolutionaryAlgorithm(
        AlgorithmConfig config,
        AlgorithmStrategies strategies,
        std::string name = {});
    ~EvolutionaryAlgorithm() override;

    EvolutionaryAlgorithm(const EvolutionaryAlgorithm&) = delete;
    EvolutionaryAlgorithm& operator=(const EvolutionaryAlgorithm&) = delete;
    EvolutionaryAlgorithm(EvolutionaryAlgorithm&&) noexcept;
    EvolutionaryAlgorithm& operator=(EvolutionaryAlgorithm&&) noexcept;

    void initialize() final;
    void step() final;
    RunResult run() final;
    DeviceRunResult run_device() final;
    RunResult result() final;
    DeviceRunResult device_result() final;
    void synchronize() final;
    void reset() final;

    bool initialized() const noexcept final;
    bool finished() const noexcept final;
    int generation() const noexcept final;

    const Population& population() const noexcept final;
    Population& population() noexcept final;

protected:
    virtual void validate_algorithm(
        const AlgorithmConfig&,
        const ProblemInfo&) const {}
    virtual void before_initialization() {}
    virtual void after_initialization() {}
    virtual void before_generation(const GenerationContext&) {}
    virtual void after_evaluation(const GenerationContext&) {}
    virtual void after_selection(const GenerationContext&) {}
    virtual void before_reset() {}
    virtual void after_reset() {}

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Small type-erased value facade returned by every builder.
class Algorithm {
public:
    explicit Algorithm(std::unique_ptr<IAlgorithm> implementation);
    ~Algorithm();

    Algorithm(const Algorithm&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;
    Algorithm(Algorithm&&) noexcept;
    Algorithm& operator=(Algorithm&&) noexcept;

    void initialize();
    void step();
    RunResult run();
    DeviceRunResult run_device();
    RunResult result();
    DeviceRunResult device_result();
    void synchronize();
    void reset();

    bool initialized() const noexcept;
    bool finished() const noexcept;
    int generation() const noexcept;

    const Population& population() const noexcept;
    Population& population() noexcept;

private:
    std::unique_ptr<IAlgorithm> implementation_;
};

} // namespace cuda_moea
