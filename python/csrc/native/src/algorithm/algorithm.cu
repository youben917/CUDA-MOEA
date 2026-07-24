#include "cuda_moea/algorithm.cuh"

// #include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "cuda_moea/algorithm/data_recorder.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"
#include "cuda_moea/reproduction/reproduction_pipeline.cuh"

namespace cuda_moea {

struct EvolutionaryAlgorithm::Impl {
    AlgorithmConfig config;
    std::unique_ptr<CudaContext> cuda;
    std::unique_ptr<IProblemEvaluator> problem;
    std::unique_ptr<ReproductionPipeline> reproduction;
    std::unique_ptr<IReferenceDirectionProvider> reference_directions;
    std::unique_ptr<IEnvironmentSelector> environment_selector;
    detail::DataRecorder data_recorder;
    std::string name;
    Population parents;
    Population offspring;
    Population next;
    AlgorithmInfo info;
    int generation = 0;
    bool initialized = false;
    double total_ms = 0.0;

    Impl(
        AlgorithmConfig run_config,
        AlgorithmStrategies strategies,
        std::string algorithm_name)
        : config(std::move(run_config)),
          problem(std::move(strategies.problem)),
          reproduction(std::make_unique<ReproductionPipeline>(
              std::move(strategies.mating),
              std::move(strategies.crossover),
              std::move(strategies.mutation))),
          reference_directions(
              std::move(strategies.reference_directions)),
          environment_selector(
              std::move(strategies.environment_selector)),
          data_recorder(config.data_save),
          name(std::move(algorithm_name))
    {
        if (!problem || !environment_selector) {
            throw std::invalid_argument(
                "EvolutionaryAlgorithm requires a problem and "
                "an environment selector");
        }
    }

    GenerationContext generation_context() {
        return {generation, config.max_generations, *cuda};
    }

    EvaluationContext evaluation_context(int value) {
        return {value, config.max_generations, *cuda};
    }

    ReferenceDirectionView reference_view() {
        return reference_directions
            ? reference_directions->view()
            : ReferenceDirectionView{};
    }

    void save_snapshot(
        int value,
        ConstPopulationView population)
    {
        if (!data_recorder.should_save(
                value,
                config.max_generations)) {
            return;
        }
        data_recorder.save_snapshot(
            value,
            population,
            reference_view(),
            environment_selector->result_indices(),
            environment_selector->active_count(),
            *cuda);
    }
};

EvolutionaryAlgorithm::EvolutionaryAlgorithm(
    AlgorithmConfig config,
    AlgorithmStrategies strategies,
    std::string name)
    : impl_(std::make_unique<Impl>(
          config,
          std::move(strategies),
          std::move(name))) {}

EvolutionaryAlgorithm::~EvolutionaryAlgorithm() = default;
EvolutionaryAlgorithm::EvolutionaryAlgorithm(
    EvolutionaryAlgorithm&&) noexcept = default;
EvolutionaryAlgorithm& EvolutionaryAlgorithm::operator=(
    EvolutionaryAlgorithm&&) noexcept = default;

void EvolutionaryAlgorithm::initialize()
{
    if (impl_->initialized) return;
    if (impl_->config.population_size <= 0 ||
        impl_->config.max_generations <= 0) {
        throw std::invalid_argument(
            "Algorithm requires positive population_size and max_generations");
    }

    before_initialization();
    impl_->cuda = std::make_unique<CudaContext>(
        impl_->config.cuda);

    auto& problem = *impl_->problem;
    problem.initialize(*impl_->cuda);
    const auto& problem_info = problem.info();
    if (problem_info.dimension <= 0 ||
        problem_info.objective_count <= 0 ||
        static_cast<int>(problem_info.lower_bounds.size()) !=
            problem_info.dimension ||
        static_cast<int>(problem_info.upper_bounds.size()) !=
            problem_info.dimension) {
        throw std::invalid_argument(
            "Problem must provide positive dimensions and one lower/upper "
            "bound per decision variable");
    }
    validate_algorithm(impl_->config, problem_info);

    impl_->info = {
        impl_->name,
        impl_->config.population_size,
        problem_info.dimension,
        problem_info.objective_count,
        impl_->config.max_generations
    };

    for (Population* population :
         {&impl_->parents, &impl_->offspring, &impl_->next}) {
        population->allocate(
            impl_->info.population_size,
            impl_->info.dimension,
            impl_->info.objective_count,
            *impl_->cuda);
        population->set_bounds(
            problem_info.lower_bounds,
            problem_info.upper_bounds,
            impl_->cuda->execution_stream());
        population->clear_evaluation(
            impl_->cuda->evaluation_stream());
    }

    if (impl_->config.initial_population.empty()) {
        impl_->parents.initialize_random(*impl_->cuda);
    } else {
        const auto expected = static_cast<std::size_t>(impl_->info.population_size)
            * impl_->info.dimension;
        if (impl_->config.initial_population.size() != expected) {
            throw std::invalid_argument(
                "initial_population must have shape (population_size, dimension)");
        }
        for (int row = 0; row < impl_->info.population_size; ++row) {
            for (int column = 0; column < impl_->info.dimension; ++column) {
                const float value = impl_->config.initial_population[
                    static_cast<std::size_t>(row) * impl_->info.dimension + column];
                if (!std::isfinite(value) ||
                    value < problem_info.lower_bounds[column] ||
                    value > problem_info.upper_bounds[column]) {
                    throw std::invalid_argument(
                        "initial_population contains a non-finite or out-of-bounds value");
                }
            }
        }
        impl_->parents.initialize_values(
            impl_->config.initial_population,
            impl_->cuda->execution_stream());
    }
    impl_->cuda->wait_execution_on_evaluation();
    problem.evaluate(
        impl_->parents.view(),
        impl_->evaluation_context(-1),
        impl_->cuda->evaluation_stream());

    if (impl_->reference_directions) {
        impl_->reference_directions->initialize(
            impl_->info.population_size,
            impl_->info.objective_count,
            *impl_->cuda);
    }

    impl_->reproduction->initialize(impl_->info, *impl_->cuda);
    impl_->environment_selector->initialize(
        impl_->info,
        impl_->reference_directions.get(),
        *impl_->cuda);

    impl_->cuda->wait_evaluation_on_execution();
    impl_->environment_selector->prepare(
        impl_->parents.view(),
        impl_->generation_context());

    impl_->generation = 0;
    impl_->total_ms = 0.0;
    impl_->initialized = true;

    impl_->data_recorder.initialize(
        impl_->info,
        problem_info,
        impl_->config.cuda,
        impl_->reference_view(),
        *impl_->cuda);
    impl_->save_snapshot(0, impl_->parents.view());

    after_initialization();
}

void EvolutionaryAlgorithm::step()
{
    if (!impl_->initialized) initialize();
    if (finished()) return;

    auto generation_context = impl_->generation_context();
    auto evaluation_context =
        impl_->evaluation_context(impl_->generation);

    before_generation(generation_context);

    if (impl_->problem->prepare_parent(
            impl_->parents.view(),
            evaluation_context,
            impl_->cuda->evaluation_stream())) {
        impl_->cuda->wait_evaluation_on_execution();
        impl_->environment_selector->prepare(
            impl_->parents.view(),
            generation_context);
    }

    impl_->reproduction->generate(
        impl_->parents.view(),
        impl_->environment_selector->mating_state(),
        impl_->offspring.view(),
        generation_context);

    impl_->cuda->wait_execution_on_evaluation();
    impl_->problem->evaluate(
        impl_->offspring.view(),
        evaluation_context,
        impl_->cuda->evaluation_stream());
    impl_->cuda->wait_evaluation_on_execution();

    after_evaluation(generation_context);

    impl_->environment_selector->select(
        impl_->parents.view(),
        impl_->offspring.view(),
        impl_->next.view(),
        generation_context);

    if (impl_->reference_directions) {
        impl_->reference_directions->update(
            impl_->next.view(),
            impl_->environment_selector->active_count(),
            generation_context,
            impl_->cuda->execution_stream());
    }

    after_selection(generation_context);

    impl_->save_snapshot(
        impl_->generation + 1,
        impl_->next.view());

    std::swap(impl_->parents, impl_->next);
    ++impl_->generation;

    if (impl_->config.print_progress &&
        (impl_->generation == impl_->config.max_generations ||
         (impl_->config.progress_interval > 0 &&
          impl_->generation % impl_->config.progress_interval == 0))) {
        std::cout << "Generation " << impl_->generation
                  << " / " << impl_->config.max_generations
                  << " completed\n";
    }
}

RunResult EvolutionaryAlgorithm::run()
{
    if (!impl_->initialized) initialize();

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, impl_->cuda->execution_stream());
    // const auto begin = std::chrono::steady_clock::now();
    while (!finished()) {
        step();
    }
    cudaEventRecord(stop, impl_->cuda->execution_stream());
    cudaEventSynchronize(stop);
    // const auto end = std::chrono::steady_clock::now();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    impl_->total_ms = elapsedTime;
        // std::chrono::duration<double, std::milli>(end - begin).count();
    return result();
}

DeviceRunResult EvolutionaryAlgorithm::run_device()
{
    if (!impl_->initialized) initialize();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, impl_->cuda->execution_stream()));
    while (!finished()) step();
    CUDA_CHECK(cudaEventRecord(stop, impl_->cuda->execution_stream()));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    impl_->total_ms = elapsed_ms;
    return device_result();
}

DeviceRunResult EvolutionaryAlgorithm::device_result()
{
    if (!impl_->initialized) {
        throw std::logic_error("Algorithm has not been initialized");
    }
    impl_->environment_selector->finalize(
        impl_->parents.view(), impl_->generation_context());
    impl_->cuda->synchronize();
    impl_->save_snapshot(impl_->generation, impl_->parents.view());
    return {
        impl_->parents.view(),
        impl_->environment_selector->result_indices(),
        impl_->environment_selector->active_count(),
        impl_->total_ms,
        impl_->config.cuda.device_id
    };
}

void EvolutionaryAlgorithm::synchronize()
{
    if (impl_->cuda) impl_->cuda->synchronize();
}

RunResult EvolutionaryAlgorithm::result()
{
    if (!impl_->initialized) {
        throw std::logic_error("Algorithm has not been initialized");
    }

    impl_->environment_selector->finalize(
        impl_->parents.view(),
        impl_->generation_context());
    impl_->cuda->synchronize();
    impl_->save_snapshot(
        impl_->generation,
        impl_->parents.view());

    const auto population = impl_->parents.view();
    RunResult output;
    output.population.resize(
        static_cast<std::size_t>(population.size) * population.dimension);
    output.objectives.resize(
        static_cast<std::size_t>(population.objective_count) * population.size);
    output.constraints.resize(population.size);
    output.active_count = impl_->environment_selector->active_count();
    output.total_ms = impl_->total_ms;

    CUDA_CHECK(cudaMemcpy(
        output.population.data(),
        population.variables,
        output.population.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        output.objectives.data(),
        population.objectives,
        output.objectives.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        output.constraints.data(),
        population.constraints,
        output.constraints.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    const auto auxiliary =
        impl_->environment_selector->result_indices();
    if (!auxiliary.empty()) {
        output.auxiliary_indices.resize(auxiliary.size);
        CUDA_CHECK(cudaMemcpy(
            output.auxiliary_indices.data(),
            auxiliary.data,
            auxiliary.size * sizeof(int),
            cudaMemcpyDeviceToHost));
    }
    return output;
}

void EvolutionaryAlgorithm::reset()
{
    if (!impl_->initialized) return;
    before_reset();
    impl_->cuda->synchronize();
    impl_->environment_selector->reset();
    impl_->problem->reset();
    if (impl_->reference_directions) {
        impl_->reference_directions->reset(
            impl_->cuda->execution_stream());
    }
    impl_->parents.initialize_random(*impl_->cuda);
    impl_->cuda->wait_execution_on_evaluation();
    impl_->problem->evaluate(
        impl_->parents.view(),
        impl_->evaluation_context(-1),
        impl_->cuda->evaluation_stream());
    impl_->cuda->wait_evaluation_on_execution();
    impl_->generation = 0;
    impl_->environment_selector->prepare(
        impl_->parents.view(),
        impl_->generation_context());
    impl_->total_ms = 0.0;
    impl_->save_snapshot(0, impl_->parents.view());
    after_reset();
}

bool EvolutionaryAlgorithm::initialized() const noexcept {
    return impl_->initialized;
}

bool EvolutionaryAlgorithm::finished() const noexcept {
    return impl_->initialized &&
        impl_->generation >= impl_->config.max_generations;
}

int EvolutionaryAlgorithm::generation() const noexcept {
    return impl_->generation;
}

const Population& EvolutionaryAlgorithm::population() const noexcept {
    return impl_->parents;
}

Population& EvolutionaryAlgorithm::population() noexcept {
    return impl_->parents;
}

Algorithm::Algorithm(std::unique_ptr<IAlgorithm> implementation)
    : implementation_(std::move(implementation))
{
    if (!implementation_) {
        throw std::invalid_argument(
            "Algorithm implementation cannot be null");
    }
}

Algorithm::~Algorithm() = default;
Algorithm::Algorithm(Algorithm&&) noexcept = default;
Algorithm& Algorithm::operator=(Algorithm&&) noexcept = default;

void Algorithm::initialize() { implementation_->initialize(); }
void Algorithm::step() { implementation_->step(); }
RunResult Algorithm::run() { return implementation_->run(); }
DeviceRunResult Algorithm::run_device() { return implementation_->run_device(); }
RunResult Algorithm::result() { return implementation_->result(); }
DeviceRunResult Algorithm::device_result() {
    return implementation_->device_result();
}
void Algorithm::synchronize() { implementation_->synchronize(); }
void Algorithm::reset() { implementation_->reset(); }

bool Algorithm::initialized() const noexcept {
    return implementation_->initialized();
}

bool Algorithm::finished() const noexcept {
    return implementation_->finished();
}

int Algorithm::generation() const noexcept {
    return implementation_->generation();
}

const Population& Algorithm::population() const noexcept {
    return implementation_->population();
}

Population& Algorithm::population() noexcept {
    return implementation_->population();
}

} // namespace cuda_moea
