#include "cuda_moea/problem/problem_evaluator.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cuda_moea/problem/dtlz/evaluation_workspace.cuh"
#include "cuda_moea/problem/dtlz/dtlz_kernels.cuh"
#include "cuda_moea/problem/dtlz/population_adapter.cuh"
#include "cuda_moea/core/context_access.cuh"

namespace cuda_moea {
namespace {

bool is_constrained(DTLZProblemType type)
{
    return static_cast<int>(type) >= static_cast<int>(DTLZProblemType::C1DTLZ1);
}

const char* problem_name(DTLZProblemType type)
{
    switch (type) {
    case DTLZProblemType::DTLZ1: return "DTLZ1";
    case DTLZProblemType::DTLZ2: return "DTLZ2";
    case DTLZProblemType::DTLZ3: return "DTLZ3";
    case DTLZProblemType::DTLZ4: return "DTLZ4";
    case DTLZProblemType::DTLZ5: return "DTLZ5";
    case DTLZProblemType::DTLZ6: return "DTLZ6";
    case DTLZProblemType::DTLZ7: return "DTLZ7";
    case DTLZProblemType::ConvexDTLZ2: return "ConvexDTLZ2";
    case DTLZProblemType::C1DTLZ1: return "C1DTLZ1";
    case DTLZProblemType::C1DTLZ3: return "C1DTLZ3";
    case DTLZProblemType::C2DTLZ2: return "C2DTLZ2";
    case DTLZProblemType::C2ConvexDTLZ2: return "C2ConvexDTLZ2";
    case DTLZProblemType::C3DTLZ1: return "C3DTLZ1";
    case DTLZProblemType::C3DTLZ4: return "C3DTLZ4";
    case DTLZProblemType::CSDP: return "CSDP";
    }
    return "Unknown";
}

PopData make_pop_view(PopulationView population)
{
    PopData view(population.size, population.dimension);
    view.d_pop = population.variables;
    view.d_bound = const_cast<float*>(population.bounds);
    return view;
}

} // namespace

struct DTLZProblem::Impl {
    DTLZProblemConfig config;
    ProblemInfo info;
    MOPAuxData aux;
    std::unique_ptr<MOEAStdTestEvaluator> evaluator;
    CudaContext* cuda = nullptr;
    bool initialized = false;

    explicit Impl(DTLZProblemConfig value) : config(value)
    {
        if (config.dimension <= 0 || config.objective_count <= 0) {
            throw std::invalid_argument("DTLZ dimensions must be positive");
        }
        if (!std::isfinite(config.constraint_activation_ratio)) {
            throw std::invalid_argument("constraint_activation_ratio must be finite");
        }
        config.constraint_activation_ratio = std::clamp(
            config.constraint_activation_ratio, 0.0f, 1.0f);

        info.name = problem_name(config.type);
        info.dimension = config.dimension;
        info.objective_count = config.objective_count;
        info.constraint_count = is_constrained(config.type) ? 1 : 0;
        info.lower_bounds.assign(config.dimension, 0.0f);
        info.upper_bounds.assign(config.dimension, 1.0f);
    }
};

DTLZProblem::DTLZProblem(DTLZProblemConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

DTLZProblem::~DTLZProblem()
{
    if (impl_ && impl_->initialized && impl_->cuda) {
        impl_->aux.free(impl_->cuda->evaluation_stream());
    }
}

const ProblemInfo& DTLZProblem::info() const noexcept {
    return impl_->info;
}

void DTLZProblem::initialize(CudaContext& cuda)
{
    if (impl_->initialized) return;

    impl_->cuda = &cuda;
    impl_->aux.mop_type = static_cast<MOPType>(impl_->config.type);
    impl_->aux.N = 0;
    impl_->aux.D = impl_->config.dimension;
    impl_->aux.M = impl_->config.objective_count;
    impl_->initialized = true;
}

void DTLZProblem::evaluate(
    PopulationView population,
    const EvaluationContext& context,
    cudaStream_t)
{
    if (!impl_->initialized) {
        throw std::logic_error("DTLZProblem must be initialized before evaluate()");
    }

    if (!impl_->evaluator || impl_->aux.N != population.size) {
        if (impl_->aux.d_fv_trans) {
            impl_->aux.free(context.cuda.evaluation_stream());
        }
        impl_->aux.N = population.size;
        impl_->aux.mallocfrompool(
            population.objective_count,
            population.size,
            context.cuda.evaluation_pool(),
            context.cuda.evaluation_stream());
        impl_->evaluator = std::make_unique<MOEAStdTestEvaluator>(
            impl_->aux,
            population.objective_count,
            population.size,
            false,
            MOEAStdTestEvaluator::Config{
                impl_->config.constraint_activation_ratio});
    }

    auto& native = detail::ContextAccess::get(context.cuda);
    impl_->evaluator->set_iteration_context(
        context.generation,
        context.max_generations);

    PopData pop = make_pop_view(population);
    impl_->evaluator->evaluate(
        native.streams,
        pop,
        population.constraints,
        population.objectives);
    pop.d_pop = nullptr;
    pop.d_bound = nullptr;
}

bool DTLZProblem::prepare_parent(
    PopulationView population,
    const EvaluationContext& context,
    cudaStream_t)
{
    if (!impl_->evaluator) return false;

    auto& native = detail::ContextAccess::get(context.cuda);
    PopData pop = make_pop_view(population);
    const bool changed = impl_->evaluator->prepare_parent_cv(
        native.streams,
        pop,
        population.objectives,
        population.constraints,
        context.generation,
        population.size);
    pop.d_pop = nullptr;
    pop.d_bound = nullptr;
    return changed;
}

void DTLZProblem::reset()
{
    impl_->evaluator.reset();
}

} // namespace cuda_moea
