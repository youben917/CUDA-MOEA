#include "cuda_moea/algorithms/rvea.cuh"

#include <stdexcept>

#include "cuda_moea/selection/rvea/kernels/angle_penalized_distance.cuh"
#include "cuda_moea/selection/rvea/kernels/elitism_selection.cuh"
#include "cuda_moea/selection/rvea/kernels/objective_translation.cuh"
#include "cuda_moea/selection/rvea/kernels/rvea_population_update.cuh"
#include "cuda_moea/selection/rvea/kernels/front_zero.cuh"
#include "cuda_moea/selection/rvea/kernels/reference_partition.cuh"
#include "cuda_moea/problem/dtlz/population_adapter.cuh"
#include "cuda_moea/core/context_access.cuh"

namespace cuda_moea {
namespace {

PopData make_pop(ConstPopulationView view)
{
    PopData pop(view.size, view.dimension);
    pop.d_pop = const_cast<float*>(view.variables);
    pop.d_bound = const_cast<float*>(view.bounds);
    return pop;
}

PopData make_pop(PopulationView view)
{
    PopData pop(view.size, view.dimension);
    pop.d_pop = view.variables;
    pop.d_bound = const_cast<float*>(view.bounds);
    return pop;
}

} // namespace

struct RVEAEnvironmentSelector::Impl {
    explicit Impl(RVEASelectionConfig value) : config(value) {}

    RVEASelectionConfig config;
    AlgorithmInfo info;
    CudaContext* cuda = nullptr;
    IReferenceDirectionProvider* references = nullptr;
    int active = 0;
    int frontzero_count = 0;

    DeviceBuffer<float> mixed_variables;
    DeviceBuffer<float> mixed_cv;
    DeviceBuffer<float> mixed_objectives;
    DeviceBuffer<float> translated_objectives;
    DeviceBuffer<float> zmin;
    DeviceBuffer<float> apd;
    DeviceBuffer<float> theta;
    DeviceBuffer<float> norm;
    DeviceBuffer<int> reference_index;
    DeviceBuffer<ull> sort_keys_ping;
    DeviceBuffer<ull> sort_keys_pong;
    DeviceBuffer<int> sort_index_ping;
    DeviceBuffer<int> sort_index_pong;
    DeviceBuffer<int> segment_head;
    DeviceBuffer<int> sorted_reference;
    DeviceBuffer<int> niche_rank;
    DeviceBuffer<int> winner_indices;
    DeviceBuffer<int> winner_count;
    DeviceBuffer<int> selected_indices;
    DeviceBuffer<float> partition_block_max;
    DeviceBuffer<int> partition_block_argmax;

    DeviceBuffer<std::uint32_t> frontzero_bitmask;
    DeviceBuffer<int> frontzero_dominatee;
    DeviceBuffer<int> frontzero_indices;

    void allocate()
    {
        const int n = info.population_size;
        const int mixed_n = 2 * n;
        const int m = info.objective_count;
        const int d = info.dimension;
        const auto refs = references->view();
        if (!refs.values || refs.count <= 0) {
            throw std::invalid_argument(
                "RVEA requires non-empty reference directions");
        }

        const auto pool = cuda->execution_pool();
        const auto stream = cuda->execution_stream();
        mixed_variables.allocate(
            static_cast<std::size_t>(mixed_n) * d, pool, stream);
        mixed_cv.allocate(mixed_n, pool, stream);
        mixed_objectives.allocate(
            static_cast<std::size_t>(m) * mixed_n, pool, stream);
        translated_objectives.allocate(
            static_cast<std::size_t>(m) * mixed_n, pool, stream);
        zmin.allocate(m, pool, stream);
        apd.allocate(mixed_n, pool, stream);
        theta.allocate(mixed_n, pool, stream);
        norm.allocate(mixed_n, pool, stream);
        reference_index.allocate(mixed_n, pool, stream);
        sort_keys_ping.allocate(mixed_n, pool, stream);
        sort_keys_pong.allocate(mixed_n, pool, stream);
        sort_index_ping.allocate(mixed_n, pool, stream);
        sort_index_pong.allocate(mixed_n, pool, stream);
        segment_head.allocate(mixed_n, pool, stream);
        sorted_reference.allocate(mixed_n, pool, stream);
        niche_rank.allocate(mixed_n, pool, stream);
        winner_indices.allocate(mixed_n, pool, stream);
        winner_count.allocate(1, pool, stream);
        selected_indices.allocate(n, pool, stream);

        const int block_rows = (refs.count + 255) / 256;
        partition_block_max.allocate(
            static_cast<std::size_t>(block_rows) * mixed_n,
            pool,
            stream);
        partition_block_argmax.allocate(
            static_cast<std::size_t>(block_rows) * mixed_n,
            pool,
            stream);

        const int tiles = (n + WS - 1) / WS;
        frontzero_bitmask.allocate(
            static_cast<std::size_t>(n) * tiles, pool, stream);
        frontzero_dominatee.allocate(n, pool, stream);
        frontzero_indices.allocate(n, pool, stream);
        active = n;
    }
};

RVEAEnvironmentSelector::RVEAEnvironmentSelector(
    RVEASelectionConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

RVEAEnvironmentSelector::~RVEAEnvironmentSelector() = default;

void RVEAEnvironmentSelector::initialize(
    const AlgorithmInfo& info,
    IReferenceDirectionProvider* references,
    CudaContext& cuda)
{
    if (!references) {
        throw std::invalid_argument(
            "RVEAEnvironmentSelector requires reference directions");
    }
    if (!(impl_->config.alpha > 0.0f)) {
        throw std::invalid_argument("RVEA alpha must be positive");
    }
    if (info.objective_count > MAX_M) {
        throw std::invalid_argument(
            "The current RVEA CUDA backend supports at most MAX_M objectives");
    }
    impl_->info = info;
    impl_->references = references;
    impl_->cuda = &cuda;
    impl_->allocate();
}

void RVEAEnvironmentSelector::prepare(
    ConstPopulationView,
    const GenerationContext&)
{
    // Keep the active count produced by the previous selection. Empty
    // reference-vector slots must not be reintroduced into random mating.
}

void RVEAEnvironmentSelector::select(
    ConstPopulationView parents,
    ConstPopulationView offspring,
    PopulationView next,
    const GenerationContext& context)
{
    auto& ctx = *impl_->cuda;
    auto& native = detail::ContextAccess::get(ctx);
    auto stream = ctx.execution_stream();
    auto& pool = native.pools.exec_pool;
    const int n = impl_->info.population_size;
    const int mixed_n = 2 * n;
    const int m = impl_->info.objective_count;
    const int d = impl_->info.dimension;
    const auto refs = impl_->references->view();
    if (!refs.gamma) {
        throw std::logic_error("RVEA reference directions require gamma values");
    }

    PopData parent_pop = make_pop(parents);
    PopData offspring_pop = make_pop(offspring);
    PopData mixed_pop(mixed_n, d);
    mixed_pop.d_pop = impl_->mixed_variables.data();
    mixed_pop.d_bound = const_cast<float*>(parents.bounds);
    PopData next_pop = make_pop(next);

    ctx.wait_evaluation_on_execution();
    rvea::merge_pop(parent_pop, offspring_pop, mixed_pop, d, n, stream);
    rvea::merge_cv(
        parents.constraints,
        offspring.constraints,
        impl_->mixed_cv.data(),
        n,
        stream);
    rvea::merge_fv(
        parents.objectives,
        offspring.objectives,
        impl_->mixed_objectives.data(),
        m,
        n,
        stream);

    rvea::translate_fv(
        impl_->mixed_objectives.data(),
        impl_->translated_objectives.data(),
        impl_->zmin.data(),
        m,
        mixed_n,
        stream);

    rvea::execute_partition(
        impl_->translated_objectives.data(),
        refs.values,
        impl_->partition_block_max.data(),
        impl_->partition_block_argmax.data(),
        impl_->theta.data(),
        impl_->reference_index.data(),
        refs.count,
        m,
        mixed_n,
        impl_->active,
        n,
        stream);

    rvea::compute_apd(
        impl_->translated_objectives.data(),
        impl_->theta.data(),
        refs.gamma,
        impl_->reference_index.data(),
        impl_->apd.data(),
        impl_->norm.data(),
        mixed_n,
        m,
        refs.count,
        // EvoX increments gen before environmental selection, so its first
        // selection uses 1 / max_gen and its last uses max_gen / max_gen.
        context.generation + 1,
        context.max_generations,
        impl_->config.alpha,
        stream);

    rvea::run_elitism_selection(
        impl_->reference_index.data(),
        impl_->mixed_cv.data(),
        impl_->apd.data(),
        impl_->sort_keys_ping.data(),
        impl_->sort_keys_pong.data(),
        impl_->sort_index_ping.data(),
        impl_->sort_index_pong.data(),
        impl_->segment_head.data(),
        impl_->sorted_reference.data(),
        impl_->niche_rank.data(),
        impl_->winner_indices.data(),
        impl_->winner_count.data(),
        impl_->selected_indices.data(),
        n,
        mixed_n,
        refs.count,
        impl_->active,
        pool,
        stream);

    rvea::update_iter_vars(
        mixed_pop,
        impl_->mixed_cv.data(),
        impl_->mixed_objectives.data(),
        impl_->selected_indices.data(),
        next_pop,
        next.constraints,
        next.objectives,
        d,
        m,
        impl_->active,
        n,
        stream);

    parent_pop.d_pop = nullptr;
    parent_pop.d_bound = nullptr;
    offspring_pop.d_pop = nullptr;
    offspring_pop.d_bound = nullptr;
    mixed_pop.d_pop = nullptr;
    mixed_pop.d_bound = nullptr;
    next_pop.d_pop = nullptr;
    next_pop.d_bound = nullptr;
}

MatingStateView RVEAEnvironmentSelector::mating_state() const noexcept
{
    return {{}, {}, {}, impl_->active};
}

int RVEAEnvironmentSelector::active_count() const noexcept {
    return impl_->active;
}

DeviceSpan<const int>
RVEAEnvironmentSelector::result_indices() const noexcept
{
    return {
        impl_->frontzero_indices.data(),
        static_cast<std::size_t>(impl_->frontzero_count)
    };
}

void RVEAEnvironmentSelector::finalize(
    ConstPopulationView population,
    const GenerationContext&)
{
    auto& ctx = *impl_->cuda;
    rvea::extract_ndsort_frontzero(
        population.constraints,
        population.objectives,
        impl_->frontzero_bitmask.data(),
        impl_->frontzero_dominatee.data(),
        impl_->frontzero_indices.data(),
        impl_->frontzero_count,
        impl_->active,
        population.objective_count,
        population.size,
        ctx.execution_pool(),
        ctx.execution_stream());
}

void RVEAEnvironmentSelector::reset()
{
    impl_->active = impl_->info.population_size;
    impl_->frontzero_count = 0;
}

} // namespace cuda_moea
