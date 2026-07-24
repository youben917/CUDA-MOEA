#include "cuda_moea/algorithms/nsga3.cuh"

#include <stdexcept>
#include <vector>

#include "cuda_moea/selection/nsga3/kernels/association.cuh"
#include "cuda_moea/selection/nsga3/kernels/cv_quantization.cuh"
#include "cuda_moea/selection/nsga3/kernels/ideal_point.cuh"
#include "cuda_moea/selection/nsga3/kernels/dynamic_workspace.cuh"
#include "cuda_moea/selection/nsga3/kernels/nsga3_population_update.cuh"
#include "cuda_moea/selection/nsga3/kernels/non_dominated_sort.cuh"
#include "cuda_moea/selection/nsga3/kernels/niching.cuh"
#include "cuda_moea/selection/nsga3/kernels/normalization.cuh"
#include "cuda_moea/problem/dtlz/population_adapter.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"
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

bool quantization_enabled(const NSGA3SelectionConfig& config)
{
    const auto& value = config.cv_quantization;
    return value.bins > 0 &&
        nsga3::validate_cv_quant_params(
            value.bins,
            value.clip_upper,
            value.log_alpha,
            value.feasibility_epsilon);
}

} // namespace

struct NSGA3EnvironmentSelector::Impl {
    explicit Impl(NSGA3SelectionConfig value) : config(value) {}

    NSGA3SelectionConfig config;
    AlgorithmInfo info;
    CudaContext* cuda = nullptr;
    IReferenceDirectionProvider* references = nullptr;
    bool initialized = false;
    bool prepared_once = false;
    bool switch_to_csr = false;
    long long pairs_capacity = 0;

    DeviceBuffer<int> fronts;
    DeviceBuffer<long long> domination_count;
    DeviceBuffer<int> assigned_count;
    DeviceBuffer<int> new_front;
    DeviceBuffer<float> parent_cv_quantized;

    DeviceBuffer<float> mixed_variables;
    DeviceBuffer<float> mixed_cv;
    DeviceBuffer<float> mixed_cv_quantized;
    DeviceBuffer<float> mixed_objectives;
    DeviceBuffer<int> domination_pairs;
    DeviceBuffer<std::uint32_t> dominatee_mask;
    DeviceBuffer<int> dominatee;
    DeviceBuffer<int> mixed_fronts;
    DeviceBuffer<int> prior_mask;
    DeviceBuffer<int> last_mask;

    DeviceBuffer<float> ideal_last;
    DeviceBuffer<float> ideal_offspring;
    DeviceBuffer<float> ideal_current;

    DeviceBuffer<int> extreme_indices;
    DeviceBuffer<float> extreme_points;
    DeviceBuffer<float> reciprocal_intercepts;
    DeviceBuffer<int> svd_info;
    DeviceBuffer<float> matrix_u;
    DeviceBuffer<float> sigma;
    DeviceBuffer<float> matrix_vh;
    DeviceBuffer<int> ill_conditioned;
    DeviceBuffer<float> sigma_cross;
    DeviceBuffer<float> ones;
    DeviceBuffer<float> temp_u;
    DeviceBuffer<float> temp_s;

    DeviceBuffer<int> rho_prior;
    DeviceBuffer<int> selected_indices;

    void allocate()
    {
        auto& ctx = *cuda;
        const int n = info.population_size;
        const int mixed_n = 2 * n;
        const int m = info.objective_count;
        const int d = info.dimension;
        const auto refs = references->view();
        if (!refs.values || refs.count <= 0) {
            throw std::invalid_argument(
                "NSGA-III requires non-empty reference directions");
        }

        const auto pool = ctx.execution_pool();
        const auto stream = ctx.execution_stream();

        fronts.allocate(n, pool, stream);
        domination_count.allocate(1, pool, stream);
        assigned_count.allocate(1, pool, stream);
        new_front.allocate(1, pool, stream);
        if (quantization_enabled(config)) {
            parent_cv_quantized.allocate(n, pool, stream);
            mixed_cv_quantized.allocate(mixed_n, pool, stream);
        }

        mixed_variables.allocate(
            static_cast<std::size_t>(mixed_n) * d, pool, stream);
        mixed_cv.allocate(mixed_n, pool, stream);
        mixed_objectives.allocate(
            static_cast<std::size_t>(m) * mixed_n, pool, stream);

        pairs_capacity = compute_max_dompairs(
            mixed_n, config.sparse_ratio);
        domination_pairs.allocate(
            static_cast<std::size_t>(2) * pairs_capacity,
            pool,
            stream);
        const int tiles = (mixed_n + WS - 1) / WS;
        dominatee_mask.allocate(
            static_cast<std::size_t>(mixed_n) * tiles,
            pool,
            stream);
        dominatee.allocate(mixed_n, pool, stream);
        mixed_fronts.allocate(mixed_n, pool, stream);
        prior_mask.allocate(mixed_n, pool, stream);
        last_mask.allocate(mixed_n, pool, stream);

        ideal_last.allocate(m, pool, stream);
        ideal_offspring.allocate(m, pool, stream);
        ideal_current.allocate(m, pool, stream);

        extreme_indices.allocate(m, pool, stream);
        extreme_points.allocate(static_cast<std::size_t>(m) * m, pool, stream);
        reciprocal_intercepts.allocate(m, pool, stream);
        svd_info.allocate(1, pool, stream);
        matrix_u.allocate(static_cast<std::size_t>(m) * m, pool, stream);
        sigma.allocate(m, pool, stream);
        matrix_vh.allocate(static_cast<std::size_t>(m) * m, pool, stream);
        ill_conditioned.allocate(1, pool, stream);
        sigma_cross.allocate(m, pool, stream);
        ones.allocate(m, pool, stream);
        temp_u.allocate(m, pool, stream);
        temp_s.allocate(m, pool, stream);

        std::vector<float> host_ones(m, 1.0f);
        CUDA_CHECK(cudaMemcpyAsync(
            ones.data(),
            host_ones.data(),
            host_ones.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));

        rho_prior.allocate(refs.count, pool, stream);
        selected_indices.allocate(n, pool, stream);
    }

    float* cv_for_sort(
        const float* source,
        int count,
        DeviceBuffer<float>& destination,
        cudaStream_t stream)
    {
        if (!quantization_enabled(config)) {
            return const_cast<float*>(source);
        }
        const auto& quant = config.cv_quantization;
        nsga3::quantize_cv(
            source,
            destination.data(),
            count,
            quant.feasibility_epsilon,
            quant.clip_upper,
            quant.log_alpha,
            quant.bins,
            stream);
        return destination.data();
    }
};

NSGA3EnvironmentSelector::NSGA3EnvironmentSelector(
    NSGA3SelectionConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

NSGA3EnvironmentSelector::~NSGA3EnvironmentSelector() = default;

void NSGA3EnvironmentSelector::initialize(
    const AlgorithmInfo& info,
    IReferenceDirectionProvider* references,
    CudaContext& cuda)
{
    if (!references) {
        throw std::invalid_argument(
            "NSGA3EnvironmentSelector requires reference directions");
    }
    if (!(impl_->config.sparse_ratio > 0.0f)) {
        throw std::invalid_argument("NSGA-III sparse_ratio must be positive");
    }
    if (impl_->config.cv_quantization.bins > 0 &&
        !quantization_enabled(impl_->config)) {
        throw std::invalid_argument(
            "Invalid NSGA-III CV quantization configuration");
    }
    if (info.population_size % 2 != 0) {
        throw std::invalid_argument(
            "The current NSGA-III CUDA backend requires an even population size");
    }
    if (info.objective_count > MAX_M) {
        throw std::invalid_argument(
            "The current NSGA-III CUDA backend supports at most MAX_M objectives");
    }
    impl_->info = info;
    impl_->references = references;
    impl_->cuda = &cuda;
    impl_->allocate();
    impl_->initialized = true;
}

void NSGA3EnvironmentSelector::prepare(
    ConstPopulationView population,
    const GenerationContext&)
{
    if (!impl_->initialized) {
        throw std::logic_error("NSGA3 selector is not initialized");
    }

    auto& ctx = *impl_->cuda;
    auto& native = detail::ContextAccess::get(ctx);
    const int n = impl_->info.population_size;
    const int m = impl_->info.objective_count;
    const auto pool = ctx.execution_pool();
    const auto stream = ctx.execution_stream();

    const long long capacity = compute_max_dompairs(
        n, impl_->config.sparse_ratio);
    DeviceBuffer<int> pairs(
        static_cast<std::size_t>(2) * capacity, pool, stream);
    DeviceBuffer<int> dominatee(n, pool, stream);

    float* cv = impl_->cv_for_sort(
        population.constraints,
        n,
        impl_->parent_cv_quantized,
        stream);

    run_ndsort_init(
        native.streams,
        cv,
        const_cast<float*>(population.objectives),
        impl_->fronts.data(),
        pairs.data(),
        dominatee.data(),
        impl_->domination_count.data(),
        impl_->assigned_count.data(),
        impl_->new_front.data(),
        capacity,
        m,
        n,
        pool);

    if (!impl_->prepared_once) {
        get_idpts(
            population.objectives,
            impl_->ideal_last.data(),
            m,
            n,
            stream);
        impl_->prepared_once = true;
    }
}

void NSGA3EnvironmentSelector::select(
    ConstPopulationView parents,
    ConstPopulationView offspring,
    PopulationView next,
    const GenerationContext& context)
{
    auto& ctx = *impl_->cuda;
    auto& native = detail::ContextAccess::get(ctx);
    const auto stream = ctx.execution_stream();
    const auto pool = ctx.execution_pool();
    const int n = impl_->info.population_size;
    const int mixed_n = 2 * n;
    const int m = impl_->info.objective_count;
    const int d = impl_->info.dimension;
    const auto refs = impl_->references->view();

    PopData parent_pop = make_pop(parents);
    PopData offspring_pop = make_pop(offspring);
    PopData mixed_pop(mixed_n, d);
    mixed_pop.d_pop = impl_->mixed_variables.data();
    mixed_pop.d_bound = const_cast<float*>(parents.bounds);
    PopData next_pop = make_pop(next);

    ctx.wait_evaluation_on_execution();
    nsga3::merge_pop(parent_pop, offspring_pop, mixed_pop, d, n, stream);
    nsga3::merge_cv(
        parents.constraints,
        offspring.constraints,
        impl_->mixed_cv.data(),
        n,
        stream);
    nsga3::merge_fv(
        parents.objectives,
        offspring.objectives,
        impl_->mixed_objectives.data(),
        m,
        n,
        stream);

    float* cv = impl_->cv_for_sort(
        impl_->mixed_cv.data(),
        mixed_n,
        impl_->mixed_cv_quantized,
        stream);

    const int max_front = run_ndsort(
        cv,
        impl_->mixed_objectives.data(),
        impl_->mixed_fronts.data(),
        impl_->domination_pairs.data(),
        impl_->dominatee_mask.data(),
        impl_->dominatee.data(),
        impl_->domination_count.data(),
        impl_->assigned_count.data(),
        impl_->new_front.data(),
        impl_->pairs_capacity,
        impl_->switch_to_csr,
        context.generation,
        context.max_generations,
        m,
        n,
        stream,
        pool);

    const NumNDSData sizes = get_nds_mask_size(
        impl_->mixed_fronts.data(),
        impl_->prior_mask.data(),
        impl_->last_mask.data(),
        max_front,
        n,
        stream);

    int* prior_indices = nullptr;
    int* last_indices = nullptr;
    int* nds_indices = nullptr;
    float* nds_objectives = nullptr;
    float* extreme_block_min = nullptr;
    int* extreme_block_argmin = nullptr;
    float* association_block_min = nullptr;
    int* association_block_argmin = nullptr;
    float* association_distance = nullptr;
    int* association_index = nullptr;
    ull* sorted_keys_ping = nullptr;
    ull* sorted_keys_pong = nullptr;
    int* sorted_local_ping = nullptr;
    int* sorted_local_pong = nullptr;
    int* segment_head = nullptr;
    int* sorted_reference = nullptr;
    int* niche_rank = nullptr;
    uint* niching_random = nullptr;

    malloc_dynmem(
        native.streams,
        native.pools,
        prior_indices,
        last_indices,
        nds_indices,
        nds_objectives,
        extreme_block_min,
        extreme_block_argmin,
        association_block_min,
        association_block_argmin,
        association_distance,
        association_index,
        sorted_keys_ping,
        sorted_keys_pong,
        sorted_local_ping,
        sorted_local_pong,
        segment_head,
        sorted_reference,
        niche_rank,
        niching_random,
        sizes,
        refs.count,
        m);

    extract_ndsfit_mskidx(
        impl_->mixed_objectives.data(),
        impl_->prior_mask.data(),
        impl_->last_mask.data(),
        prior_indices,
        last_indices,
        nds_indices,
        nds_objectives,
        sizes,
        m,
        n,
        stream);

    get_idpts(
        offspring.objectives,
        impl_->ideal_offspring.data(),
        m,
        n,
        stream);
    update_idpts(
        impl_->ideal_last.data(),
        impl_->ideal_offspring.data(),
        impl_->ideal_current.data(),
        m,
        stream);

    int* chosen = prior_indices;
    if (sizes.N_rem > 0) {
        execute_normalization(
            nds_objectives,
            impl_->ideal_current.data(),
            extreme_block_min,
            extreme_block_argmin,
            impl_->extreme_indices.data(),
            impl_->extreme_points.data(),
            impl_->reciprocal_intercepts.data(),
            impl_->svd_info.data(),
            impl_->matrix_u.data(),
            impl_->sigma.data(),
            impl_->matrix_vh.data(),
            impl_->ill_conditioned.data(),
            impl_->sigma_cross.data(),
            impl_->ones.data(),
            impl_->temp_u.data(),
            impl_->temp_s.data(),
            m,
            sizes.N_nds,
            pool,
            stream);

        execute_association(
            nds_objectives,
            refs.values,
            association_block_min,
            association_block_argmin,
            association_distance,
            association_index,
            refs.count,
            m,
            sizes.N_nds,
            stream);

        execute_niching(
            ctx.random_seed(),
            ctx.random_offset(),
            nds_indices,
            association_distance,
            association_index,
            impl_->rho_prior.data(),
            sorted_keys_ping,
            sorted_keys_pong,
            sorted_local_ping,
            sorted_local_pong,
            segment_head,
            sorted_reference,
            niche_rank,
            niching_random,
            impl_->selected_indices.data(),
            sizes,
            refs.count,
            n,
            pool,
            stream,
            false);
        chosen = impl_->selected_indices.data();
    }

    nsga3::update_iter_vars(
        mixed_pop,
        impl_->mixed_cv.data(),
        impl_->mixed_objectives.data(),
        impl_->mixed_fronts.data(),
        impl_->ideal_current.data(),
        chosen,
        next_pop,
        next.constraints,
        next.objectives,
        impl_->fronts.data(),
        impl_->ideal_last.data(),
        d,
        m,
        n,
        stream);

    free_dynmem(
        stream,
        prior_indices,
        last_indices,
        nds_indices,
        nds_objectives,
        extreme_block_min,
        extreme_block_argmin,
        association_block_min,
        association_block_argmin,
        association_distance,
        association_index,
        sorted_keys_ping,
        sorted_keys_pong,
        sorted_local_ping,
        sorted_local_pong,
        segment_head,
        sorted_reference,
        niche_rank,
        niching_random);

    parent_pop.d_pop = nullptr;
    parent_pop.d_bound = nullptr;
    offspring_pop.d_pop = nullptr;
    offspring_pop.d_bound = nullptr;
    mixed_pop.d_pop = nullptr;
    mixed_pop.d_bound = nullptr;
    next_pop.d_pop = nullptr;
    next_pop.d_bound = nullptr;
}

MatingStateView NSGA3EnvironmentSelector::mating_state() const noexcept
{
    return {
        {impl_->fronts.data(), impl_->fronts.size()},
        {},
        {},
        impl_->info.population_size
    };
}

int NSGA3EnvironmentSelector::active_count() const noexcept {
    return impl_->info.population_size;
}

DeviceSpan<const int>
NSGA3EnvironmentSelector::result_indices() const noexcept
{
    return {impl_->fronts.data(), impl_->fronts.size()};
}

void NSGA3EnvironmentSelector::reset()
{
    impl_->prepared_once = false;
    impl_->switch_to_csr = false;
}

} // namespace cuda_moea
