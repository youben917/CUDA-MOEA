#include "cuda_moea/reproduction/reproduction_pipeline.cuh"

#include <stdexcept>
#include <utility>

namespace cuda_moea {

ReproductionPipeline::ReproductionPipeline(
    std::unique_ptr<IMatingSelector> mating,
    std::unique_ptr<ICrossoverOperator> crossover,
    std::unique_ptr<IMutationOperator> mutation)
    : mating_(std::move(mating)),
      crossover_(std::move(crossover)),
      mutation_(std::move(mutation))
{
    if (!mating_ || !crossover_ || !mutation_) {
        throw std::invalid_argument(
            "ReproductionPipeline requires mating, crossover, and mutation");
    }
}

void ReproductionPipeline::initialize(
    const AlgorithmInfo& info,
    CudaContext& cuda)
{
    info_ = info;
    cuda_ = &cuda;
    parent_indices_.allocate(
        info.population_size,
        cuda.execution_pool(),
        cuda.execution_stream());
    mating_->initialize(info, cuda);
    crossover_->initialize(info, cuda);
    mutation_->initialize(info, cuda);
}

void ReproductionPipeline::generate(
    ConstPopulationView parents,
    MatingStateView mating_state,
    PopulationView offspring,
    const GenerationContext& context)
{
    if (!cuda_) {
        throw std::logic_error("ReproductionPipeline is not initialized");
    }
    const auto stream = cuda_->execution_stream();
    DeviceSpan<int> indices{parent_indices_.data(), parent_indices_.size()};
    mating_->select(parents, mating_state, indices, context, stream);
    crossover_->apply(
        parents,
        {indices.data, indices.size},
        offspring,
        context,
        stream);
    mutation_->mutate(offspring, context, stream);
}

void ReproductionPipeline::set_mating(
    std::unique_ptr<IMatingSelector> mating)
{
    if (!mating) throw std::invalid_argument("mating cannot be null");
    mating_ = std::move(mating);
    if (cuda_) mating_->initialize(info_, *cuda_);
}

void ReproductionPipeline::set_crossover(
    std::unique_ptr<ICrossoverOperator> crossover)
{
    if (!crossover) throw std::invalid_argument("crossover cannot be null");
    crossover_ = std::move(crossover);
    if (cuda_) crossover_->initialize(info_, *cuda_);
}

void ReproductionPipeline::set_mutation(
    std::unique_ptr<IMutationOperator> mutation)
{
    if (!mutation) throw std::invalid_argument("mutation cannot be null");
    mutation_ = std::move(mutation);
    if (cuda_) mutation_->initialize(info_, *cuda_);
}

} // namespace cuda_moea
