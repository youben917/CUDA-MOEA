#pragma once

#include <memory>

#include "cuda_moea/reproduction/crossover_operator.cuh"
#include "cuda_moea/reproduction/mating_selector.cuh"
#include "cuda_moea/reproduction/mutation_operator.cuh"

namespace cuda_moea {

class ReproductionPipeline {
public:
    ReproductionPipeline(
        std::unique_ptr<IMatingSelector> mating,
        std::unique_ptr<ICrossoverOperator> crossover,
        std::unique_ptr<IMutationOperator> mutation);

    void initialize(const AlgorithmInfo& info, CudaContext& cuda);
    void generate(
        ConstPopulationView parents,
        MatingStateView mating_state,
        PopulationView offspring,
        const GenerationContext& context);

    void set_mating(std::unique_ptr<IMatingSelector> mating);
    void set_crossover(std::unique_ptr<ICrossoverOperator> crossover);
    void set_mutation(std::unique_ptr<IMutationOperator> mutation);

private:
    std::unique_ptr<IMatingSelector> mating_;
    std::unique_ptr<ICrossoverOperator> crossover_;
    std::unique_ptr<IMutationOperator> mutation_;
    DeviceBuffer<int> parent_indices_;
    AlgorithmInfo info_;
    CudaContext* cuda_ = nullptr;
};

} // namespace cuda_moea
