#include "cuda_moea/builder/algorithm_builder.cuh"

namespace cuda_moea {

AlgorithmBuilder::AlgorithmBuilder(AlgorithmKind kind)
    : BasicAlgorithmBuilder<AlgorithmBuilder>(kind) {}

Algorithm AlgorithmBuilder::build() {
    return buildConfigured();
}

} // namespace cuda_moea
