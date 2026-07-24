#pragma once

#include "cuda_moea/builder/basic_algorithm_builder.cuh"

namespace cuda_moea {

class AlgorithmBuilder final
    : public BasicAlgorithmBuilder<AlgorithmBuilder> {
public:
    explicit AlgorithmBuilder(AlgorithmKind kind);
    Algorithm build();
};

} // namespace cuda_moea
