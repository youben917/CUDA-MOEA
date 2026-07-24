#pragma once

#include "cuda_moea/algorithm.cuh"
#include "cuda_moea/factory/kinds.cuh"

namespace cuda_moea {
namespace detail {

struct AlgorithmBuildState {
    AlgorithmKind kind;
    AlgorithmConfig config;
    AlgorithmStrategies strategies;
};

} // namespace detail

class AlgorithmFactory {
public:
    static detail::AlgorithmBuildState defaults(AlgorithmKind kind);
    static Algorithm create(detail::AlgorithmBuildState state);
};

} // namespace cuda_moea
