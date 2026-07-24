#include "cuda_moea/algorithms/nsga3.cuh"

#include <stdexcept>

namespace cuda_moea {

NSGA3::Builder::Builder()
    : BasicAlgorithmBuilder<NSGA3::Builder>(
          AlgorithmKind::NSGA3) {}

NSGA3::Builder& NSGA3::Builder::selection(
    NSGA3SelectionConfig config)
{
    selection_config_ = config;
    state_.strategies.environment_selector =
        std::make_unique<NSGA3EnvironmentSelector>(
            selection_config_);
    return *this;
}

NSGA3::Builder& NSGA3::Builder::sparseRatio(float value)
{
    selection_config_.sparse_ratio = value;
    state_.strategies.environment_selector =
        std::make_unique<NSGA3EnvironmentSelector>(
            selection_config_);
    return *this;
}

NSGA3::Builder& NSGA3::Builder::referencePointPartition(
    int partitions)
{
    if (partitions <= 0) {
        throw std::invalid_argument(
            "referencePointPartition must be positive");
    }
    state_.strategies.reference_directions =
        std::make_unique<DasDennisDirections>(
            DasDennisConfig{partitions});
    return *this;
}

Algorithm NSGA3::Builder::build() {
    return buildConfigured();
}

} // namespace cuda_moea
