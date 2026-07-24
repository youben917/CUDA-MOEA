#include "cuda_moea/algorithms/rvea.cuh"

namespace cuda_moea {

RVEA::Builder::Builder()
    : BasicAlgorithmBuilder<RVEA::Builder>(
          AlgorithmKind::RVEA) {}

RVEA::Builder& RVEA::Builder::selection(
    RVEASelectionConfig config)
{
    selection_config_ = config;
    state_.strategies.environment_selector =
        std::make_unique<RVEAEnvironmentSelector>(
            selection_config_);
    return *this;
}

RVEA::Builder& RVEA::Builder::selectionAlpha(float value)
{
    selection_config_.alpha = value;
    state_.strategies.environment_selector =
        std::make_unique<RVEAEnvironmentSelector>(
            selection_config_);
    return *this;
}

RVEA::Builder& RVEA::Builder::referenceAdaptation(
    AdaptiveDirectionConfig config)
{
    reference_config_ = config;
    state_.strategies.reference_directions =
        std::make_unique<AdaptiveRVEADirections>(
            reference_config_);
    return *this;
}

RVEA::Builder& RVEA::Builder::referenceAdaptationFrequency(
    float value)
{
    reference_config_.frequency = value;
    state_.strategies.reference_directions =
        std::make_unique<AdaptiveRVEADirections>(
            reference_config_);
    return *this;
}

Algorithm RVEA::Builder::build() {
    return buildConfigured();
}

} // namespace cuda_moea
