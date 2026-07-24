#pragma once

#include <filesystem>

#include "cuda_moea/algorithm.cuh"

namespace cuda_moea::detail {

class DataRecorder {
public:
    explicit DataRecorder(DataSaveConfig config);

    bool enabled() const noexcept;
    bool should_save(int generation, int max_generations) const noexcept;

    void initialize(
        const AlgorithmInfo& algorithm,
        const ProblemInfo& problem,
        const CudaConfig& cuda_config,
        ReferenceDirectionView references,
        CudaContext& cuda);

    void save_snapshot(
        int generation,
        ConstPopulationView population,
        ReferenceDirectionView references,
        DeviceSpan<const int> auxiliary_indices,
        int active_count,
        CudaContext& cuda);

private:
    DataSaveConfig config_;
    std::filesystem::path root_;
    bool initialized_ = false;
};

} // namespace cuda_moea::detail
