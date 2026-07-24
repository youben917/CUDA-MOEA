#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuda_moea {

template <typename T>
struct DeviceSpan {
    T* data = nullptr;
    std::size_t size = 0;

    bool empty() const noexcept { return data == nullptr || size == 0; }
    explicit operator bool() const noexcept { return !empty(); }
};

struct ConstPopulationView {
    const float* variables = nullptr;    // (N, D), row-major
    const float* objectives = nullptr;   // (M, N), objective-major
    const float* constraints = nullptr;  // (N,)
    const float* bounds = nullptr;       // interleaved [lb0, ub0, ...]
    int size = 0;
    int dimension = 0;
    int objective_count = 0;
};

struct PopulationView {
    float* variables = nullptr;
    float* objectives = nullptr;
    float* constraints = nullptr;
    const float* bounds = nullptr;
    int size = 0;
    int dimension = 0;
    int objective_count = 0;

    operator ConstPopulationView() const noexcept {
        return {
            variables,
            objectives,
            constraints,
            bounds,
            size,
            dimension,
            objective_count
        };
    }
};

struct ProblemInfo {
    std::string name;
    int dimension = 0;
    int objective_count = 0;
    int constraint_count = 0;
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;
};

struct AlgorithmInfo {
    std::string name;
    int population_size = 0;
    int dimension = 0;
    int objective_count = 0;
    int max_generations = 0;
};

struct MatingStateView {
    DeviceSpan<const int> rank;
    DeviceSpan<const int> reference_index;
    DeviceSpan<const float> score;
    int active_count = 0;
};

struct ReferenceDirectionView {
    float* values = nullptr;               // (M, K), objective-major
    const float* initial_values = nullptr; // optional immutable V0
    const float* gamma = nullptr;          // optional nearest-neighbor angle per direction
    int objective_count = 0;
    int count = 0;
};

struct RunResult {
    std::vector<float> population;
    std::vector<float> objectives;
    std::vector<float> constraints;
    std::vector<int> auxiliary_indices;
    int active_count = 0;
    double total_ms = 0.0;
};

// Non-owning device-side result view. The owning Algorithm must outlive it.
struct DeviceRunResult {
    ConstPopulationView population;
    DeviceSpan<const int> auxiliary_indices;
    int active_count = 0;
    double total_ms = 0.0;
    int device_id = 0;
};

} // namespace cuda_moea
