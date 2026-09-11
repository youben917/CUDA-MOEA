#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "cuda_moea/native/build_id.h"
#include "cuda_moea/problem/problem_evaluator.cuh"

namespace cuda_moea::native {

using Parameter = std::variant<std::int64_t, double, bool, std::string, std::vector<double>>;
struct ParameterSpec {
    const char* name;
    Parameter default_value;
    const char* description;
};

struct ProblemConfig {
    int dimension;
    int objectives;
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;
    std::unordered_map<std::string, Parameter> parameters;

    template<class T> const T& get(const std::string& name) const {
        return std::get<T>(parameters.at(name));
    }
};

// The primitive prefix is checked before touching C++ data or invoking factories.
// This is an exact-build extension interface, not a cross-version stable ABI.
struct Plugin {
    std::uint32_t abi;
    const char* build_id;
    const char* name;
    const std::vector<ParameterSpec>* parameters;
    IProblemEvaluator* (*create)(const ProblemConfig&);
    void (*destroy)(IProblemEvaluator*) noexcept;
};
using Entry = const Plugin* (*)();

} // namespace cuda_moea::native

// Place in exactly one translation unit. Type takes ProblemConfig in its ctor.
#define CUDA_MOEA_REGISTER_PROBLEM(Type, Name, Schema)                         \
    extern "C" __attribute__((visibility("default")))                        \
    const ::cuda_moea::native::Plugin* cuda_moea_problem_v1() {                 \
        static const ::cuda_moea::native::Plugin plugin{                      \
            1, CUDA_MOEA_NATIVE_BUILD_ID, Name, &(Schema),                     \
            [](const ::cuda_moea::native::ProblemConfig& config)              \
                -> ::cuda_moea::IProblemEvaluator* { return new Type(config); }, \
            [](::cuda_moea::IProblemEvaluator* value) noexcept { delete value; } \
        };                                                                  \
        return &plugin;                                                     \
    }
