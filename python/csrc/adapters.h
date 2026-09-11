#pragma once

#include <memory>
#include <pybind11/pybind11.h>

#include "cuda_moea/algorithm.cuh"
#include "cuda_moea/problem/problem_evaluator.cuh"
#include "cuda_moea/reference/reference_direction_provider.cuh"
#include "cuda_moea/reproduction/crossover_operator.cuh"
#include "cuda_moea/reproduction/mating_selector.cuh"
#include "cuda_moea/reproduction/mutation_operator.cuh"
#include "cuda_moea/selection/environment_selector.cuh"

namespace cuda_moea::python {
namespace py = pybind11;

std::unique_ptr<IProblemEvaluator> make_problem(const py::dict& spec);
py::dict benchmark_problem(const py::dict& spec, py::object variables, int repeats, int warmup);
std::unique_ptr<IMatingSelector> make_mating(const py::dict& spec);
std::unique_ptr<ICrossoverOperator> make_crossover(const py::dict& spec);
std::unique_ptr<IMutationOperator> make_mutation(const py::dict& spec);
std::unique_ptr<IReferenceDirectionProvider> make_references(const py::dict& spec);
std::unique_ptr<IEnvironmentSelector> make_environment(const py::dict& spec);

} // namespace cuda_moea::python
