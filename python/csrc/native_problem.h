#pragma once
#include "adapters.h"

namespace cuda_moea::python {
std::unique_ptr<IProblemEvaluator> make_native_problem(const py::dict& spec);
py::dict native_problem_schema(const std::string& path);
}
