#include "problem.cuh"

namespace {
const std::vector<cuda_moea::native::ParameterSpec> parameters{
    {"offset", 2.0, "Center of the second sphere"},
    {"radius", 3.0, "Feasible radius around the origin"},
    {"constrained", false, "Enable sum(x*x) <= radius*radius"},
};
}

CUDA_MOEA_REGISTER_PROBLEM(BiSphere, "BiSphere", parameters)
