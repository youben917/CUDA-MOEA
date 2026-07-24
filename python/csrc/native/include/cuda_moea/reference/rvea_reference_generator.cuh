#pragma once

#include <tuple>
#include <vector>

namespace rvea {

std::tuple<std::vector<float>, std::vector<float>, int> initialize_refpts(
    int N,   // input: scalar - approximate number of desired points
    int M    // input: scalar - number of objectives
);

// Returns an even population size derived from N_approx.
int compute_N_ref(int N_approx, int M);

} // namespace rvea
