#pragma once

#include <tuple>
#include <vector>

std::tuple<std::vector<float>, std::vector<float>, int> initialize_refpts(
    int N,   // input: scalar - approximate number of desired points
    int M    // input: scalar - number of objectives
);

std::tuple<std::vector<float>, std::vector<float>, int>
initialize_refpts_by_partitions(
    int partitions,
    int objective_count);
