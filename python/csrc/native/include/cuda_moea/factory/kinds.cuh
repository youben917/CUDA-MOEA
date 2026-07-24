#pragma once

namespace cuda_moea {

enum class AlgorithmKind {
    NSGA3,
    RVEA
};

enum class ProblemKind {
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    ConvexDTLZ2,
    C1DTLZ1,
    C1DTLZ3,
    C2DTLZ2,
    C2ConvexDTLZ2,
    C3DTLZ1,
    C3DTLZ4,
    CSDP
};

enum class MatingKind {
    Random,
    Tournament
};

enum class CrossoverKind {
    SBX
};

enum class MutationKind {
    Polynomial,
    None
};

enum class ReferenceDirectionKind {
    DasDennis,
    AdaptiveRVEA
};

enum class EnvironmentSelectorKind {
    NSGA3,
    RVEA
};

} // namespace cuda_moea
