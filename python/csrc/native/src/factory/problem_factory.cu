#include "cuda_moea/factory/problem_factory.cuh"

#include <stdexcept>

namespace cuda_moea {
namespace {

DTLZProblemType to_dtlz_type(ProblemKind kind)
{
    switch (kind) {
    case ProblemKind::DTLZ1: return DTLZProblemType::DTLZ1;
    case ProblemKind::DTLZ2: return DTLZProblemType::DTLZ2;
    case ProblemKind::DTLZ3: return DTLZProblemType::DTLZ3;
    case ProblemKind::DTLZ4: return DTLZProblemType::DTLZ4;
    case ProblemKind::DTLZ5: return DTLZProblemType::DTLZ5;
    case ProblemKind::DTLZ6: return DTLZProblemType::DTLZ6;
    case ProblemKind::DTLZ7: return DTLZProblemType::DTLZ7;
    case ProblemKind::ConvexDTLZ2:
        return DTLZProblemType::ConvexDTLZ2;
    case ProblemKind::C1DTLZ1: return DTLZProblemType::C1DTLZ1;
    case ProblemKind::C1DTLZ3: return DTLZProblemType::C1DTLZ3;
    case ProblemKind::C2DTLZ2: return DTLZProblemType::C2DTLZ2;
    case ProblemKind::C2ConvexDTLZ2:
        return DTLZProblemType::C2ConvexDTLZ2;
    case ProblemKind::C3DTLZ1: return DTLZProblemType::C3DTLZ1;
    case ProblemKind::C3DTLZ4: return DTLZProblemType::C3DTLZ4;
    case ProblemKind::CSDP: return DTLZProblemType::CSDP;
    }
    throw std::invalid_argument("Unsupported problem kind");
}

} // namespace

std::unique_ptr<IProblemEvaluator> ProblemFactory::create(
    ProblemKind kind,
    int dimension,
    int objective_count,
    float constraint_activation_ratio)
{
    return create(DTLZProblemConfig{
        to_dtlz_type(kind),
        dimension,
        objective_count,
        constraint_activation_ratio
    });
}

std::unique_ptr<IProblemEvaluator> ProblemFactory::create(
    DTLZProblemConfig config)
{
    return std::make_unique<DTLZProblem>(config);
}

} // namespace cuda_moea
