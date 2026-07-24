#include "cuda_moea/reproduction/mating_selector.cuh"

#include <stdexcept>

#include "cuda_moea/core/cuda/cuda_random.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"

namespace cuda_moea {
namespace {

__global__ void tournament_winner_kernel(
    const int2* candidates,
    const int* rank,
    const float* constraints,
    int* winners,
    int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int a = candidates[i].x;
    const int b = candidates[i].y;
    const float cv_a = constraints[a];
    const float cv_b = constraints[b];

    if (cv_a == 0.0f && cv_b != 0.0f) {
        winners[i] = a;
    } else if (cv_a != 0.0f && cv_b == 0.0f) {
        winners[i] = b;
    } else if (cv_a != 0.0f && cv_b != 0.0f) {
        winners[i] = cv_a <= cv_b ? a : b;
    } else {
        winners[i] = rank[a] <= rank[b] ? a : b;
    }
}

} // namespace

void RandomMating::select(
    ConstPopulationView parents,
    MatingStateView state,
    DeviceSpan<int> parent_indices,
    const GenerationContext& context,
    cudaStream_t stream)
{
    const int active = state.active_count > 0
        ? state.active_count
        : parents.size;
    if (active <= 0 || active > parents.size) {
        throw std::invalid_argument("RandomMating received invalid active_count");
    }

    launch_random_int_kernel(
        parent_indices.data,
        static_cast<int>(parent_indices.size),
        0,
        active,
        context.cuda.random_seed(),
        context.cuda.random_offset(),
        256,
        stream);
}

void TournamentMating::initialize(
    const AlgorithmInfo& info,
    CudaContext& cuda)
{
    candidates_.allocate(
        info.population_size,
        cuda.execution_pool(),
        cuda.execution_stream());
}

void TournamentMating::select(
    ConstPopulationView parents,
    MatingStateView state,
    DeviceSpan<int> parent_indices,
    const GenerationContext& context,
    cudaStream_t stream)
{
    if (state.rank.empty() ||
        state.rank.size < static_cast<std::size_t>(parents.size)) {
        throw std::invalid_argument(
            "TournamentMating requires a rank array for every parent");
    }

    const int active = state.active_count > 0
        ? state.active_count
        : parents.size;
    launch_random_int2_kernel(
        candidates_.data(),
        parents.size,
        0,
        active,
        context.cuda.random_seed(),
        context.cuda.random_offset(),
        256,
        stream);

    const int blocks = (parents.size + 255) / 256;
    tournament_winner_kernel<<<blocks, 256, 0, stream>>>(
        candidates_.data(),
        state.rank.data,
        parents.constraints,
        parent_indices.data,
        parents.size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_moea
