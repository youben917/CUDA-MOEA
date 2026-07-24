#include "cuda_moea/problem/dtlz/population_adapter.cuh"
#include "cuda_moea/core/cuda/cuda_random.cuh"

// ======================================================================================================================================================= //
//                                              PopData Implementation                                                                                       //
// ======================================================================================================================================================= //
__host__ PopData::PopData()
    : IPopulationData(), d_pop(nullptr), d_bound(nullptr), D(0)
{
}

__host__ PopData::PopData(int _N, int _D)
    : IPopulationData(_N), d_pop(nullptr), d_bound(nullptr), D(_D)
{
}

__host__ PopData::PopData(PopData&& other) noexcept
    : IPopulationData(std::move(other))
    , d_pop(other.d_pop), d_bound(other.d_bound), D(other.D)
{
    other.d_pop   = nullptr;
    other.d_bound = nullptr;
    other.D       = 0;
}

__host__ PopData& PopData::operator=(PopData&& other) noexcept
{
    if (this != &other) {
        IPopulationData::operator=(std::move(other));
        d_pop   = other.d_pop;
        d_bound = other.d_bound;
        D       = other.D;
        other.d_pop   = nullptr;
        other.d_bound = nullptr;
        other.D       = 0;
    }
    return *this;
}

// =========================================================================
// Memory Management - Atomic Operations
// =========================================================================
__host__ void PopData::mallocfrompool_pop(cudaMemPool_t mempool, cudaStream_t stream)
{
    const size_t pop_bytes = static_cast<size_t>(N) * D * FLOAT_SIZE;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_pop, pop_bytes, mempool, stream));
}

__host__ void PopData::mallocfrompool_inst_attrs(cudaMemPool_t mempool, cudaStream_t stream)
{
    const size_t bound_bytes = static_cast<size_t>(2) * D * FLOAT_SIZE;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_bound, bound_bytes, mempool, stream));
}

__host__ void PopData::free_pop(cudaStream_t stream)
{
    if (d_pop) CUDA_CHECK(cudaFreeAsync(d_pop, stream));
    reset_pop_device_pointers();
}

__host__ void PopData::free_inst_attrs(cudaStream_t stream)
{
    if (d_bound) CUDA_CHECK(cudaFreeAsync(d_bound, stream));
    reset_inst_attrs_device_pointers();
}

__host__ void PopData::reset_device_pointers()
{
    d_pop   = nullptr;
    d_bound = nullptr;
}

__host__ void PopData::reset_pop_device_pointers()
{
    d_pop = nullptr;
}

__host__ void PopData::reset_inst_attrs_device_pointers()
{
    d_bound = nullptr;
}

// ======================================================================================================================================================= //
//                                            PopInitData Implementation                                                                                     //
// ======================================================================================================================================================= //
__host__ PopInitData::PopInitData(int _N, int _D, const std::vector<float>& _h_bound)
    : PopData(_N, _D), h_bound(_h_bound)
{
    assert(static_cast<int>(h_bound.size()) == 2 * D);
}

__host__ PopInitData::PopInitData(PopInitData&& other) noexcept
    : PopData(std::move(other)), h_bound(std::move(other.h_bound))
{
}

__host__ PopInitData& PopInitData::operator=(PopInitData&& other) noexcept
{
    if (this != &other) {
        PopData::operator=(std::move(other));
        h_bound = std::move(other.h_bound);
    }
    return *this;
}

__host__ void PopInitData::mallocfrompool(cudaMemPool_t mempool, cudaStream_t stream)
{
    mallocfrompool_pop(mempool, stream);
    mallocfrompool_inst_attrs(mempool, stream);

    // H2D transfer of bounds
    CUDA_CHECK(cudaMemcpyAsync(
        d_bound, h_bound.data(),
        static_cast<size_t>(2) * D * FLOAT_SIZE,
        cudaMemcpyHostToDevice, stream));
}

__host__ void PopInitData::free(cudaStream_t stream)
{
    IPopulationData::free(stream);
}

// =========================================================================
// Factory Method
// =========================================================================
__host__ PopInitData PopInitData::create_and_initialize(
    ull                        rnd_seed,
    ull&                       glb_rnd_offset,
    int                        N,
    int                        D,
    const std::vector<float>&  h_bound,
    cudaMemPool_t              mempool,
    cudaStream_t               stream,
    bool                       enable_h_save
)
{
    PopInitData pop_init(N, D, h_bound);
    pop_init.mallocfrompool(mempool, stream);
    generate_pop_init(rnd_seed, glb_rnd_offset, pop_init, N, mempool, stream, enable_h_save);
    return pop_init;
}

// ======================================================================================================================================================= //
//                                              Population Initialization                                                                                    //
// ======================================================================================================================================================= //
void generate_pop_init(
    ull                       rnd_seed,
    ull&                      glb_rnd_offset,
    PopInitData&              d_pop_init,
    int                       N,
    cudaMemPool_t             mempool,
    cudaStream_t              stream,
    const bool                enable_h_save
)
{
    const int D = d_pop_init.D;
    const int tot_genes = N * D;

    // Check if all dimensions share same bounds [0,1] (standard DTLZ)
    bool uniform_bounds = true;
    float lb0 = d_pop_init.h_bound[0];
    float ub0 = d_pop_init.h_bound[1];
    for (int i = 1; i < D; ++i) {
        if (d_pop_init.h_bound[2*i] != lb0 || d_pop_init.h_bound[2*i+1] != ub0) {
            uniform_bounds = false;
            break;
        }
    }

    constexpr int BLK_SIZE = 256;

    if (uniform_bounds) {
        // Fast path: direct generation in [lb, ub]
        launch_random_bounded_uniform_kernel(
            d_pop_init.d_pop, tot_genes, lb0, ub0,
            rnd_seed, glb_rnd_offset, BLK_SIZE, stream);
    } else {
        // Slow path: generate [0,1) then map to heterogeneous bounds
        launch_random_uniform_kernel(d_pop_init.d_pop, tot_genes, rnd_seed, glb_rnd_offset, BLK_SIZE, stream);
        // Map from [0,1) to [lb, ub] using bounds
        // For now, use the bounded kernel with lb=0, ub=1 as fallback
        // The bounds conversion will happen during evaluation
        // TODO: Add bounds_conversion_kernel if needed for heterogeneous bounds
    }

    saveDeviceArrayToBin<float>("h_pop_init", d_pop_init.d_pop, tot_genes, enable_h_save);
    saveDeviceArrayToBin<float>("h_bound_init", d_pop_init.d_bound, 2 * D, enable_h_save);
}
