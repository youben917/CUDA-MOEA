#pragma once

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_manager.cuh"
#include "population_adapter_base.cuh"

// ======================================================================================================================================================= //
//                                   PopData - Standard Test Population Data Implementation                                                                 //
// ======================================================================================================================================================= //
struct PopData : public IPopulationData {
    // =========================================================================
    // Device Pointers - Population Data (N individuals x D dimensions)
    // =========================================================================
    float* d_pop;       ///< (N, D) - decision variables, row-major
    float* d_bound;     ///< (2*D,) - interleaved bounds [lb0,ub0, lb1,ub1, ...]

    // =========================================================================
    // Application-Specific Parameters
    // =========================================================================
    int D;              ///< Number of decision variables per individual

    // =========================================================================
    // Constructors & Destructor
    // =========================================================================
    __host__ PopData();
    __host__ PopData(int _N, int _D);
    virtual ~PopData() = default;

    // =========================================================================
    // Copy/Move Semantics
    // =========================================================================
    PopData(const PopData&) = delete;
    PopData& operator=(const PopData&) = delete;
    __host__ PopData(PopData&& other) noexcept;
    __host__ PopData& operator=(PopData&& other) noexcept;

    // =========================================================================
    // IPopulationData Interface Implementation
    // =========================================================================
    __host__ int get_dimension() const override { return D; }
    __host__ size_t get_total_elements() const override {
        return static_cast<size_t>(N) * static_cast<size_t>(D);
    }

    // =========================================================================
    // Atomic Memory Operations
    // =========================================================================
    __host__ void mallocfrompool_pop(cudaMemPool_t mempool, cudaStream_t stream) override;
    __host__ void free_pop(cudaStream_t stream) override;

    __host__ void mallocfrompool_inst_attrs(cudaMemPool_t mempool, cudaStream_t stream) override;
    __host__ void free_inst_attrs(cudaStream_t stream) override;

protected:
    __host__ void reset_device_pointers();
    __host__ void reset_pop_device_pointers();
    __host__ void reset_inst_attrs_device_pointers();
};

// ======================================================================================================================================================= //
//                                   PopInitData - Population Initialization Data Container                                                                 //
// ======================================================================================================================================================= //
struct PopInitData : public PopData {
    // =========================================================================
    // Host-Side Initialization Data
    // =========================================================================
    std::vector<float> h_bound;   ///< (2*D,) - host copy of interleaved bounds

    // =========================================================================
    // Constructors
    // =========================================================================
    __host__ PopInitData(int _N, int _D, const std::vector<float>& _h_bound);
    virtual ~PopInitData() = default;

    // =========================================================================
    // Copy/Move Semantics
    // =========================================================================
    PopInitData(const PopInitData&) = delete;
    PopInitData& operator=(const PopInitData&) = delete;
    __host__ PopInitData(PopInitData&& other) noexcept;
    __host__ PopInitData& operator=(PopInitData&& other) noexcept;

    // =========================================================================
    // Factory Method
    // =========================================================================
    __host__ static PopInitData create_and_initialize(
        ull                        rnd_seed,
        ull&                       glb_rnd_offset,
        int                        N,
        int                        D,
        const std::vector<float>&  h_bound,
        cudaMemPool_t              mempool,
        cudaStream_t               stream,
        bool                       enable_h_save = false
    );

    // =========================================================================
    // Memory Management - Override with H2D Transfer
    // =========================================================================
    __host__ void mallocfrompool(cudaMemPool_t mempool, cudaStream_t stream) override;
    __host__ void free(cudaStream_t stream) override;
};

// ======================================================================================================================================================= //
//                                              Population Initialization API                                                                               //
// ======================================================================================================================================================= //
void generate_pop_init(
    ull                       rnd_seed,
    ull&                      glb_rnd_offset,
    PopInitData&              d_pop_init,
    int                       N,
    cudaMemPool_t             mempool,
    cudaStream_t              stream,
    const bool                enable_h_save
);
