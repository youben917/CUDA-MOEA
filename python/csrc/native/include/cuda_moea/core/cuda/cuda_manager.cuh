#pragma once

#include <mutex>
#include <cassert> 

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuda_globals.cuh"
#include "cuda_utils.cuh"
// ============================================================================================================================================== //
/**
 * @class RNDManager
 * @brief Self-contained random number generator state manager for CUDA NSGA-III.
 *
 * This class encapsulates the global random seed and offset used by CUDA random
 * number generators (Philox-based). It uses value semantics for host-side state
 * management and supports island model parallelism.
 *
 * Design principle:
 *   - Self-contained: manages seed derivation and offset tracking on host
 *   - Simple interface: construct, get values, pass to kernels
 *   - Island-aware: derives statistically independent seeds per island
 *
 * Usage (single population):
 *   RNDManager rng_mgr(2887ULL, 0);
 *   cuda_nsga3<<<...>>>(..., rng_mgr.get_seed(), rng_mgr.get_offset_ptr(), ...);
 *
 * Usage (island model):
 *   std::vector<RNDManager> island_rngs;
 *   for (int i = 0; i < num_islands; ++i) {
 *       island_rngs.emplace_back(2887ULL, i);
 *   }
 */
// ============================================================================================================================================== //
class RNDManager {
public:
    // ======================================================================== //
    /**
     * @brief Construct and initialize RNDManager for a specific island.
     *
     * Derives island-specific seed and initializes offset to zero.
     *
     * @param default_seed  Base seed value (will be combined with island_idx)
     * @param island_idx    Island index (0 for single-population mode)
     */
    // ======================================================================== //
    explicit RNDManager(
        ull default_seed = 2887ULL,
        int island_idx = 0
    );
    
    // ======================================================================== //
    /**
     * @brief Destructor - no dynamic resources to free (host-side values).
     */
    // ======================================================================== //
    ~RNDManager() = default;
    
    // Enable copy semantics (pure value types, safe to copy)
    RNDManager(const RNDManager&) = default;
    RNDManager& operator=(const RNDManager&) = default;
    
    // Enable move semantics
    RNDManager(RNDManager&& other) noexcept = default;
    RNDManager& operator=(RNDManager&& other) noexcept = default;
    
    // ======================================================================== //
    /**
     * @brief Reset offset to zero (for reproducibility across runs).
     */
    // ======================================================================== //
    void reset_offset();
    
    // ======================================================================== //
    /**
     * @brief Reset both seed and offset to initial state.
     */
    // ======================================================================== //
    void reset_state();
    
    // ======================================================================== //
    /**
     * @brief Update base seed and re-derive island-specific seed.
     *
     * @param new_default_seed  New base seed value
     */
    // ======================================================================== //
    void set_seed(ull new_default_seed);
    
    // ======================================================================== //
    // Accessors - Value getters for kernel parameter passing
    // ======================================================================== //
    ull  get_seed()          const { return glb_rnd_seed; }
    ull  get_offset()        const { return glb_rnd_offset; }

    // Reference accessors for functions that need to modify seed/offset
    ull&       get_seed_ref()         { return glb_rnd_seed; }
    const ull& get_seed_ref()   const { return glb_rnd_seed; }
    ull&       get_offset_ref()       { return glb_rnd_offset; }
    const ull& get_offset_ref() const { return glb_rnd_offset; }
    
    // ======================================================================== //
    /**
     * @brief Increment offset by specified amount (for manual offset management).
     *
     * @param delta  Amount to add to current offset
     */
    // ======================================================================== //
    void advance_offset(ull delta) { glb_rnd_offset += delta; }
    
    // Metadata accessors
    int  get_island_idx()    const { return island_idx; }
    ull  get_default_seed()     const { return default_seed; }
    ull  get_derived_seed()  const { return island_seed; }

private:
    ull  default_seed;      // User-provided base seed
    int  island_idx;        // Island index for parallel populations
    ull  island_seed;       // Derived seed (default_seed mixed with island_idx)
    ull  glb_rnd_seed;      // Global random seed (host value, passed to kernels)
    ull  glb_rnd_offset;    // Global random offset counter (host value)
    
    // ======================================================================== //
    /**
     * @brief Derive island-specific seed using splitmix64-style mixing.
     */
    // ======================================================================== //
    void derive_seed();
    
    // ======================================================================== //
    /**
     * @brief Initialize seed and offset values.
     */
    // ======================================================================== //
    void init();
};
// ========================================================================================================== //
// ============================================================================
// cudaStreamSync - CUDA Stream and Event Synchronization Manager
//
// Manages two CUDA streams with cross-stream synchronization via events:
//   - fcal_stream:  for fitness evaluation kernels and input data transfer
//   - exec_stream: for GA execution kernels (crossover, mutation, selection)
//
// Enables safe kernel launches when inputs come from multiple streams.
// ============================================================================
struct cudaStreamSync {
    cudaStream_t fcal_stream;  // Stream for fitness evaluation
    cudaStream_t exec_stream;  // Stream for GA island evolution
    cudaEvent_t  fcal_event;   // Marks fcal_stream progress
    cudaEvent_t  exec_event;   // Marks exec_stream progress

    // Constructor: Initialize all handles to nullptr
    __host__ cudaStreamSync()
        : fcal_stream(nullptr)
        , exec_stream(nullptr)
        , fcal_event(nullptr)
        , exec_event(nullptr) {}

    // Create streams and events (events use cudaEventDisableTiming for minimal overhead)
    __host__ void create() {
        CUDA_CHECK(cudaStreamCreate(&fcal_stream));
        CUDA_CHECK(cudaStreamCreate(&exec_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&fcal_event,  cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&exec_event, cudaEventDisableTiming));
    }

    // Destroy streams/events and reset to nullptr (safe to call multiple times)
    __host__ void destroy() {
        if (fcal_event) {
            CUDA_CHECK(cudaEventDestroy(fcal_event));
            fcal_event = nullptr;
        }
        if (exec_event) {
            CUDA_CHECK(cudaEventDestroy(exec_event));
            exec_event = nullptr;
        }
        if (fcal_stream) {
            CUDA_CHECK(cudaStreamDestroy(fcal_stream));
            fcal_stream = nullptr;
        }
        if (exec_stream) {
            CUDA_CHECK(cudaStreamDestroy(exec_stream));
            exec_stream = nullptr;
        }
    }

    // Record current progress on both streams
    __host__ void recordBoth() {
        CUDA_CHECK(cudaEventRecord(fcal_event,  fcal_stream));
        CUDA_CHECK(cudaEventRecord(exec_event, exec_stream));
    }

    // Make kernel_stream wait for both fcal_stream and exec_stream (non-blocking to host)
    __host__ void syncToStream(cudaStream_t kernel_stream = 0) {
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, fcal_event,  0));
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, exec_event, 0));
    }

    // Convenience: record + sync in one call (use before kernel launch)
    __host__ void sync_with_all_streams(cudaStream_t kernel_stream) {
        recordBoth();
        syncToStream(kernel_stream);
    }

    __host__ void wait_fcalstream_done_and_execute(cudaStream_t kernel_stream) {
        CUDA_CHECK(cudaEventRecord(fcal_event, fcal_stream));
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, fcal_event, 0));
    }
    
    __host__ void wait_execstream_done_and_execute(cudaStream_t kernel_stream) {
        CUDA_CHECK(cudaEventRecord(exec_event, exec_stream));
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, exec_event, 0));
    }
};
// ===================================================================================================================================== //
cudaError_t create_mempool(
    cudaMemPool_t& pool,
    cudaStream_t& stream,
    float reserved_ratio = 0.9f,
    int policy_level = 1,
    int device = 0);
// ======================================================================================== //
// ============================================================================
// cudaMemPools - CUDA Memory Pool Manager
//
// Manages stream-ordered memory pools for async allocation:
//   - fcal_pool:  for fitness evaluation data (RF links, obstructions, nets)
//   - exec_pool: for GA island operations (offspring, selection buffers)
//
// Default split: 30% fcal_pool, 70% exec_pool
// ============================================================================
struct cudaMemPools {
    cudaMemPool_t fcal_pool;  // Memory pool bound to fcal_stream
    cudaMemPool_t exec_pool; // Memory pool bound to exec_stream
    
    __host__ cudaMemPools()
        : fcal_pool(nullptr)
        , exec_pool(nullptr) {}

    // Create memory pools (requires valid streams in streamsync)
    __host__ void create(
        cudaStreamSync& streamsync,
        float inputpool_percentage  = 0.3f, // Fraction of GPU memory for fcal_pool
        float islandpool_percentage = 0.7f, // Fraction of GPU memory for exec_pool
        int mempool_policylvl = 1,          // Caching level: 0=none, 1=basic, 2=aggressive
        int device_idx = 0)                 // Target GPU device
    {
        assert(streamsync.fcal_stream  != nullptr && "fcal_stream not created!");
        assert(streamsync.exec_stream != nullptr && "exec_stream not created!");
        
        create_mempool(fcal_pool,  streamsync.fcal_stream,  inputpool_percentage,  mempool_policylvl, device_idx);
        create_mempool(exec_pool, streamsync.exec_stream, islandpool_percentage, mempool_policylvl, device_idx);
    }

    // Destroy pools and reset to nullptr (safe to call multiple times)
    __host__ void destroy() {
        if (fcal_pool) {
            CUDA_CHECK(cudaMemPoolDestroy(fcal_pool));
            fcal_pool = nullptr;
        }
        if (exec_pool) {
            CUDA_CHECK(cudaMemPoolDestroy(exec_pool));
            exec_pool = nullptr;
        }
    }
};
// ===================================================================================================================================== //
// Singleton manager for cuBLAS handle with thread-safe reference counting
class CuBlasManager {
public:
    static cublasHandle_t getHandle();
    static void initialize();
    static void cleanup();
    static void addRef();
    static void release();
    static bool isInitialized();
    static void setMathMode(cublasMath_t mode);
    static cublasMath_t getMathMode();

private:
    static void initializeInternal();
    
    static cublasHandle_t handle_;
    static std::mutex mutex_;
    static bool initialized_;
    static int ref_count_;
    
    // Prevent instantiation
    CuBlasManager() = delete;
    ~CuBlasManager() = delete;
    CuBlasManager(const CuBlasManager&) = delete;
    CuBlasManager& operator=(const CuBlasManager&) = delete;
};

// Singleton manager for cuSOLVER handle with thread-safe reference counting
class CuSolverManager {
public:
    static cusolverDnHandle_t getHandle();
    static void initialize();
    static void cleanup();
    static void addRef();
    static void release();
    static bool isInitialized();

private:
    static void initializeInternal();
    
    static cusolverDnHandle_t handle_;
    static std::mutex mutex_;
    static bool initialized_;
    static int ref_count_;
    
    // Prevent instantiation
    CuSolverManager() = delete;
    ~CuSolverManager() = delete;
    CuSolverManager(const CuSolverManager&) = delete;
    CuSolverManager& operator=(const CuSolverManager&) = delete;
};