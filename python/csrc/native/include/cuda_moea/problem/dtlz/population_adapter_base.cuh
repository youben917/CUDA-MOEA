#pragma once

#include "cuda_runtime.h"
#include <cstddef>

// ======================================================================================================================================================= //
//                                    IPopulationData - Abstract Base Class for Population Data                                                            //
// ======================================================================================================================================================= //
/**
 * @brief Abstract base class defining the interface for population data containers
 * 
 * This class provides a common interface for all population data structures used in
 * evolutionary algorithms (EA/GA). Derived classes implement application-specific
 * data layouts while maintaining a consistent memory management interface.
 * 
 * Design Philosophy:
 * - Decouples algorithm logic from data representation
 * - Enables polymorphic handling of different problem domains
 * - Standardizes CUDA memory pool integration
 * - Supports flexible memory allocation strategies (full/partial)
 * 
 * Inheritance Hierarchy:
 *   IPopulationData (abstract)
 *       │
 *       ├── PopData (RF SiP placement - coordinates + angles)
 *       │       └── PopInitData (with host initialization data)
 *       │
 *       └── (Future: other problem-specific implementations)
 * 
 * Memory Management Contract:
 * - All derived classes must implement mallocfrompool() and free()
 * - Memory operations are asynchronous (stream-ordered)
 * - CUDA memory pool used for efficient allocation/deallocation
 * - Supports partial allocation for memory sharing between populations
 */
struct IPopulationData {
    // =========================================================================
    // Core Parameters (Common to All Population Data)
    // =========================================================================
    int N;                        ///< Population size (number of individuals)

    // =========================================================================
    // Constructors & Destructor
    // =========================================================================
    /**
     * @brief Default constructor - initializes population size to 0
     */
    __host__ IPopulationData() : N(0) {}

    /**
     * @brief Parameterized constructor
     * @param _N  Population size
     */
    __host__ explicit IPopulationData(int _N) : N(_N) {}

    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     * 
     * IMPORTANT: Derived classes should call free() before destruction
     * if device memory was allocated. The destructor itself does NOT
     * free device memory to avoid implicit synchronization.
     */
    virtual ~IPopulationData() = default;

    // =========================================================================
    // Pure Virtual Interface - Dimension Information (Required)
    // =========================================================================
    /**
     * @brief Get the dimensionality of the search space
     * 
     * @return Number of decision variables per individual
     * 
     * For RF SiP placement: D = num_tot_insts * 2 (x,y coords) + num_tot_insts (angles)
     * For general optimization: D = number of continuous/discrete variables
     */
    __host__ virtual int get_dimension() const = 0;

    /**
     * @brief Get total number of elements in the population data
     * 
     * @return Total element count (N × dimension-specific size)
     * 
     * Used for buffer sizing and memory allocation calculations.
     */
    __host__ virtual size_t get_total_elements() const = 0;

    // =========================================================================
    // Pure Virtual Interface - Atomic Memory Operations (Required Building Blocks)
    // =========================================================================
    /**
     * @brief [ATOMIC OP - REQUIRED] Allocate per-individual population arrays
     * 
     * @param mempool  CUDA memory pool for allocation
     * @param stream   CUDA stream for async operations
     * 
     * Contract:
     * - Must allocate all per-individual arrays (e.g., coordinates, angles)
     * - Allocation must be stream-ordered (async)
     * - Must be idempotent (safe to call multiple times after free_pop)
     * 
     * This is a building block operation that MUST be implemented by all derived classes.
     * Every population has per-individual data.
     */
    __host__ virtual void mallocfrompool_pop(cudaMemPool_t mempool, cudaStream_t stream) = 0;

    /**
     * @brief [ATOMIC OP - REQUIRED] Free per-individual population arrays
     * 
     * @param stream  CUDA stream for async operations
     * 
     * Contract:
     * - Must free all arrays allocated by mallocfrompool_pop()
     * - Must reset all population data pointers to nullptr
     * - Must be safe to call even if memory was never allocated
     * - Must be idempotent (safe to call multiple times)
     * 
     * This is a building block operation that MUST be implemented by all derived classes.
     */
    __host__ virtual void free_pop(cudaStream_t stream) = 0;

    // =========================================================================
    // Virtual Interface - Optional Atomic Operations (Instance Attributes)
    // =========================================================================
    /**
     * @brief [ATOMIC OP - OPTIONAL] Allocate instance attribute arrays
     * 
     * @param mempool  CUDA memory pool for allocation
     * @param stream   CUDA stream for async operations
     * 
     * Default implementation: no operation (assumes no instance-level attributes).
     * 
     * Override in derived classes that have instance-level data shared across
     * all individuals (e.g., movability masks, bounds, constraints).
     * 
     * Use Case:
     * - RF SiP placement: instance movability masks, rotation masks, coordinate bounds
     * - TSP: city coordinates shared by all tours
     * - Job scheduling: machine capabilities shared by all schedules
     */
    __host__ virtual void mallocfrompool_inst_attrs(cudaMemPool_t mempool, cudaStream_t stream)
    {
        // Default: no instance attributes to allocate
        // Derived classes override if they have instance-level data
    }

    /**
     * @brief [ATOMIC OP - OPTIONAL] Free instance attribute arrays
     * 
     * @param stream  CUDA stream for async operations
     * 
     * Default implementation: no operation (assumes no instance-level attributes).
     * 
     * Override in derived classes that have instance-level data.
     */
    __host__ virtual void free_inst_attrs(cudaStream_t stream)
    {
        // Default: no instance attributes to free
        // Derived classes override if they have instance-level data
    }

    // =========================================================================
    // Virtual Interface - Composite Operations (Built from Atomic Operations)
    // =========================================================================
    /**
     * @brief [COMPOSITE OP] Allocate all device memory (population + instance attributes)
     * 
     * @param mempool  CUDA memory pool for allocation
     * @param stream   CUDA stream for async operations
     * 
     * Default implementation:
     *   mallocfrompool() = mallocfrompool_pop() + mallocfrompool_inst_attrs()
     * 
     * This composite operation calls the atomic operations in sequence.
     * Override only if you need custom allocation ordering or additional logic.
     * 
     * Contract:
     * - Must allocate all device arrays required by the derived class
     * - Must be idempotent (safe to call multiple times after free)
     */
    __host__ virtual void mallocfrompool(cudaMemPool_t mempool, cudaStream_t stream)
    {
        mallocfrompool_pop(mempool, stream);
        mallocfrompool_inst_attrs(mempool, stream);
    }

    /**
     * @brief [COMPOSITE OP] Free all device memory (population + instance attributes)
     * 
     * @param stream  CUDA stream for async operations
     * 
     * Default implementation:
     *   free() = free_pop() + free_inst_attrs()
     * 
     * This composite operation calls the atomic operations in sequence.
     * Override only if you need custom deallocation ordering or additional logic.
     * 
     * Contract:
     * - Must free all device memory allocated by mallocfrompool()
     * - Must reset all device pointers to nullptr
     * - Must be safe to call even if memory was never allocated
     * - Must be idempotent (safe to call multiple times)
     */
    __host__ virtual void free(cudaStream_t stream)
    {
        free_pop(stream);
        free_inst_attrs(stream);
    }

    // =========================================================================
    // Deleted Copy/Move Operations (Prevent Shallow Copy of Device Pointers)
    // =========================================================================
    IPopulationData(const IPopulationData&) = delete;            // Disallow copy construction (e.g., T b(a); / T b = a;)
    IPopulationData& operator=(const IPopulationData&) = delete; // Disallow copy assignment   (e.g., b = a;)
    
    IPopulationData(IPopulationData&& other) noexcept : N(other.N) { other.N = 0; }
    IPopulationData& operator=(IPopulationData&& other) noexcept {
        if (this != &other) { N = other.N; other.N = 0; }
        return *this;
    }

};

// ======================================================================================================================================================= //
//                                                    Type Aliases for Convenience                                                                         //
// ======================================================================================================================================================= //

/// Shared pointer type for polymorphic population data handling
// using PopDataPtr = std::shared_ptr<IPopulationData>;  // Uncomment if using shared_ptr

/// Raw pointer type for performance-critical code paths
using IPopDataRawPtr = IPopulationData*;
