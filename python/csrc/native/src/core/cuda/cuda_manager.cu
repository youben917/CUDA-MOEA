#include <iostream>

#include <cuda_runtime.h>

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_utils.cuh"
#include "cuda_moea/core/cuda/cuda_manager.cuh"

// ============================================================================================================================================== //
// RNDManager Implementation
// ============================================================================================================================================== //
RNDManager::RNDManager(
    ull _default_seed,    // Default seed 2887 - personal favorite :) (originated from the self-explosion code in Gundam Seed)
    int _island_idx
)
    : default_seed(_default_seed)   // User-provided global default seed (same across islands for reproducibility)
    , island_idx(_island_idx)    // Island identifier (used to derive an independent RNG stream per island)
    , island_seed(0ULL)         // Island-specific root seed derived from (default_seed, island_idx) via splitmix64 mixing
    , glb_rnd_seed(0ULL)        // Global random seed (host value)
    , glb_rnd_offset(0ULL)      // Global random offset counter (host value)
{
    assert(island_idx >= 0 && "Island ID must be non-negative");
    derive_seed();
    init();
}

void RNDManager::derive_seed() {
    // Splitmix64-style mixing for statistical independence between islands
    // Reference: https://prng.di.unimi.it/splitmix64.c
    // Combine default_seed with island_idx using golden ratio constant (2^64 / phi)
    
    // This ensures widely separated starting points for different islands
    ull x = default_seed + static_cast<ull>(island_idx) * 0x9E3779B97F4A7C15ULL;
    
    // First mixing round: XOR-shift right 30 bits, multiply by odd constant
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    
    // Second mixing round: XOR-shift right 27 bits, multiply by odd constant
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    
    // Final avalanche: XOR-shift right 31 bits for full bit diffusion
    island_seed = x ^ (x >> 31);
}

void RNDManager::init() {
    // Initialize host-side values directly (no device memory allocation needed)
    glb_rnd_seed   = island_seed;
    glb_rnd_offset = 0ULL;
}

void RNDManager::reset_offset() {
    // Reset offset counter to zero for reproducible random sequences
    glb_rnd_offset = 0ULL;
}

void RNDManager::reset_state() {
    // Restore seed to derived island seed and reset offset
    init();  // Equivalent: glb_rnd_seed = island_seed; glb_rnd_offset = 0;
}

void RNDManager::set_seed(ull new_default_seed) {
    // Update default seed and re-derive island-specific seed
    default_seed = new_default_seed;
    derive_seed();
    glb_rnd_seed = island_seed;
    glb_rnd_offset = 0ULL;
}

// ======================================================================================================================================== //
// CUDA Memory Pool
// ======================================================================================================================================== //
cudaError_t create_mempool(
    cudaMemPool_t& pool,
    cudaStream_t& stream,
    float reserved_ratio,
    int policy_level,
    int device)
{
    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    // create independent memory pool
    cudaMemPoolProps pool_props = {};
    pool_props.allocType     = cudaMemAllocationTypePinned;
    pool_props.handleTypes   = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id   = device;
    CUDA_CHECK(cudaMemPoolCreate(&pool, &pool_props));
    
    // setting the size of mempool
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    uint64_t release_threshold = static_cast<uint64_t>(free_mem * reserved_ratio);
    // Setting pool attribute
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,           &release_threshold));
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic,             &policy_level));
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies,      &policy_level));
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseFollowEventDependencies,        &policy_level));

    return cudaSuccess;
}
// ======================================================================================================================================== //
// This file manages global handles for cuBLAS and cuSOLVER libraries.
// It ensures that only one handle is created for each library and provides
// thread-safe access to these handles using the singleton pattern.

// Static members for CuBlasManager
cublasHandle_t CuBlasManager::handle_ = nullptr;  // The global cuBLAS handle, initially null
std::mutex CuBlasManager::mutex_;                 // Mutex to ensure thread-safe access to the handle
bool CuBlasManager::initialized_ = false;         // Flag indicating whether the handle is initialized
int CuBlasManager::ref_count_ = 0;                // Reference counter to track usage of the handle

// Returns the global cuBLAS handle, initializing it if not already done.
// Ensures thread safety using a mutex.
cublasHandle_t CuBlasManager::getHandle() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex to prevent concurrent access
    if (!initialized_) {                       // Check if the handle needs initialization
        initializeInternal();                  // Initialize the handle if not already initialized
    }
    return handle_;                            // Return the initialized handle
}

// Public method to initialize the cuBLAS manager.
// Thread-safe initialization of the handle.
void CuBlasManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    initializeInternal();                      // Perform the actual initialization
}

// Private method to create and configure the cuBLAS handle.
// Only called when the mutex is locked to ensure single initialization.
void CuBlasManager::initializeInternal() {
    if (initialized_) return;                  // Exit if already initialized to avoid redundant setup
    
    cublasStatus_t status = cublasCreate(&handle_);  // Create the cuBLAS handle
    if (status != CUBLAS_STATUS_SUCCESS) {     // Check for creation errors
        std::cerr << "Failed to create global cuBLAS handle! Error code: " << status << std::endl;
        throw std::runtime_error("cuBLAS initialization failed");  // Throw exception on failure
    }
    
    // Configure the handle to use Tensor Core acceleration for improved performance
    // cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH); // fast but low accuracy
    cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);   // balance
    // cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);  // high accuracy
    
    initialized_ = true;                       // Mark the handle as initialized
    std::cout << "[CuBLAS] Global handle initialized" << std::endl;
}

// Cleans up the cuBLAS handle and resets the manager's state.
// Should be called when the handle is no longer needed, e.g., at program exit.
void CuBlasManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    if (initialized_ && handle_) {             // Check if the handle exists and is initialized
        cublasDestroy(handle_);                // Destroy the cuBLAS handle
        handle_ = nullptr;                     // Reset the handle pointer to null
        initialized_ = false;                  // Reset the initialization flag
        ref_count_ = 0;                        // Reset the reference counter
        std::cout << "[CuBLAS] Global handle destroyed" << std::endl;
    }
}

// Increments the reference count for the cuBLAS handle.
// Called when a new user begins using the handle.
void CuBlasManager::addRef() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    ref_count_++;                              // Increment the reference count
}

// Decrements the reference count for the cuBLAS handle.
// Cleanup is not automatic in this implementation, even if count reaches zero.
void CuBlasManager::release() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    ref_count_--;                              // Decrement the reference count
    if (ref_count_ <= 0) {                     // Check if no users remain
        // Cleanup could be triggered here, but it is currently disabled
        // cleanup();
    }
}

// Checks whether the cuBLAS manager is initialized.
// Thread-safe check of the initialization status.
bool CuBlasManager::isInitialized() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    return initialized_;                       // Return the current initialization status
}

// Sets the math mode of the cuBLAS handle, e.g., to enable/disable Tensor Core usage.
// Only applies if the handle is initialized.
void CuBlasManager::setMathMode(cublasMath_t mode) {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    if (initialized_ && handle_) {             // Verify the handle is ready
        cublasStatus_t status = cublasSetMathMode(handle_, mode);  // Set the math mode
        if (status != CUBLAS_STATUS_SUCCESS) {  // Check for errors
            std::cerr << "Failed to set cuBLAS math mode! Error code: " << status << std::endl;
        } else {
            std::cout << "[CuBLAS] Math mode set to " << mode << std::endl;
        }
    }
}

// Retrieves the current math mode of the cuBLAS handle.
// Returns default mode if uninitialized or on error.
cublasMath_t CuBlasManager::getMathMode() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    if (initialized_ && handle_) {             // Check if the handle is initialized
        cublasMath_t mode;                     // Variable to store the mode
        cublasStatus_t status = cublasGetMathMode(handle_, &mode);  // Get the math mode
        if (status == CUBLAS_STATUS_SUCCESS) {  // Check if retrieval succeeded
            return mode;                       // Return the current mode
        }
    }
    return CUBLAS_DEFAULT_MATH;                // Return default mode if uninitialized or failed
}

//////////////////////////////////////////////////////////////////////////////////////

// Static members for CuSolverManager
cusolverDnHandle_t CuSolverManager::handle_ = nullptr;  // The global cuSOLVER handle, initially null
std::mutex CuSolverManager::mutex_;                     // Mutex to ensure thread-safe access
bool CuSolverManager::initialized_ = false;             // Flag indicating whether the handle is initialized
int CuSolverManager::ref_count_ = 0;                    // Reference counter to track usage of the handle

// Returns the global cuSOLVER handle, initializing it if not already done.
// Ensures thread safety using a mutex.
cusolverDnHandle_t CuSolverManager::getHandle() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex to prevent concurrent access
    if (!initialized_) {                       // Check if the handle needs initialization
        initializeInternal();                  // Initialize the handle if not already initialized
    }
    return handle_;                            // Return the initialized handle
}

// Public method to initialize the cuSOLVER manager.
// Thread-safe initialization of the handle.
void CuSolverManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    initializeInternal();                      // Perform the actual initialization
}

// Private method to create the cuSOLVER handle.
// Only called when the mutex is locked to ensure single initialization.
void CuSolverManager::initializeInternal() {
    if (initialized_) return;                  // Exit if already initialized to avoid redundant setup
    
    cusolverStatus_t status = cusolverDnCreate(&handle_);  // Create the cuSOLVER handle
    if (status != CUSOLVER_STATUS_SUCCESS) {   // Check for creation errors
        std::cerr << "Failed to create global cuSolver handle!" << std::endl;
        throw std::runtime_error("cuSolver initialization failed");  // Throw exception on failure
    }
    
    initialized_ = true;                       // Mark the handle as initialized
    std::cout << "[CuSolver] Global handle initialized" << std::endl;
}

// Cleans up the cuSOLVER handle and resets the manager's state.
// Should be called when the handle is no longer needed, e.g., at program exit.
void CuSolverManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    if (initialized_ && handle_) {             // Check if the handle exists and is initialized
        cusolverDnDestroy(handle_);            // Destroy the cuSOLVER handle
        handle_ = nullptr;                     // Reset the handle pointer to null
        initialized_ = false;                  // Reset the initialization flag
        ref_count_ = 0;                        // Reset the reference counter
        std::cout << "[CuSolver] Global handle destroyed" << std::endl;
    }
}

// Increments the reference count for the cuSOLVER handle.
// Called when a new user begins using the handle.
void CuSolverManager::addRef() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    ref_count_++;                              // Increment the reference count
}

// Decrements the reference count for the cuSOLVER handle.
// Cleanup is not automatic in this implementation, even if count reaches zero.
void CuSolverManager::release() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    ref_count_--;                              // Decrement the reference count
    if (ref_count_ <= 0) {                     // Check if no users remain
        // Cleanup could be triggered here, but it is currently disabled
        // cleanup();
    }
}

// Checks whether the cuSOLVER manager is initialized.
// Thread-safe check of the initialization status.
bool CuSolverManager::isInitialized() {
    std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for thread safety
    return initialized_;                       // Return the current initialization status
}
