/*
 * CUDA Random Number Generation Library
 * 
 * This file implements optimized random number generation kernels for CUDA GPUs.
 * It supports multiple types of random number generation including uniform floats,
 * boolean values, integers within specified ranges, packed float2 values, and 
 * packed int2 values for optimized memory access.
 * The implementation is optimized for different GPU architectures
 * (RTX 4060, 4090, 5090, and RTX 6000 Pro)
 * with architecture-specific threading configurations.
 */

#include <curand_kernel.h>    // CUDA random number generation library
#include <stdio.h>            // Standard I/O for printf functions
#include <algorithm>          // STL algorithms (std::min)

#include "cuda_moea/core/cuda/cuda_globals.cuh"   // Custom CUDA global definitions
#include "cuda_moea/core/cuda/cuda_utils.cuh"     // Custom CUDA utility functions

//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef UNROLL_FACTOR_CUH
#define UNROLL_FACTOR_CUH

namespace unroll_factor{
    constexpr int UNROLL_FACTOR = 4;
}
#endif

// ================================================================================
// Common Utility Functions (Fully Reusable)
// ================================================================================
using namespace gpu::models;
using namespace unroll_factor;

// Query maximum GPU occupancy (cached for performance)
inline uint32_t get_max_occupied_threads() {
    static uint32_t fully_occupied_num_threads = 0;
    if (fully_occupied_num_threads == 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        uint32_t max_threads_per_sm = props.maxThreadsPerMultiProcessor;
        uint32_t sm_count = props.multiProcessorCount;
        fully_occupied_num_threads = max_threads_per_sm * sm_count;
    }
    return fully_occupied_num_threads;
};


/**
 * @brief Reverses boundary conditions for uniform random numbers
 * @param value Input random value
 * @return 0.0f if value is exactly 1.0f, otherwise returns the original value
 * 
 * This function ensures that uniform random numbers are in the range [0, 1)
 * instead of [0, 1] by converting any 1.0f values to 0.0f.
 */
__device__ __forceinline__ float reverse_bounds(float value) {
    return (value == 1.0f) ? 0.0f : value;
}

/**
 * @brief Calculates rounded size for memory-aligned operations
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor
 * @param count Total number of elements to process
 * @param stride Grid stride (total threads in grid)
 * @return Rounded size that's aligned to stride * UNROLL_FACTOR boundaries
 * 
 * This ensures memory operations are aligned and optimized for vectorized access.
 */
template<const int UNROLL_FACTOR>
inline int cal_rounded_size(int count, int stride) {
    return ((count - 1) / (stride * UNROLL_FACTOR) + 1) * stride * UNROLL_FACTOR;
}

/**
 * @brief Calculates optimal execution policy for CUDA kernel launches
 * @param count Total number of elements to process
 * @param block_size Number of threads per block (default: 256)
 * @return Tuple containing: counter_offset, grid_dim, block_dim, use_grid_stride, stride, rounded_size
 * 
 * This function determines the optimal grid and block configuration based on:
 * - The number of elements to process
 * - Hardware constraints (maximum blocks)
 * - Whether grid-stride loops are needed for large datasets
 */
std::tuple<ull, dim3, dim3, bool, int, int> calc_execution_policy(int count, const int block_size) {
    // Configure block dimensions
    dim3 dim_block(block_size);
    
    // Calculate initial grid size based on element count
    dim3 grid((count + block_size - 1) / block_size);
    
    // Get hardware-specific maximum number of blocks
    uint32_t fully_occupied_num_threads = get_max_occupied_threads();
    uint32_t fully_occupied_num_blocks  = fully_occupied_num_threads / block_size;
    
    // Limit grid size to hardware maximum
    grid.x = std::min(fully_occupied_num_blocks, grid.x);
    
    // Calculate total number of threads that will be launched
    int stride = grid.x * block_size;
    
    // Determine if we need grid-stride loops for large datasets
    bool use_grid_stride = (count > stride);
    
    int rounded_size = 0;
    ull counter_offset;
    
    if (count > stride) {
        // For large datasets: calculate rounded size and offset for grid-stride loops
        rounded_size = cal_rounded_size<UNROLL_FACTOR>(count, stride);
        counter_offset = ((count - 1) / (stride * UNROLL_FACTOR) + 1) * UNROLL_FACTOR;
    } else {
        // For small datasets: each thread calls curand4 once
        counter_offset = UNROLL_FACTOR;
    }
    
    return std::make_tuple(counter_offset, grid, dim_block, use_grid_stride, stride, rounded_size);
}

// ================================================================================
// Uniform Random Number Generator
// ================================================================================

/**
 * @brief Generator for uniform random numbers in range [0, 1)
 * 
 * This struct encapsulates the logic for generating uniform random floating-point
 * numbers using CUDA's cuRAND library with proper boundary handling.
 */
struct UniformGenerator {
    /**
     * @brief Generate a single uniform random float
     * @param state cuRAND state for the current thread
     * @return Random float in range [0, 1)
     */
    __device__ float operator()(curandStatePhilox4_32_10_t& state) const {
        return reverse_bounds(curand_uniform(&state));
    }
    
    /**
     * @brief Generate four uniform random floats simultaneously
     * @param state cuRAND state for the current thread
     * @return float4 containing four random values in range [0, 1)
     * 
     * This vectorized version improves performance by generating 4 numbers at once.
     */
    __device__ float4 generate4(curandStatePhilox4_32_10_t& state) const {
        float4 rand = curand_uniform4(&state);
        rand.x = reverse_bounds(rand.x);
        rand.y = reverse_bounds(rand.y);
        rand.z = reverse_bounds(rand.z);
        rand.w = reverse_bounds(rand.w);
        return rand;
    }
};

/**
 * @brief Simple uniform random kernel for small datasets
 * @param rnd_output Output array to store random numbers
 * @param count Number of random numbers to generate
 * @param glb_rnd_offset Global random number offset for reproducibility
 * 
 * This kernel is used when the dataset is small enough that each thread
 * processes exactly one element without needing grid-stride loops.
 */
__global__ void random_uniform_kernel(
    float* rnd_output,                 // output: (count,)  
    int count,
    ull rnd_seed, ull glb_rnd_offset) {
    
    // Calculate global thread index
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;  // Bounds check
    
    ull local_offset = glb_rnd_offset;

    // Initialize cuRAND state for this thread
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    // Generate and store random number
    UniformGenerator generator;
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized uniform random kernel for large datasets using grid-stride loops
 * @tparam STRIDE Template parameter for grid stride (total threads)
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor
 * @param rnd_output Output array to store random numbers
 * @param count Number of random numbers to generate
 * @param rounded_size Rounded size for memory alignment
 * @param glb_rnd_offset Global random number offset for reproducibility
 * 
 * This kernel uses grid-stride loops to handle datasets larger than the number
 * of available threads, with vectorized operations for improved performance.
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_uniform_kernel(
    float* rnd_output,                // output: (count,)           
    int count,
    int rounded_size,
    ull rnd_seed, ull glb_rnd_offset) 
{
    // Calculate global thread index
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    // Initialize cuRAND state for this thread
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    UniformGenerator generator;
    
    // Grid-stride loop: each thread processes multiple elements
    for (int linear_idx = tidx; linear_idx < rounded_size; linear_idx += STRIDE * UNROLL_FACTOR) {
        // Generate 4 random numbers at once for efficiency
        float4 rand = generator.generate4(state);
        
        // Unroll the loop for storing the 4 values
        #pragma unroll
        for (int ii = 0; ii < UNROLL_FACTOR; ii++) {
            int li = linear_idx + STRIDE * ii;
            if (li < count) {  // Bounds check
                float value = (&rand.x)[ii];  // Access float4 components as array
                rnd_output[li] = value;
            }
        }
    }
}

// ================================================================================
// Boolean Random Number Generator
// ================================================================================

/**
 * @brief Generator for boolean random numbers (0 or 1)
 * 
 * This struct generates random boolean values by using the least significant
 * bit of random integers.
 */
struct BooleanGenerator {
    /**
     * @brief Generate a single random boolean (0 or 1)
     * @param state cuRAND state for the current thread
     * @return Random integer (0 or 1)
     */
    __device__ int operator()(curandStatePhilox4_32_10_t& state) const {
        // Use LSB of random integer to get 0 or 1
        float value = static_cast<float>(curand(&state) & 1);
        return static_cast<int>(value);
    }
    
    /**
     * @brief Generate four random booleans simultaneously
     * @param state cuRAND state for the current thread
     * @return uint4 containing four random boolean values
     */
    __device__ uint4 generate4(curandStatePhilox4_32_10_t& state) const {
        uint4 rand4 = curand4(&state);
        uint4 result;
        // Extract LSB from each component to get boolean values
        result.x = static_cast<int>(static_cast<float>(rand4.x & 1));
        result.y = static_cast<int>(static_cast<float>(rand4.y & 1));
        result.z = static_cast<int>(static_cast<float>(rand4.z & 1));
        result.w = static_cast<int>(static_cast<float>(rand4.w & 1));
        return result;
    }
};

/**
 * @brief Simple boolean random kernel for small datasets
 * @param rnd_output Output array to store random boolean values
 * @param count Number of random booleans to generate
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
__global__ void random_boolean_kernel(int* rnd_output, int count, ull rnd_seed, ull glb_rnd_offset) {
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    BooleanGenerator generator;
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized boolean random kernel for large datasets using grid-stride loops
 * @tparam STRIDE Template parameter for grid stride
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor
 * @param rnd_output Output array to store random boolean values
 * @param count Number of random booleans to generate
 * @param rounded_size Rounded size for memory alignment
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_boolean_kernel(
    int* rnd_output,
    int count,
    int rounded_size,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    BooleanGenerator generator;
    
    // Grid-stride loop with vectorized operations
    for (int linear_idx = tidx; linear_idx < rounded_size; linear_idx += STRIDE * UNROLL_FACTOR) {
        uint4 rand4 = generator.generate4(state);
        
        #pragma unroll
        for (int ii = 0; ii < UNROLL_FACTOR; ii++) {
            int li = linear_idx + STRIDE * ii;
            if (li < count) {
                uint value = (&rand4.x)[ii];
                rnd_output[li] = static_cast<int>(value);
            }
        }
    }
}

// ================================================================================
// Integer Range Random Number Generator
// ================================================================================

/**
 * @brief Generator for random integers within a specified range [lb, ub)
 * 
 * This struct generates random integers uniformly distributed within the
 * specified range using modulo operations.
 */
struct IntRangeGenerator {
    int lb, ub;  // Lower bound (inclusive) and upper bound (exclusive)
    
    /**
     * @brief Constructor to set the range bounds
     * @param l Lower bound (inclusive)
     * @param u Upper bound (exclusive)
     */
    __device__ IntRangeGenerator(int l, int u) : lb(l), ub(u) {}
    
    /**
     * @brief Generate a single random integer in range [lb, ub)
     * @param state cuRAND state for the current thread
     * @return Random integer in the specified range
     */
    __device__ int operator()(curandStatePhilox4_32_10_t& state) const {
        int range   = ub - lb;
        float value = lb + (curand(&state) % range);
        return static_cast<int>(value);
    }
    
    /**
     * @brief Generate four random integers simultaneously
     * @param state cuRAND state for the current thread
     * @return uint4 containing four random integers in the specified range
     */
    __device__ uint4 generate4(curandStatePhilox4_32_10_t& state) const {
        uint4 rand4 = curand4(&state);
        uint4 result;
        int range = ub - lb;
        // Apply modulo operation to each component to get values in range
        result.x = static_cast<int>(lb + (rand4.x % range));
        result.y = static_cast<int>(lb + (rand4.y % range));
        result.z = static_cast<int>(lb + (rand4.z % range));
        result.w = static_cast<int>(lb + (rand4.w % range));
        return result;
    }
};

/**
 * @brief Simple integer range random kernel for small datasets
 * @param rnd_output Output array to store random integers
 * @param count Number of random integers to generate
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
__global__ void random_int_kernel(int* rnd_output, int count, int lb, int ub, ull rnd_seed, ull glb_rnd_offset) {
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    IntRangeGenerator generator{lb, ub};
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized integer range random kernel for large datasets using grid-stride loops
 * @tparam STRIDE Template parameter for grid stride
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor
 * @param rnd_output Output array to store random integers
 * @param count Number of random integers to generate
 * @param rounded_size Rounded size for memory alignment
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_int_kernel(
    int* rnd_output,
    int count,
    int rounded_size,
    int lb,
    int ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    IntRangeGenerator generator{lb, ub};
    
    // Grid-stride loop with vectorized operations
    for (int linear_idx = tidx; linear_idx < rounded_size; linear_idx += STRIDE * UNROLL_FACTOR) {
        uint4 rand4 = generator.generate4(state);
        
        #pragma unroll
        for (int ii = 0; ii < UNROLL_FACTOR; ii++) {
            int li = linear_idx + STRIDE * ii;
            if (li < count) {
                uint value = (&rand4.x)[ii];
                rnd_output[li] = static_cast<int>(value);
            }
        }
    }
}

// ================================================================================
// Float2 Packed Random Number Generator
// ================================================================================

/**
 * @brief Generator for packed float2 random numbers for optimized memory access
 * 
 * This struct generates packed float2 values to reduce memory bandwidth by 50%
 * when two random values are needed per element (e.g., mutation operations)
 */
struct Float2Generator {
    /**
     * @brief Generate a single float2 packed random value
     * @param state cuRAND state for the current thread
     * @return float2 containing two random values in range [0, 1)
     */
    __device__ float2 operator()(curandStatePhilox4_32_10_t& state) const {
        float2 result;
        result.x = reverse_bounds(curand_uniform(&state));
        result.y = reverse_bounds(curand_uniform(&state));
        return result;
    }
    
    /**
     * @brief Generate two float2 values simultaneously using curand4
     * @param state cuRAND state for the current thread  
     * @param out1 First float2 output
     * @param out2 Second float2 output
     * 
     * This fully utilizes curand4 by packing all 4 values into 2 float2s
     */
    __device__ void generate2_float2(curandStatePhilox4_32_10_t& state, 
                                     float2& out1, float2& out2) const {
        float4 rand = curand_uniform4(&state);
        rand.x = reverse_bounds(rand.x);
        rand.y = reverse_bounds(rand.y);
        rand.z = reverse_bounds(rand.z);
        rand.w = reverse_bounds(rand.w);
        
        out1 = make_float2(rand.x, rand.y);
        out2 = make_float2(rand.z, rand.w);
    }
};

/**
 * @brief Simple float2 packed random kernel for small datasets
 * @param rnd_output Output array to store packed float2 random values
 * @param count Number of float2 elements to generate
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
__global__ void random_float2_kernel(
    float2* rnd_output,
    int count,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    Float2Generator generator;
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized float2 packed random kernel for large datasets using grid-stride loops
 * @tparam STRIDE Template parameter for grid stride (total threads)
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor (should be 2 for float2)
 * @param rnd_output Output array to store packed float2 random values
 * @param count Number of float2 elements to generate
 * @param rounded_size Rounded size for memory alignment
 * @param glb_rnd_offset Global random number offset for reproducibility
 * 
 * This kernel maximally utilizes curand4 by generating 4 values and packing them
 * into 2 float2 structures, achieving 100% efficiency
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_float2_kernel(
    float2* rnd_output,
    int count,
    int rounded_size,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    Float2Generator generator;
    
    // Grid-stride loop: each thread processes multiple float2 elements
    // We process 2 float2s at a time to fully utilize curand4
    for (int linear_idx = tidx * 2; linear_idx < rounded_size; linear_idx += STRIDE * 2) {
        float2 val1, val2;
        generator.generate2_float2(state, val1, val2);
        
        // Store the two float2 values
        if (linear_idx < count) {
            rnd_output[linear_idx] = val1;
        }
        if (linear_idx + 1 < count) {
            rnd_output[linear_idx + 1] = val2;
        }
    }
}

// ================================================================================
// Float4 Packed Random Number Generator
// ================================================================================

/**
 * @brief Generator for packed float4 random numbers for optimized memory access
 * 
 * This struct generates packed float4 values to maximize memory bandwidth efficiency
 * when four random values are needed per element. This is particularly useful for
 * operations like polynomial mutation where multiple random values are required
 * per instance (e.g., decision + mu_x + mu_y + reserved).
 * 
 * The generator directly maps to curand_uniform4, achieving 100% utilization
 * of the underlying random number generation hardware.
 */
struct Float4Generator {
    /**
     * @brief Generate a single float4 packed random value
     * @param state cuRAND state for the current thread
     * @return float4 containing four random values in range [0, 1)
     * 
     * This directly utilizes curand_uniform4 for maximum efficiency,
     * generating all 4 values in a single hardware operation.
     */
    __device__ float4 operator()(curandStatePhilox4_32_10_t& state) const {
        float4 rand = curand_uniform4(&state);
        rand.x = reverse_bounds(rand.x);
        rand.y = reverse_bounds(rand.y);
        rand.z = reverse_bounds(rand.z);
        rand.w = reverse_bounds(rand.w);
        return rand;
    }
    
    /**
     * @brief Generate multiple float4 values for grid-stride processing
     * @param state cuRAND state for the current thread
     * @param out Output array of float4 values (size = UNROLL_FACTOR)
     * @tparam UNROLL_FACTOR Number of float4 values to generate
     * 
     * This method generates UNROLL_FACTOR float4 values consecutively,
     * enabling efficient grid-stride loop processing with full vectorization.
     */
    template<int UNROLL_FACTOR>
    __device__ void generate_n(curandStatePhilox4_32_10_t& state, float4* out) const {
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            out[i] = (*this)(state);
        }
    }
};

/**
 * @brief Simple float4 packed random kernel for small datasets
 * @param rnd_output Output array to store packed float4 random values
 * @param count Number of float4 elements to generate
 * @param glb_rnd_offset Global random number offset for reproducibility
 * 
 * This kernel is used when the dataset is small enough that each thread
 * processes exactly one float4 element without needing grid-stride loops.
 * Each thread generates 4 random values packed into a single float4.
 */
__global__ void random_float4_kernel(
    float4* rnd_output,
    int count,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    Float4Generator generator;
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized float4 packed random kernel for large datasets using grid-stride loops
 * @tparam STRIDE Template parameter for grid stride (total threads)
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor
 * @param rnd_output Output array to store packed float4 random values
 * @param count Number of float4 elements to generate
 * @param rounded_size Rounded size for memory alignment
 * @param glb_rnd_offset Global random number offset for reproducibility
 * 
 * This kernel uses grid-stride loops to handle datasets larger than the number
 * of available threads. Each iteration generates UNROLL_FACTOR float4 values,
 * with each float4 containing 4 random values, for a total of 4*UNROLL_FACTOR
 * random values per iteration per thread.
 * 
 * Memory access pattern is optimized for coalescing: consecutive threads
 * access consecutive float4 elements in global memory.
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_float4_kernel(
    float4* rnd_output,
    int count,
    int rounded_size,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    Float4Generator generator;
    
    // Grid-stride loop: each thread processes multiple float4 elements
    // Each iteration processes UNROLL_FACTOR float4 values
    for (int linear_idx = tidx; linear_idx < rounded_size; linear_idx += STRIDE * UNROLL_FACTOR) {
        // Generate UNROLL_FACTOR float4 values
        float4 rand_vals[UNROLL_FACTOR];
        generator.generate_n<UNROLL_FACTOR>(state, rand_vals);
        
        // Store the generated values with bounds checking
        #pragma unroll
        for (int ii = 0; ii < UNROLL_FACTOR; ii++) {
            int li = linear_idx + STRIDE * ii;
            if (li < count) {
                rnd_output[li] = rand_vals[ii];
            }
        }
    }
}

// ================================================================================
// Int2 Packed Random Number Generator
// ================================================================================

/**
 * @brief Generator for packed int2 random numbers for optimized memory access
 * 
 * This struct generates packed int2 values to reduce memory bandwidth by 50%
 * when two random integer values are needed per element. Supports range generation.
 */
struct Int2Generator {
    int lb, ub;  // Lower bound (inclusive) and upper bound (exclusive)
    
    /**
     * @brief Constructor to set the range bounds
     * @param l Lower bound (inclusive, default: INT_MIN)
     * @param u Upper bound (exclusive, default: INT_MAX)
     */
    __device__ Int2Generator(int l = INT_MIN, int u = INT_MAX) : lb(l), ub(u) {}
    
    /**
     * @brief Generate a single int2 packed random value
     * @param state cuRAND state for the current thread
     * @return int2 containing two random values in range [lb, ub)
     */
    __device__ int2 operator()(curandStatePhilox4_32_10_t& state) const {
        int2 result;
        if (lb == INT_MIN && ub == INT_MAX) {
            // Full range: use raw random values
            result.x = static_cast<int>(curand(&state));
            result.y = static_cast<int>(curand(&state));
        } else {
            // Specific range: use modulo
            int range = ub - lb;
            result.x = lb + (curand(&state) % range);
            result.y = lb + (curand(&state) % range);
        }
        return result;
    }
    
    /**
     * @brief Generate two int2 values simultaneously using curand4
     * @param state cuRAND state for the current thread  
     * @param out1 First int2 output
     * @param out2 Second int2 output
     * 
     * This fully utilizes curand4 by packing all 4 values into 2 int2s
     */
    __device__ void generate2_int2(curandStatePhilox4_32_10_t& state, 
                                    int2& out1, int2& out2) const {
        uint4 rand = curand4(&state);
        
        if (lb == INT_MIN && ub == INT_MAX) {
            // Full range: use raw random values
            out1 = make_int2(static_cast<int>(rand.x), static_cast<int>(rand.y));
            out2 = make_int2(static_cast<int>(rand.z), static_cast<int>(rand.w));
        } else {
            // Specific range: apply modulo to each component
            int range = ub - lb;
            out1 = make_int2(lb + (rand.x % range), lb + (rand.y % range));
            out2 = make_int2(lb + (rand.z % range), lb + (rand.w % range));
        }
    }
};

/**
 * @brief Simple int2 packed random kernel for small datasets
 * @param rnd_output Output array to store packed int2 random values
 * @param count Number of int2 elements to generate
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
__global__ void random_int2_kernel(
    int2* rnd_output,
    int count,
    int lb,
    int ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    Int2Generator generator{lb, ub};
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized int2 packed random kernel for large datasets using grid-stride loops
 * @tparam STRIDE Template parameter for grid stride (total threads)
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor (should be 2 for int2)
 * @param rnd_output Output array to store packed int2 random values
 * @param count Number of int2 elements to generate
 * @param rounded_size Rounded size for memory alignment
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Global random number offset for reproducibility
 * 
 * This kernel maximally utilizes curand4 by generating 4 values and packing them
 * into 2 int2 structures, achieving 100% efficiency
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_int2_kernel(
    int2* rnd_output,
    int count,
    int rounded_size,
    int lb,
    int ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    Int2Generator generator{lb, ub};
    
    // Grid-stride loop: each thread processes multiple int2 elements
    // We process 2 int2s at a time to fully utilize curand4
    for (int linear_idx = tidx * 2; linear_idx < rounded_size; linear_idx += STRIDE * 2) {
        int2 val1, val2;
        generator.generate2_int2(state, val1, val2);
        
        // Store the two int2 values
        if (linear_idx < count) {
            rnd_output[linear_idx] = val1;
        }
        if (linear_idx + 1 < count) {
            rnd_output[linear_idx + 1] = val2;
        }
    }
}


// ================================================================================
// Bounded Uniform Generator (Single Float)
// ================================================================================

/**
 * @brief Generator for uniform random floats in arbitrary range [lb, ub)
 * 
 * This struct extends UniformGenerator to support custom bounds through
 * linear transformation: x = lb + u * (ub - lb), where u ∈ [0, 1)
 */
struct BoundedUniformGenerator {
    float lb, ub;  // Lower bound (inclusive) and upper bound (exclusive)
    
    /**
     * @brief Constructor to set the range bounds
     * @param l Lower bound (inclusive)
     * @param u Upper bound (exclusive)
     */
    __device__ BoundedUniformGenerator(float l, float u) : lb(l), ub(u) {}
    
    /**
     * @brief Generate a single uniform random float in [lb, ub)
     * @param state cuRAND state for the current thread
     * @return Random float in range [lb, ub)
     */
    __device__ float operator()(curandStatePhilox4_32_10_t& state) const {
        float u_norm = reverse_bounds(curand_uniform(&state));  // u ∈ [0, 1)
        float range = ub - lb;
        return fmaf(u_norm, range, lb);  // FMA: u_norm * range + lb
    }
    
    /**
     * @brief Generate four uniform random floats simultaneously
     * @param state cuRAND state for the current thread
     * @return float4 containing four random values in range [lb, ub)
     */
    __device__ float4 generate4(curandStatePhilox4_32_10_t& state) const {
        float4 rand = curand_uniform4(&state);
        
        // Apply boundary reversal to ensure [0, 1)
        rand.x = reverse_bounds(rand.x);
        rand.y = reverse_bounds(rand.y);
        rand.z = reverse_bounds(rand.z);
        rand.w = reverse_bounds(rand.w);
        
        // Scale to [lb, ub)
        float range = ub - lb;
        rand.x = fmaf(rand.x, range, lb);
        rand.y = fmaf(rand.y, range, lb);
        rand.z = fmaf(rand.z, range, lb);
        rand.w = fmaf(rand.w, range, lb);
        
        return rand;
    }
};

// ================================================================================
// Bounded Uniform Kernels (Simple + Grid-Stride)
// ================================================================================

/**
 * @brief Simple bounded uniform random kernel for small datasets
 * @param rnd_output Output array to store random numbers in [lb, ub)
 * @param count Number of random numbers to generate
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Global random number offset for reproducibility
 */
__global__ void random_bounded_uniform_kernel(
    float* rnd_output,
    int count,
    float lb,
    float ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    BoundedUniformGenerator generator{lb, ub};
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized bounded uniform random kernel for large datasets
 * @tparam STRIDE Template parameter for grid stride (total threads)
 * @tparam UNROLL_FACTOR Template parameter for vectorization factor
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_bounded_uniform_kernel(
    float* rnd_output,
    int count,
    int rounded_size,
    float lb,
    float ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    BoundedUniformGenerator generator{lb, ub};
    
    // Grid-stride loop with vectorized operations
    for (int linear_idx = tidx; linear_idx < rounded_size; linear_idx += STRIDE * UNROLL_FACTOR) {
        float4 rand = generator.generate4(state);
        
        #pragma unroll
        for (int ii = 0; ii < UNROLL_FACTOR; ii++) {
            int li = linear_idx + STRIDE * ii;
            if (li < count) {
                float value = (&rand.x)[ii];
                rnd_output[li] = value;
            }
        }
    }
}

// ================================================================================
// Bounded Float2 Generator (Packed for 50% Memory Bandwidth Reduction)
// ================================================================================

/**
 * @brief Generator for packed float2 random numbers in arbitrary range [lb, ub)
 * 
 * This struct generates packed float2 values to reduce memory bandwidth by 50%
 * when two random values are needed per element, now with custom bounds support
 */
struct BoundedFloat2Generator {
    float lb, ub;  // Lower bound (inclusive) and upper bound (exclusive)
    
    /**
     * @brief Constructor to set the range bounds
     * @param l Lower bound (inclusive)
     * @param u Upper bound (exclusive)
     */
    __device__ BoundedFloat2Generator(float l, float u) : lb(l), ub(u) {}
    
    /**
     * @brief Generate a single float2 packed random value in [lb, ub)
     * @param state cuRAND state for the current thread
     * @return float2 containing two random values in range [lb, ub)
     */
    __device__ float2 operator()(curandStatePhilox4_32_10_t& state) const {
        float2 result;
        float range = ub - lb;
        
        result.x = reverse_bounds(curand_uniform(&state));
        result.y = reverse_bounds(curand_uniform(&state));
        
        result.x = fmaf(result.x, range, lb);
        result.y = fmaf(result.y, range, lb);
        
        return result;
    }
    
    /**
     * @brief Generate two float2 values simultaneously using curand4
     * @param state cuRAND state for the current thread  
     * @param out1 First float2 output
     * @param out2 Second float2 output
     * 
     * This fully utilizes curand4 by packing all 4 values into 2 float2s
     */
    __device__ void generate2_float2(curandStatePhilox4_32_10_t& state, 
                                     float2& out1, float2& out2) const {
        float4 rand = curand_uniform4(&state);
        
        // Apply boundary reversal
        rand.x = reverse_bounds(rand.x);
        rand.y = reverse_bounds(rand.y);
        rand.z = reverse_bounds(rand.z);
        rand.w = reverse_bounds(rand.w);
        
        // Scale to [lb, ub)
        float range = ub - lb;
        rand.x = fmaf(rand.x, range, lb);
        rand.y = fmaf(rand.y, range, lb);
        rand.z = fmaf(rand.z, range, lb);
        rand.w = fmaf(rand.w, range, lb);
        
        out1 = make_float2(rand.x, rand.y);
        out2 = make_float2(rand.z, rand.w);
    }
};

// ================================================================================
// Bounded Float2 Kernels (Simple + Grid-Stride)
// ================================================================================

/**
 * @brief Simple bounded float2 packed random kernel for small datasets
 */
__global__ void random_bounded_float2_kernel(
    float2* rnd_output,
    int count,
    float lb,
    float ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidxx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidxx >= count) return;
    
    ull local_offset = glb_rnd_offset;
    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidxx, local_offset, &state);
    
    BoundedFloat2Generator generator{lb, ub};
    rnd_output[tidxx] = generator(state);
}

/**
 * @brief Optimized bounded float2 packed random kernel for large datasets
 */
template<const int STRIDE, const int UNROLL_FACTOR>
__global__ void random_bounded_float2_kernel(
    float2* rnd_output,
    int count,
    int rounded_size,
    float lb,
    float ub,
    ull rnd_seed, ull glb_rnd_offset) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull local_offset = glb_rnd_offset;

    curandStatePhilox4_32_10_t state;
    curand_init(rnd_seed, tidx, local_offset, &state);
    
    BoundedFloat2Generator generator{lb, ub};
    
    // Grid-stride loop: each thread processes multiple float2 elements
    for (int linear_idx = tidx * 2; linear_idx < rounded_size; linear_idx += STRIDE * 2) {
        float2 val1, val2;
        generator.generate2_float2(state, val1, val2);
        
        if (linear_idx < count) {
            rnd_output[linear_idx] = val1;
        }
        if (linear_idx + 1 < count) {
            rnd_output[linear_idx + 1] = val2;
        }
    }
}

// ================================================================================
// Unified Launch Helper Definition
// ================================================================================
// ================================================================================================================ //
/**
 * @brief Unified helper for launching random number generation kernels
 * @param kernel_name Name of the kernel to launch
 * @param d_output Device output array
 * @param count Number of elements to generate
 * @param glb_rnd_offset Pointer to global random offset
 * @param block_size Number of threads per block
 * @param stream CUDA stream for kernel execution
 * @param ... Additional kernel parameters (variadic)
 * 
 * This helper automatically:
 * 1. Calculates optimal execution policy
 * 2. Selects appropriate kernel variant (simple vs grid-stride)
 * 3. Chooses GPU-optimized template parameters
 * 4. Handles error checking and synchronization
 * 5. Updates the global random offset
 */
// ================================================================================================================ //
#define LAUNCH_RANDOM_KERNEL(kernel_name, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream, ...) \
do { \
    /* Calculate optimal execution policy for this workload */ \
    auto [counter_offset, grid, block, use_grid_stride, stride, rounded_size] = calc_execution_policy(count, block_size); \
    \
    if (use_grid_stride) { \
        /* Large dataset: use grid-stride kernel with GPU-specific optimizations */ \
        switch(stride) { \
            case gpu::models::rtx4060::MAX_THREADS: \
                /* Launch optimized kernel for RTX 4060 */ \
                kernel_name<gpu::models::rtx4060::MAX_THREADS, unroll_factor::UNROLL_FACTOR><<<grid, block, 0, stream>>>( \
                    d_output, count, rounded_size, ##__VA_ARGS__, rnd_seed, glb_rnd_offset); \
                CUDA_CHECK(cudaGetLastError()); \
                break; \
            case gpu::models::rtx4090::MAX_THREADS: \
                /* Launch optimized kernel for RTX 4090 */ \
                kernel_name<gpu::models::rtx4090::MAX_THREADS, unroll_factor::UNROLL_FACTOR><<<grid, block, 0, stream>>>( \
                    d_output, count, rounded_size, ##__VA_ARGS__, rnd_seed, glb_rnd_offset); \
                CUDA_CHECK(cudaGetLastError()); \
                break; \
            case gpu::models::rtx5090::MAX_THREADS: \
                /* Launch optimized kernel for RTX 5090 */ \
                kernel_name<gpu::models::rtx5090::MAX_THREADS, unroll_factor::UNROLL_FACTOR><<<grid, block, 0, stream>>>( \
                    d_output, count, rounded_size, ##__VA_ARGS__, rnd_seed, glb_rnd_offset); \
                CUDA_CHECK(cudaGetLastError()); \
                break; \
            case gpu::models::rtx6000pro::MAX_THREADS: \
                /* Launch optimized kernel for RTX 6000 Pro */ \
                kernel_name<gpu::models::rtx6000pro::MAX_THREADS, unroll_factor::UNROLL_FACTOR><<<grid, block, 0, stream>>>( \
                    d_output, count, rounded_size, ##__VA_ARGS__, rnd_seed, glb_rnd_offset); \
                CUDA_CHECK(cudaGetLastError()); \
                break; \
            default: \
                /* GPU not in optimized list - provide helpful information */ \
                printf("[INFO] Current GPU stride=%d not in pre-optimized list\n", stride); \
                printf("[SUGGESTION] Consider adding 'constexpr int STRIDE_CURRENT_GPU = %d;' for optimal performance\n", stride); \
                printf("[INFO] Using runtime stride for now...\n"); \
                break; \
        } \
    } else { \
        /* Small dataset: use simple kernel without grid-stride loops */ \
        kernel_name<<<grid, block, 0, stream>>>(d_output, count, ##__VA_ARGS__, rnd_seed, glb_rnd_offset); \
        CUDA_CHECK(cudaGetLastError()); \
    } \
    /* Update global random offset to maintain reproducibility across kernel calls */ \
    glb_rnd_offset += counter_offset; \
} while(0)
// ================================================================================================================ //
/**
 * @brief Specialized helper for launching float2 random number generation kernels
 * This helper handles the special case where we process 2 float2s per thread
 */
#define LAUNCH_RANDOM_FLOAT2_KERNEL(kernel_name, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream) \
do { \
    /* For float2, we need to adjust the execution policy */ \
    /* Each thread processes 2 float2s to fully utilize curand4 */ \
    dim3 dim_block(block_size); \
    dim3 grid((count + block_size * 2 - 1) / (block_size * 2)); \
    \
    uint32_t fully_occupied_num_threads = get_max_occupied_threads(); \
    uint32_t fully_occupied_num_blocks = fully_occupied_num_threads / block_size; \
    grid.x = std::min(fully_occupied_num_blocks, grid.x); \
    \
    int stride = grid.x * block_size; \
    bool use_grid_stride = (count > stride * 2); \
    \
    if (use_grid_stride) { \
        /* Calculate rounded size for float2 alignment */ \
        int rounded_size = ((count - 1) / (stride * 2) + 1) * stride * 2; \
        ull counter_offset = ((count - 1) / (stride * 2) + 1) * 4; /* 4 values per curand4 */ \
        \
        switch(stride) { \
            case gpu::models::rtx4060::MAX_THREADS: \
                kernel_name<gpu::models::rtx4060::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx4090::MAX_THREADS: \
                kernel_name<gpu::models::rtx4090::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx5090::MAX_THREADS: \
                kernel_name<gpu::models::rtx5090::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx6000pro::MAX_THREADS: \
                kernel_name<gpu::models::rtx6000pro::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            default: \
                printf("[INFO] Current GPU stride=%d not in pre-optimized list for float2\n", stride); \
                printf("[INFO] Falling back to non-templated version\n"); \
                /* For fallback, we need to compute stride dynamically in kernel */ \
                return; \
        } \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += counter_offset; \
    } else { \
        /* Small dataset: use simple kernel */ \
        dim3 simple_grid((count + block_size - 1) / block_size); \
        kernel_name<<<simple_grid, dim_block, 0, stream>>>( \
            d_output, count, rnd_seed, glb_rnd_offset); \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += count * 2; /* 2 values per float2 */ \
    } \
} while(0)
// ================================================================================================================ //
/**
 * @brief Specialized helper for launching float4 random number generation kernels
 * 
 * This helper handles the special case where each thread generates one float4 (4 values).
 * Unlike float2, float4 maps directly to curand_uniform4, achieving 100% efficiency
 * without requiring special packing logic.
 * 
 * Memory access pattern: Each thread reads/writes one float4 (16 bytes), which is
 * optimal for GPU memory coalescing on modern NVIDIA architectures.
 */
#define LAUNCH_RANDOM_FLOAT4_KERNEL(kernel_name, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream) \
do { \
    /* For float4, each thread generates exactly one float4 (4 random values) */ \
    /* This maps directly to curand_uniform4 for 100% efficiency */ \
    dim3 dim_block(block_size); \
    dim3 grid((count + block_size - 1) / block_size); \
    \
    uint32_t fully_occupied_num_threads = get_max_occupied_threads(); \
    uint32_t fully_occupied_num_blocks = fully_occupied_num_threads / block_size; \
    grid.x = std::min(fully_occupied_num_blocks, grid.x); \
    \
    int stride = grid.x * block_size; \
    bool use_grid_stride = (count > stride); \
    \
    if (use_grid_stride) { \
        /* Calculate rounded size for float4 alignment */ \
        /* UNROLL_FACTOR float4s per iteration = UNROLL_FACTOR * 4 random values */ \
        int rounded_size = cal_rounded_size<unroll_factor::UNROLL_FACTOR>(count, stride); \
        ull counter_offset = ((count - 1) / (stride * unroll_factor::UNROLL_FACTOR) + 1) \
                           * unroll_factor::UNROLL_FACTOR * 4; /* 4 values per float4 */ \
        \
        switch(stride) { \
            case gpu::models::rtx4060::MAX_THREADS: \
                kernel_name<gpu::models::rtx4060::MAX_THREADS, unroll_factor::UNROLL_FACTOR> \
                    <<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx4090::MAX_THREADS: \
                kernel_name<gpu::models::rtx4090::MAX_THREADS, unroll_factor::UNROLL_FACTOR> \
                    <<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx5090::MAX_THREADS: \
                kernel_name<gpu::models::rtx5090::MAX_THREADS, unroll_factor::UNROLL_FACTOR> \
                    <<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx6000pro::MAX_THREADS: \
                kernel_name<gpu::models::rtx6000pro::MAX_THREADS, unroll_factor::UNROLL_FACTOR> \
                    <<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, rnd_seed, glb_rnd_offset); \
                break; \
            default: \
                printf("[INFO] Current GPU stride=%d not in pre-optimized list for float4\n", stride); \
                printf("[INFO] Falling back to non-templated version\n"); \
                return; \
        } \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += counter_offset; \
    } else { \
        /* Small dataset: use simple kernel, each thread generates 1 float4 */ \
        kernel_name<<<grid, dim_block, 0, stream>>>( \
            d_output, count, rnd_seed, glb_rnd_offset); \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += count * 4; /* 4 values per float4 */ \
    } \
} while(0)
// ================================================================================================================ //
/**
 * @brief Specialized helper for launching int2 random number generation kernels
 * This helper handles the special case where we process 2 int2s per thread
 */
#define LAUNCH_RANDOM_INT2_KERNEL(kernel_name, d_output, count, lb, ub, rnd_seed, glb_rnd_offset, block_size, stream) \
do { \
    /* For int2, we need to adjust the execution policy */ \
    /* Each thread processes 2 int2s to fully utilize curand4 */ \
    dim3 dim_block(block_size); \
    dim3 grid((count + block_size * 2 - 1) / (block_size * 2)); \
    \
    uint32_t fully_occupied_num_threads = get_max_occupied_threads(); \
    uint32_t fully_occupied_num_blocks = fully_occupied_num_threads / block_size; \
    grid.x = std::min(fully_occupied_num_blocks, grid.x); \
    \
    int stride = grid.x * block_size; \
    bool use_grid_stride = (count > stride * 2); \
    \
    if (use_grid_stride) { \
        /* Calculate rounded size for int2 alignment */ \
        int rounded_size = ((count - 1) / (stride * 2) + 1) * stride * 2; \
        ull counter_offset = ((count - 1) / (stride * 2) + 1) * 4; /* 4 values per curand4 */ \
        \
        switch(stride) { \
            case gpu::models::rtx4060::MAX_THREADS: \
                kernel_name<gpu::models::rtx4060::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx4090::MAX_THREADS: \
                kernel_name<gpu::models::rtx4090::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx5090::MAX_THREADS: \
                kernel_name<gpu::models::rtx5090::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx6000pro::MAX_THREADS: \
                kernel_name<gpu::models::rtx6000pro::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            default: \
                printf("[INFO] Current GPU stride=%d not in pre-optimized list for int2\n", stride); \
                printf("[INFO] Falling back to non-templated version\n"); \
                /* For fallback, we need to compute stride dynamically in kernel */ \
                return; \
        } \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += counter_offset; \
    } else { \
        /* Small dataset: use simple kernel */ \
        dim3 simple_grid((count + block_size - 1) / block_size); \
        kernel_name<<<simple_grid, dim_block, 0, stream>>>( \
            d_output, count, lb, ub, rnd_seed, glb_rnd_offset); \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += count * 2; /* 2 values per int2 */ \
    } \
} while(0)

// ================================================================================================================ //
/**
 * @brief Specialized helper for launching bounded float2 random number generation kernels
 * This helper handles the special case where we process 2 float2s per thread with custom bounds
 */
#define LAUNCH_RANDOM_BOUNDED_FLOAT2_KERNEL(kernel_name, d_output, count, lb, ub, rnd_seed, glb_rnd_offset, block_size, stream) \
do { \
    /* For bounded float2, we need to adjust the execution policy */ \
    /* Each thread processes 2 float2s to fully utilize curand4 */ \
    dim3 dim_block(block_size); \
    dim3 grid((count + block_size * 2 - 1) / (block_size * 2)); \
    \
    uint32_t fully_occupied_num_threads = get_max_occupied_threads(); \
    uint32_t fully_occupied_num_blocks = fully_occupied_num_threads / block_size; \
    grid.x = std::min(fully_occupied_num_blocks, grid.x); \
    \
    int stride = grid.x * block_size; \
    bool use_grid_stride = (count > stride * 2); \
    \
    if (use_grid_stride) { \
        /* Calculate rounded size for bounded float2 alignment */ \
        int rounded_size = ((count - 1) / (stride * 2) + 1) * stride * 2; \
        ull counter_offset = ((count - 1) / (stride * 2) + 1) * 4; /* 4 values per curand4 */ \
        \
        switch(stride) { \
            case gpu::models::rtx4060::MAX_THREADS: \
                kernel_name<gpu::models::rtx4060::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx4090::MAX_THREADS: \
                kernel_name<gpu::models::rtx4090::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx5090::MAX_THREADS: \
                kernel_name<gpu::models::rtx5090::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            case gpu::models::rtx6000pro::MAX_THREADS: \
                kernel_name<gpu::models::rtx6000pro::MAX_THREADS, 2><<<grid, dim_block, 0, stream>>>( \
                    d_output, count, rounded_size, lb, ub, rnd_seed, glb_rnd_offset); \
                break; \
            default: \
                printf("[INFO] Current GPU stride=%d not in pre-optimized list for bounded float2\n", stride); \
                printf("[INFO] Falling back to non-templated version\n"); \
                /* For fallback, we need to compute stride dynamically in kernel */ \
                return; \
        } \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += counter_offset; \
    } else { \
        /* Small dataset: use simple kernel */ \
        dim3 simple_grid((count + block_size - 1) / block_size); \
        kernel_name<<<simple_grid, dim_block, 0, stream>>>( \
            d_output, count, lb, ub, rnd_seed, glb_rnd_offset); \
        CUDA_CHECK(cudaGetLastError()); \
        glb_rnd_offset += count * 2; /* 2 values per float2 */ \
    } \
} while(0)
// ================================================================================
// Launch Function Implementations
// ================================================================================
// ================================================================================================================ //
/**
 * @brief Launch function for uniform random number generation
 * @param d_output Device array to store uniform random floats
 * @param count Number of random numbers to generate
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 * 
 * This function provides a high-level interface for generating uniform random
 * numbers, automatically selecting the optimal kernel configuration.
 */
void launch_random_uniform_kernel(
    float* d_output,                  
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset, 
    const int block_size,
    cudaStream_t stream) 
{   
    LAUNCH_RANDOM_KERNEL(random_uniform_kernel, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream);
}
// ================================================================================================================ //
/**
 * @brief Launch function for single float random number generation (alias for uniform)
 * @param d_output Device array to store random floats in [0, 1)
 * @param count Number of random numbers to generate
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 * 
 * This function is an alias for launch_random_uniform_kernel, providing consistent
 * naming convention alongside launch_random_float2_kernel.
 * Generates float values in range [0, 1).
 */
void launch_random_float_kernel(
    float* d_output,                  
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset, 
    const int block_size,
    cudaStream_t stream) 
{   
    // Delegate to uniform kernel - same underlying implementation
    LAUNCH_RANDOM_KERNEL(random_uniform_kernel, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream);
}
// ================================================================================================================ //
/**
 * @brief Launch function for boolean random number generation
 * @param d_output Device array to store random boolean values
 * @param count Number of random booleans to generate
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 */
void launch_random_boolean_kernel(
    int* d_output,
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream) 
{
    LAUNCH_RANDOM_KERNEL(random_boolean_kernel, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream);
}
// ================================================================================================================ //
/**
 * @brief Launch function for integer range random number generation
 * @param d_output Device array to store random integers
 * @param count Number of random integers to generate
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 */
void launch_random_int_kernel(
    int* d_output,
    int count,
    int lb,
    int ub,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream) 
{
    LAUNCH_RANDOM_KERNEL(random_int_kernel, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream, lb, ub);
}
// ================================================================================================================ //
/**
 * @brief Launch function for packed float2 random number generation
 * @param d_output Device array to store packed float2 random values
 * @param count Number of float2 elements to generate
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 * 
 * This function provides optimal memory bandwidth usage by packing two random
 * values into a single float2, reducing memory transactions by 50%
 */
void launch_random_float2_kernel(
    float2* d_output,
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream) 
{
    LAUNCH_RANDOM_FLOAT2_KERNEL(random_float2_kernel, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream);
}
// ================================================================================================================ //
/**
 * @brief Launch function for packed float4 random number generation
 * @param d_output Device array to store packed float4 random values
 * @param count Number of float4 elements to generate
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 * 
 * This function provides maximum memory bandwidth efficiency by packing four random
 * values into a single float4 (16 bytes). This is particularly useful for operations
 * requiring multiple random values per work item, such as:
 * - Polynomial mutation (decision + mu_x + mu_y + reserved)
 * - Complex crossover operations
 * - Monte Carlo simulations
 * 
 * The implementation directly maps to curand_uniform4 for 100% hardware utilization.
 * 
 * Memory layout: Each float4 contains {.x, .y, .z, .w} all in range [0, 1)
 */
void launch_random_float4_kernel(
    float4* d_output,
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream) 
{
    LAUNCH_RANDOM_FLOAT4_KERNEL(random_float4_kernel, d_output, count, rnd_seed, glb_rnd_offset, block_size, stream);
}
// ================================================================================================================ //
/**
 * @brief Launch function for packed int2 random number generation
 * @param d_output Device array to store packed int2 random values
 * @param count Number of int2 elements to generate
 * @param lb Lower bound (inclusive, default: INT_MIN)
 * @param ub Upper bound (exclusive, default: INT_MAX)
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 * 
 * This function provides optimal memory bandwidth usage by packing two random
 * integer values into a single int2, reducing memory transactions by 50%
 */
void launch_random_int2_kernel(
    int2* d_output,
    int count,
    int lb,
    int ub,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream) 
{
    LAUNCH_RANDOM_INT2_KERNEL(random_int2_kernel, d_output, count, lb, ub, rnd_seed, glb_rnd_offset, block_size, stream);
}

// ================================================================================================================ //
/**
 * @brief Launch function for bounded uniform random number generation
 * @param d_output Device array to store uniform random floats in [lb, ub)
 * @param count Number of random numbers to generate
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 */
void launch_random_bounded_uniform_kernel(
    float* d_output,
    int count,
    float lb,
    float ub,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream
) 
{
    LAUNCH_RANDOM_KERNEL(random_bounded_uniform_kernel, d_output, count, 
                         rnd_seed, glb_rnd_offset, block_size, stream, lb, ub);
}
// ================================================================================================================ //
/**
 * @brief Launch function for bounded packed float2 random number generation
 * @param d_output Device array to store packed float2 random values in [lb, ub)
 * @param count Number of float2 elements to generate
 * @param lb Lower bound (inclusive)
 * @param ub Upper bound (exclusive)
 * @param glb_rnd_offset Reference to global random offset pointer
 * @param block_size Number of threads per block (default: 256)
 * @param stream CUDA stream for kernel execution (default: 0)
 */
void launch_random_bounded_float2_kernel(
    float2* d_output,
    int count,
    float lb,
    float ub,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream) 
{
    LAUNCH_RANDOM_BOUNDED_FLOAT2_KERNEL(random_bounded_float2_kernel, d_output, count, 
                                        lb, ub, rnd_seed, glb_rnd_offset, block_size, stream);
}


// ========================================================================================================================= //

// =============================================================================================== //
// Philox-style counter-based RNG for uint32 generation
// Uses the same mixing constants as SplitMix64 for high-quality randomness
// =============================================================================================== //

__device__ __forceinline__ unsigned int philox_uint32(ull counter, ull rnd_seed) {
    // SplitMix64-style mixing - fast and statistically excellent
    ull z = counter ^ rnd_seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<unsigned int>(z);
}

__device__ __forceinline__ uint2 philox_uint2(ull counter, ull rnd_seed) {
    // Generate two uint32 values from one counter
    ull z = counter ^ rnd_seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    
    // Split 64-bit result into two 32-bit values
    return make_uint2(
        static_cast<unsigned int>(z),
        static_cast<unsigned int>(z >> 32)
    );
}

// =============================================================================================== //
// Kernel: Generate uint32 random values
// =============================================================================================== //
__global__ void random_uint32_kernel(
    unsigned int* __restrict__ d_output,  // output: (count,)
    ull base_offset,                       // input:  scalar - starting counter value
    ull rnd_seed,                          // input:  scalar - random seed
    int count                              // input:  scalar - number of values
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= count) return;
    
    d_output[tidx] = philox_uint32(base_offset + tidx, rnd_seed);
}

// =============================================================================================== //
// Kernel: Generate uint2 random values (vectorized)
// =============================================================================================== //
__global__ void random_uint2_kernel(
    uint2* __restrict__ d_output,  // output: (count,) as uint2
    ull base_offset,                // input:  scalar - starting counter value
    ull rnd_seed,                   // input:  scalar - random seed
    int count                       // input:  scalar - number of uint2 pairs
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= count) return;
    
    // Each thread generates 2 uint32 values
    d_output[tidx] = philox_uint2(base_offset + tidx * 2ULL, rnd_seed);
}

// =============================================================================================== //
// Launcher: uint32 random values
// =============================================================================================== //
void launch_random_uint32_kernel(
    unsigned int* d_output,
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream)
{
    if (count <= 0) return;
    
    const ull base_offset = glb_rnd_offset;
    glb_rnd_offset += static_cast<ull>(count);
    
    const int grid_size = (count + block_size - 1) / block_size;
    random_uint32_kernel<<<grid_size, block_size, 0, stream>>>(d_output, base_offset, rnd_seed, count);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================================== //
// Launcher: uint2 random values (vectorized)
// =============================================================================================== //
void launch_random_uint2_kernel(
    uint2* d_output,
    int count,
    ull  rnd_seed,
    ull&  glb_rnd_offset,
    const int block_size,
    cudaStream_t stream)
{
    if (count <= 0) return;
    
    const ull base_offset = glb_rnd_offset;
    glb_rnd_offset += static_cast<ull>(count) * 2ULL;  // Each uint2 consumes 2 counters
    
    const int grid_size = (count + block_size - 1) / block_size;
    random_uint2_kernel<<<grid_size, block_size, 0, stream>>>(d_output, base_offset, rnd_seed, count);
    CUDA_CHECK(cudaGetLastError());
}
