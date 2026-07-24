#include <cusolverDn.h>
#include <math_constants.h>

#include <mutex>

#include "cuda_moea/core/cuda/cuda_globals.cuh"
#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"

// =================================================================================== //
//                          UNIFIED THRESHOLD CONSTANTS
// =================================================================================== //
/**
 * @brief Unified threshold constants for NSGA-III normalization
 *
 * Design Philosophy:
 * - SVD-related thresholds use a relative threshold system for scale-invariance (rescale version)
 * - Legacy version uses absolute threshold for backward compatibility
 * - Range protection uses a separate absolute threshold (pure divide-by-zero guard)
 *
 * Condition Number Criterion (rescale version):
 *     ill-conditioned if: σ_min <= τ_abs OR σ_min <= τ_rel * σ_max
 *     equivalently:       κ₂(A) >= 1/τ_rel  OR  σ_max <= τ_abs
 *
 * For float32 with ~7 significant digits:
 * - τ_rel = 1e-6 means κ₂ <= 1e6, preserving ~1 significant digit (marginal)
 * - τ_rel = 1e-5 means κ₂ <= 1e5, preserving ~2 significant digits (safer)
 */
// =================================================================================== //
namespace NormConstants {
    // ============================================================
    // SVD-related thresholds (relative threshold system - rescale version)
    // ============================================================

    // Relative threshold: σ_min / σ_max must exceed this
    // Equivalent to condition number threshold of 1/TAU_REL = 1e6
    constexpr float TAU_REL = 1e-6f;

    // Absolute floor: handles edge case where σ_max itself is near zero
    // (e.g., all objectives converged to ideal point)
    constexpr float TAU_ABS = 1e-10f;

    // ============================================================
    // Legacy thresholds (non-scale version - backward compatibility)
    // ============================================================

    // Legacy condition number threshold (absolute)
    // [FIX #11] Reduced from 1e7 to 1e6 for better numerical safety with float32
    // Rationale: float32 has ~7 significant digits, κ=1e6 preserves ~1 digit (marginal but safer)
    constexpr float LEGACY_COND_THRESHOLD = 1e6f;

    // Legacy epsilon for numerical stability
    constexpr float LEGACY_EPSILON = 1e-10f;

    // Legacy pinv threshold
    constexpr float LEGACY_PINV_THRESHOLD = 1e-5f;

    // ============================================================
    // Range protection threshold (shared by both versions)
    // ============================================================

    // Minimum absolute range to prevent division by zero in ASF normalization
    // Changed from 1e-30f to 1e-6f for better numerical stability
    // Rationale: 1e-30 could cause extreme amplification (1/1e-30 = 1e30)
    constexpr float RANGE_MIN_ABS = 1e-6f;

    // Relative range floor: range must be at least this fraction of row_max
    // Prevents extreme scaling when range is tiny relative to values
    constexpr float RANGE_MIN_REL = 1e-6f;

    // ============================================================
    // Fallback intercept threshold
    // ============================================================

    // When row-max falls below this, use intercept = 1.0 (no normalization)
    // Semantics: objective has converged, no need to normalize
    constexpr float INTERCEPT_MIN = 1e-10f;

    // ============================================================
    // NaN Handling Constants (Unified Strategy)
    // ============================================================
    // [FIX #4] Unified NaN handling: NaN is treated as "worst possible value"
    // - For minimization (ASF): NaN -> +INF (infinitely bad)
    // - For maximization (range): NaN -> -INF (excluded from max)
    // This ensures NaN individuals are effectively excluded from selection

    // ============================================================
    // [FIX F] ASF Penalty Constant (Centralized)
    // ============================================================
    // ASF penalty for non-target objectives
    // Centralized here for easy tuning and consistency across kernels
    // Note: Very large values may cause INF when multiplied with large fitness values
    constexpr float ASF_PENALTY = 1e6f;
}

// =================================================================================== //
//                          ERROR CODE DEFINITIONS
// =================================================================================== //
/**
 * @brief Error codes for normalization pipeline
 * [FIX #6] Improved error handling with explicit return codes
 */
// =================================================================================== //
namespace NormError {
    constexpr int SUCCESS = 0;
    constexpr int UNSUPPORTED_M = -1;        // M > 8 not supported
    // constexpr int SVD_NOT_CONVERGED = -2;    // SVD failed to converge
    // constexpr int CUDA_ERROR = -3;           // Generic CUDA error
}

// =================================================================================== //
//                     SCOPED DEVICE GUARD (RAII)
// =================================================================================== //
/**
 * @brief RAII guard for temporarily switching CUDA device
 * 
 * [FIX H] Required for correct multi-GPU handle management
 * Saves current device on construction, restores on destruction.
 * Used in HandlePool to ensure operations happen on the correct device.
 */
// =================================================================================== //
class ScopedDevice {
public:
    explicit ScopedDevice(int target_device) : prev_device_(-1), need_restore_(false) {
        cudaError_t err = cudaGetDevice(&prev_device_);
        if (err != cudaSuccess) {
            prev_device_ = -1;
            return;
        }
        if (prev_device_ != target_device) {
            err = cudaSetDevice(target_device);
            if (err == cudaSuccess) {
                need_restore_ = true;
            }
        }
    }
    
    ~ScopedDevice() {
        if (need_restore_ && prev_device_ >= 0) {
            // Restore previous device, ignore errors during destruction
            cudaSetDevice(prev_device_);
        }
    }
    
    // Non-copyable
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    
private:
    int prev_device_;
    bool need_restore_;
};

// =================================================================================== //
//                     PER-STREAM HANDLE POOL (THREAD-SAFE)
// =================================================================================== //
/**
 * @brief Thread-safe handle pool for cuBLAS and cuSOLVER
 *
 * Problem Addressed:
 * - Global singleton handles with SetStream() cause race conditions in multi-stream scenarios
 * - Stream A calls setStream(A) -> Stream B calls setStream(B) -> Stream A launches kernel
 *   -> Kernel actually runs on Stream B (TOCTOU race)
 *
 * Solution:
 * - Each (device, cudaStream_t) pair gets its own dedicated handle
 * - Handle is created on first access and cached for reuse
 * - Mutex protects the map during lookup/insertion (not during actual CUDA calls)
 *
 * [FIX #2] ABA Problem Mitigation:
 * - Before returning cached handle, verify stream is still valid via cudaStreamQuery
 * - If stream was destroyed and recreated with same pointer, detect and recreate handle
 *
 * [FIX D] Multi-GPU Support:
 * - Key now includes device ID to prevent cross-device handle misuse
 * - cuBLAS/cuSOLVER handles are device-specific and cannot be used on different devices
 *
 * [FIX H] Correct device context for all operations:
 * - cleanup/destructor now switches to correct device before destroying handles
 * - cudaStreamQuery is called on the correct device
 *
 * [FIX K] Pointer mode explicitly set:
 * - cuBLAS handles have pointer mode set to HOST on creation
 * - Prevents issues if handle is accidentally modified elsewhere
 *
 * Usage:
 *   cublasHandle_t h = HandlePool::instance().get_cublas(my_stream);
 *   // h is now permanently bound to (current_device, my_stream), no SetStream needed
 */
// =================================================================================== //

// [FIX D] Key structure for handle pool - includes device ID for multi-GPU support
struct StreamKey {
    int device;
    cudaStream_t stream;
    
    bool operator==(const StreamKey& other) const {
        return device == other.device && stream == other.stream;
    }
};

// [FIX D] Hash function for StreamKey
struct StreamKeyHash {
    size_t operator()(const StreamKey& key) const {
        // Combine device ID and stream pointer into a single hash
        // Using a prime multiplier for better hash distribution
        size_t stream_hash = std::hash<void*>{}(reinterpret_cast<void*>(key.stream));
        size_t device_hash = static_cast<size_t>(key.device) * 1315423911u;
        return stream_hash ^ device_hash;
    }
};

class HandlePool {
public:
    static HandlePool& instance() {
        static HandlePool pool;
        return pool;
    }

    cublasHandle_t get_cublas(
        cudaStream_t stream               // input: scalar - CUDA stream to bind this cuBLAS handle to
    ) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // [FIX D] Get current device ID and use (device, stream) as key
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        StreamKey key{device, stream};
        
        auto it = cublas_handles_.find(key);
        if (it != cublas_handles_.end()) {
            // [FIX #2] Verify stream is still valid (ABA problem mitigation)
            // [FIX H] Stream query must happen on the correct device
            cudaError_t status = cudaStreamQuery(stream);
            if (status != cudaErrorInvalidResourceHandle) {
                // Stream is valid (either idle or busy), return cached handle
                return it->second;
            }
            // Stream was destroyed and pointer reused - destroy old handle
            cublasDestroy(it->second);
            cublas_handles_.erase(it);
        }
        
        // Create new handle for this (device, stream) pair
        cublasHandle_t handle{};
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        
        // [FIX K] Explicitly set pointer mode to HOST to prevent issues
        // if handle is accidentally modified elsewhere
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        
        cublas_handles_[key] = handle;
        return handle;
    }

    cusolverDnHandle_t get_cusolver(
        cudaStream_t stream               // input: scalar - CUDA stream to bind this cuSOLVER handle to
    ) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // [FIX D] Get current device ID and use (device, stream) as key
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        StreamKey key{device, stream};
        
        auto it = cusolver_handles_.find(key);
        if (it != cusolver_handles_.end()) {
            // [FIX #2] Verify stream is still valid (ABA problem mitigation)
            // [FIX H] Stream query must happen on the correct device
            cudaError_t status = cudaStreamQuery(stream);
            if (status != cudaErrorInvalidResourceHandle) {
                // Stream is valid (either idle or busy), return cached handle
                return it->second;
            }
            // Stream was destroyed and pointer reused - destroy old handle
            cusolverDnDestroy(it->second);
            cusolver_handles_.erase(it);
        }
        
        // Create new handle for this (device, stream) pair
        cusolverDnHandle_t handle{};
        CUSOLVER_CHECK(cusolverDnCreate(&handle));
        CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
        cusolver_handles_[key] = handle;
        return handle;
    }

    /**
     * @brief Explicit cleanup interface for safe shutdown
     * [FIX #10] Call this before CUDA context destruction to avoid errors
     * [FIX H] Now correctly switches device before destroying handles
     */
    void cleanup() {
        std::lock_guard<std::mutex> lock(mtx_);
        
        for (auto& pair : cublas_handles_) {
            // [FIX H] Switch to correct device before destroying handle
            ScopedDevice guard(pair.first.device);
            cublasDestroy(pair.second);
        }
        cublas_handles_.clear();
        
        for (auto& pair : cusolver_handles_) {
            // [FIX H] Switch to correct device before destroying handle
            ScopedDevice guard(pair.first.device);
            cusolverDnDestroy(pair.second);
        }
        cusolver_handles_.clear();
    }

    ~HandlePool() {
        // [FIX #10] Safe destruction with error suppression
        // [FIX H] Now correctly switches device before operations
        // Note: If CUDA context is already destroyed, these calls may fail silently
        // For guaranteed safe cleanup, call cleanup() explicitly before context destruction
        std::lock_guard<std::mutex> lock(mtx_);
        
        for (auto& pair : cublas_handles_) {
            // [FIX H] Switch to correct device before query/destroy
            ScopedDevice guard(pair.first.device);
            
            // Check if CUDA runtime is still available before destroying
            cudaError_t status = cudaStreamQuery(pair.first.stream);
            if (status != cudaErrorInvalidResourceHandle && 
                status != cudaErrorCudartUnloading) {
                cublasDestroy(pair.second);
            }
        }
        
        for (auto& pair : cusolver_handles_) {
            // [FIX H] Switch to correct device before query/destroy
            ScopedDevice guard(pair.first.device);
            
            cudaError_t status = cudaStreamQuery(pair.first.stream);
            if (status != cudaErrorInvalidResourceHandle && 
                status != cudaErrorCudartUnloading) {
                cusolverDnDestroy(pair.second);
            }
        }
    }

private:
    HandlePool() = default;
    HandlePool(const HandlePool&) = delete;
    HandlePool& operator=(const HandlePool&) = delete;

    std::mutex mtx_;
    // [FIX D] Changed key type from cudaStream_t to StreamKey for multi-GPU support
    std::unordered_map<StreamKey, cublasHandle_t, StreamKeyHash>     cublas_handles_;
    std::unordered_map<StreamKey, cusolverDnHandle_t, StreamKeyHash> cusolver_handles_;
};

void cleanup_normalization_handle_pool() {
    HandlePool::instance().cleanup();
}

// =================================================================================== //
//                          SHARED KERNELS (BOTH VERSIONS)
// =================================================================================== //

// =================================================================================== //
/**
 * @brief CUDA Kernel: Translation - Subtract ideal points from population fitness (Scalar Version)
 * 
 * [FIX #4] Unified NaN handling: NaN values are preserved (subtraction with NaN = NaN)
 *          Downstream kernels will handle NaN appropriately
 * [FIX J] Clamp shifted values to >= 0 to prevent negative ASF values
 */
// =================================================================================== //
__global__ void translated_ndsfit_kernel(
    float* __restrict__ d_ndsfv,          // update: (M, N_nds) - fitness after NDS (shifted by ideal points after update)
    const float* __restrict__ d_ipt,      // input: (M,) - ideal points
    int M,                                // input: scalar - number of objectives
    int N_nds                             // input: scalar - N_prior + N_last
) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N_nds) return;

    const float ideal = d_ipt[row];
    const int idx = row * N_nds + col;
    float val = d_ndsfv[idx] - ideal;
    
    // [FIX J] Clamp to non-negative to prevent negative ASF values
    // Negative values can occur due to:
    // 1. Historical ideal points (cross-generation accumulation)
    // 2. Numerical errors in float32 arithmetic
    // Only clamp finite values; preserve NaN for downstream handling
    if (isfinite(val)) {
        val = fmaxf(val, 0.0f);
    }
    
    d_ndsfv[idx] = val;
}

// =================================================================================== //
/**
 * @brief CUDA Kernel: Translation with float4 Vectorization (Alignment-Safe Version)
 * 
 * [FIX #4] Unified NaN handling: NaN values are preserved through subtraction
 * [FIX J] Clamp shifted values to >= 0 to prevent negative ASF values
 */
// =================================================================================== //
__global__ void translated_ndsfit_vec_kernel(
    float* __restrict__ d_ndsfv,          // update: (M, N_nds) - fitness after NDS (shifted by ideal points after update)
    const float* __restrict__ d_ipt,      // input: (M,) - ideal points
    int M,                                // input: scalar - number of objectives
    int N_nds                             // input: scalar - N_prior + N_last (must be multiple of 4 for best vectorization)
) {
    const int row = blockIdx.y;
    const int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (row >= M) return;

    const float ideal = d_ipt[row];
    const int offset = row * N_nds + col_base;

    if (col_base + 3 < N_nds) {
        float4 vals = *reinterpret_cast<float4*>(&d_ndsfv[offset]);
        vals.x -= ideal;
        vals.y -= ideal;
        vals.z -= ideal;
        vals.w -= ideal;
        
        // [FIX J] Clamp to non-negative (only finite values)
        if (isfinite(vals.x)) vals.x = fmaxf(vals.x, 0.0f);
        if (isfinite(vals.y)) vals.y = fmaxf(vals.y, 0.0f);
        if (isfinite(vals.z)) vals.z = fmaxf(vals.z, 0.0f);
        if (isfinite(vals.w)) vals.w = fmaxf(vals.w, 0.0f);
        
        *reinterpret_cast<float4*>(&d_ndsfv[offset]) = vals;
    } else {
        #pragma unroll
        for (int i = 0; i < 4 && col_base + i < N_nds; ++i) {
            float val = d_ndsfv[offset + i] - ideal;
            // [FIX J] Clamp to non-negative (only finite values)
            if (isfinite(val)) {
                val = fmaxf(val, 0.0f);
            }
            d_ndsfv[offset + i] = val;
        }
    }
}

// =================================================================================== //
/**
 * @brief CUDA Kernel: Stage 2 - Global Reduction for Extreme Points
 * 
 * [FIX #4] NaN handling: Block-level NaN values (from phase 1) are treated as +INF
 */
// =================================================================================== //
template <const int WPB>
__global__ void find_expts_phase2_kernel(
    const float* __restrict__ d_expts_blk_min,   // input: (num_blks_nds, M) - block-level minimum ASF values
    const int*   __restrict__ d_expts_blk_amin,  // input: (num_blks_nds, M) - block-level argmin indices
    int* d_expts_idx,                            // output: (M,) - global extreme point indices
    int num_blks_nds,                            // input: scalar - number of blocks from Stage 1
    int M                                        // input: scalar - number of objectives
) {
    __shared__ float s_min_val[WPB];
    __shared__ int   s_min_idx[WPB];

    const int obj_idx = blockIdx.x;
    const int tidx    = threadIdx.x;
    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    float local_min_val = CUDART_INF_F;
    int   local_min_idx = -1;

    for (int bx = tidx; bx < num_blks_nds; bx += blockDim.x) {
        const int idx = bx * M + obj_idx;
        float val = d_expts_blk_min[idx];
        const int idx_val = d_expts_blk_amin[idx];

        // [FIX #4] Unified NaN handling: treat NaN as infinitely bad (excluded from selection)
        val = isnan(val) ? CUDART_INF_F : val;

        const bool is_better = (val < local_min_val) ||
                               (val == local_min_val && idx_val < local_min_idx);
        local_min_val = is_better ? val : local_min_val;
        local_min_idx = is_better ? idx_val : local_min_idx;
    }

    warp_reduce_packed_min<WS>(local_min_val, local_min_idx);

    if (lane_idx == 0) {
        s_min_val[warp_idx] = local_min_val;
        s_min_idx[warp_idx] = local_min_idx;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_min_val = CUDART_INF_F;
        int   block_min_idx = -1;

        if (lane_idx < WPB) {
            block_min_val = s_min_val[lane_idx];
            block_min_idx = s_min_idx[lane_idx];
        }

        warp_reduce_packed_min<WS>(block_min_val, block_min_idx);

        if (lane_idx == 0) {
            d_expts_idx[obj_idx] = block_min_idx;
        }
    }
}

// =================================================================================== //
/**
 * @brief CUDA Kernel: Stage 3 - Extract Extreme Points Matrix
 * 
 * Note on data layout:
 * - d_expts is stored as ROW-MAJOR: d_expts[row * M + col]
 * - cuSOLVER interprets this as COLUMN-MAJOR, effectively computing SVD of E^T
 * - The subsequent pinv computation accounts for this transposition
 */
// =================================================================================== //
__global__ void find_expts_phase3_kernel(
    const float* __restrict__ d_ndsfv,         // input: (M, N_nds) - translated fitness matrix (shifted by ideal points)
    const int*   __restrict__ d_expts_idx,     // input: (M,) - global extreme point indices
    float*       __restrict__ d_expts,         // output: (M, M) - extreme points matrix (ROW-MAJOR)
    int M,                                     // input: scalar - number of objectives
    int N_nds                                  // input: scalar - N_prior + N_last
) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    if (col < M && row < M) {
        const int min_idx = d_expts_idx[col];

        if (min_idx >= 0 && min_idx < N_nds) {
            d_expts[row * M + col] = d_ndsfv[row * N_nds + min_idx];
        } else {
            d_expts[row * M + col] = CUDART_INF_F;
        }
    }
}

// ============================================================================================= //
/**
 * @brief CUDA Kernel: Element-wise multiplication (Hadamard product)
 */
// ============================================================================================= //
__global__ void hadamard_product_kernel(
    const float* __restrict__ a,   // input: (M,) - first input vector
    const float* __restrict__ b,   // input: (M,) - second input vector
    float* __restrict__ out,       // output: (M,) - element-wise product result
    int M                          // input: scalar - vector length
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        out[i] = a[i] * b[i];
    }
}

// ============================================================================================= //
/**
 * @brief CUDA Kernel: Sanitize reciprocal intercepts (d_rcp_b)
 * 
 * [FIX E] Final sanitization pass for d_rcp_b after SVD-based computation
 * Even if SVD succeeds, the hyperplane solution a = E^+ * 1 may contain:
 * - Negative values (invalid for normalization)
 * - NaN/INF values (numerical instability)
 * 
 * This kernel ensures all reciprocal intercepts are finite and positive.
 * Invalid values are replaced with 1.0 (no normalization for that objective).
 */
// ============================================================================================= //
__global__ void sanitize_rcp_b_kernel(
    float* __restrict__ d_rcp_b,   // update: (M,) - reciprocal intercepts to sanitize
    int M                          // input: scalar - number of objectives
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    float val = d_rcp_b[idx];
    
    // [FIX E] Check for invalid values: NaN, INF, negative, or zero
    // If invalid, use 1.0 (no normalization) as safe fallback
    if (!isfinite(val) || val <= 0.0f) {
        d_rcp_b[idx] = 1.0f;
    }
}

// ============================================================================================== //
/**
 * @brief CUDA Kernel: Apply normalization to population fitness
 * 
 * [FIX #4] NaN handling: NaN fitness values remain NaN after normalization
 *          This is intentional - NaN individuals should be handled at selection stage
 */
// ============================================================================================== //
__global__ void normalize_ndsfit_kernel(
    const float* __restrict__ d_rcp_b,   // input: (M,) - reciprocal of intercepts
    float* __restrict__ d_ndsfv,         // update: (M, N_nds) - normalized in-place: f'' = f' * (1 / intercept)
    const int M,                         // input: scalar - number of objectives
    const int N_nds                      // input: scalar - N_prior + N_last
) {
    int row_idx = blockIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= M || col_idx >= N_nds) return;

    float fitness       = d_ndsfv[row_idx * N_nds + col_idx];
    float inv_intercept = d_rcp_b[row_idx];
    d_ndsfv[row_idx * N_nds + col_idx] = fitness * inv_intercept;
}

// ================================================================================================= //
/**
 * @brief Host Function: Apply normalization to population fitness (Shared by both versions)
 * 
 * [FIX E] Added sanitization of d_rcp_b before applying normalization
 */
// ================================================================================================= //
void normalize_ndsfit(
    float* d_ndsfv,                 // update: (M, N_nds) - input: shifted fitness; output: normalized fitness
    float* d_rcp_b,                 // update: (M,) - reciprocal of intercepts (sanitized, then read-only)
    int M,                          // input: scalar - number of objectives
    int N_nds,                      // input: scalar - N_prior + N_last
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    // [FIX E] Sanitize d_rcp_b before normalization to ensure all values are valid
    sanitize_rcp_b_kernel<<<(M + 31) / 32, 32, 0, exec_stream>>>(d_rcp_b, M);
    CUDA_CHECK(cudaGetLastError());

    constexpr int BLK_SIZE = 256;
    dim3 grid_intercept((N_nds + BLK_SIZE - 1) / BLK_SIZE, M);
    normalize_ndsfit_kernel<<<grid_intercept, BLK_SIZE, 0, exec_stream>>>(d_rcp_b, d_ndsfv, M, N_nds);
    CUDA_CHECK(cudaGetLastError());
}

// =================================================================================== //
//                     LEGACY VERSION SPECIFIC KERNELS
// =================================================================================== //

// =================================================================================== //
/**
 * @brief CUDA Kernel: Stage 1 - Block-level ASF Computation (Legacy Version)
 * 
 * [FIX #4] Unified NaN handling: NaN fitness values are treated as +INF (worst)
 * [FIX F] ASF penalty now uses centralized constant NormConstants::ASF_PENALTY
 */
// =================================================================================== //
template <const int BPT, const int M, const int WPB>
__global__ void compute_asf_legacy_kernel(
    const float* __restrict__ d_ndsfv,    // input: (M, N_nds) - translated fitness matrix (shifted by ideal points)
    float* d_expts_blk_min,               // output: (num_blks_nds, M) - block-level minimum ASF values
    int*   d_expts_blk_amin,              // output: (num_blks_nds, M) - block-level argmin indices
    int N_nds                             // input: scalar - N_prior + N_last
) {
    __shared__ float s_min_val[WPB];
    __shared__ int   s_min_idx[WPB];

    const int obj_idx  = blockIdx.y;
    const int tidx     = threadIdx.x;
    const int glb_idx  = blockIdx.x * blockDim.x + tidx;
    const int col_base = glb_idx * BPT;

    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    float loc_min = CUDART_INF_F;
    int   loc_min_idx = -1;

    if (col_base < N_nds) {
        float lane_max[BPT];
        #pragma unroll
        for (int i = 0; i < BPT; ++i) lane_max[i] = -CUDART_INF_F;

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            // [FIX F] Use centralized ASF penalty constant
            const float penalty = (m == obj_idx) ? 1.0f : NormConstants::ASF_PENALTY;
            const int row_offset = m * N_nds;

            #pragma unroll
            for (int i = 0; i < BPT; ++i) {
                const int col_idx = col_base + i;
                if (col_idx < N_nds) {
                    float val = d_ndsfv[row_offset + col_idx];
                    // [FIX #4] Unified NaN handling: NaN -> +INF (worst possible)
                    val = isnan(val) ? CUDART_INF_F : val;
                    const float asf_val = val * penalty;
                    lane_max[i] = fmaxf(lane_max[i], asf_val);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < BPT; ++i) {
            const int col_idx = col_base + i;
            if (col_idx < N_nds) {
                const bool is_better = (lane_max[i] < loc_min) ||
                                       (lane_max[i] == loc_min && col_idx < loc_min_idx);
                loc_min = is_better ? lane_max[i] : loc_min;
                loc_min_idx = is_better ? col_idx : loc_min_idx;
            }
        }
    }

    warp_reduce_packed_min<WS>(loc_min, loc_min_idx);

    if (lane_idx == 0) {
        s_min_val[warp_idx] = loc_min;
        s_min_idx[warp_idx] = loc_min_idx;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_min_val = CUDART_INF_F;
        int   block_min_idx = -1;

        if (lane_idx < WPB) {
            block_min_val = s_min_val[lane_idx];
            block_min_idx = s_min_idx[lane_idx];
        }

        warp_reduce_packed_min<WS>(block_min_val, block_min_idx);

        if (lane_idx == 0) {
            const int out_idx = blockIdx.x * M + obj_idx;
            d_expts_blk_min[out_idx]  = block_min_val;
            d_expts_blk_amin[out_idx] = block_min_idx;
        }
    }
}

// ============================================================================================= //
/**
 * @brief CUDA Kernel: Check Extreme Points Ill-Conditioning (Legacy Version)
 * 
 * [FIX #8] Fixed logic: separate checks for sigma_min < EPSILON and condition number
 * [FIX C] Added check for NaN/INF singular values - force fallback if sigma is not finite
 */
// ============================================================================================= //
__global__ void check_expt_illcond_legacy_kernel(
    const float* __restrict__ d_sigma,    // input: (M,) - singular values from SVD (descending)
    int M,                                // input: scalar - number of objectives
    int* illcond_flag                     // output: (1,) - ill-conditioning flag (1 if ill-conditioned)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float sigma_max = d_sigma[0];
    const float sigma_min = d_sigma[M - 1];

    constexpr float EPSILON = NormConstants::LEGACY_EPSILON;
    constexpr float COND_THRESHOLD = NormConstants::LEGACY_COND_THRESHOLD;

    // [FIX C] Check for NaN/INF singular values first - force fallback if not finite
    // IEEE 754: NaN comparisons always return false, so we must check explicitly
    if (!isfinite(sigma_max) || !isfinite(sigma_min)) {
        *illcond_flag = 1;
        return;
    }

    // [FIX #8] Corrected logic: check sigma_min first, then compute true condition number
    // Previous logic was flawed: it computed condition_number with clamped sigma_min
    bool is_sigma_min_tiny = (sigma_min < EPSILON);
    bool is_ill_conditioned = false;
    
    if (!is_sigma_min_tiny) {
        // Safe to compute condition number only when sigma_min is not too small
        float condition_number = sigma_max / sigma_min;
        is_ill_conditioned = (condition_number > COND_THRESHOLD);
    }
    
    *illcond_flag = (is_sigma_min_tiny || is_ill_conditioned) ? 1 : 0;
}

// ================================================================================= //
/**
 * @brief CUDA Kernel: Compute pseudo-inverse of singular values (Legacy - Absolute Threshold)
 */
// ================================================================================= //
__global__ void cal_pinv_sigma_abs_kernel(
    const float* __restrict__ d_sigma,        // input: (M,) - singular values from SVD
    float*       __restrict__ d_sigma_cross,  // output: (M,) - pseudo-inverse singular values (masked)
    int M,                                    // input: scalar - number of objectives
    float threshold = NormConstants::LEGACY_PINV_THRESHOLD // input: scalar - absolute threshold for pinv masking
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= M) return;

    float val = d_sigma[tidx];
    float reciprocal = 1.0f / val;
    uint32_t mask = -(uint32_t)(val > threshold);
    uint32_t result_bits = __float_as_uint(reciprocal) & mask;
    d_sigma_cross[tidx]  = __uint_as_float(result_bits);
}

// ============================================================================================ //
/**
 * @brief CUDA Kernel: Find row maximum and compute reciprocal (Legacy - Safe Fallback)
 * 
 * [FIX #4] Unified NaN handling: NaN values are excluded from max computation
 * [FIX A] Changed fallback from INF to 1.0 to prevent NaN pollution (0 * INF = NaN)
 */
// ============================================================================================ //
template <const int BPT, const int WPB>
__global__ void find_intercepts_legacy_kernel(
    const float* __restrict__ d_ndsfv,    // input: (M, N_nds) - translated fitness matrix
    float* __restrict__ d_rcp_b,          // output: (M,) - reciprocal of row maxima (1.0 fallback if too small)
    int M,                                // input: scalar - number of objectives
    int N_nds                             // input: scalar - N_prior + N_last
) {
    __shared__ float s_max[WPB];

    const int obj_idx  = blockIdx.y;
    const int tidx     = threadIdx.x;
    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    float warp_max = -CUDART_INF_F;

    for (int col_idx = tidx * BPT; col_idx < N_nds; col_idx += blockDim.x * BPT) {
        #pragma unroll
        for (int i = 0; i < BPT; ++i) {
            int current_col = col_idx + i;
            if (current_col < N_nds) {
                float val = d_ndsfv[obj_idx * N_nds + current_col];
                // [FIX #4] Unified NaN handling: exclude NaN from max computation
                val = isnan(val) ? -CUDART_INF_F : val;
                warp_max = fmaxf(warp_max, val);
            }
        }
    }

    warp_reduce_max_f32<WS>(warp_max);

    if (lane_idx == 0) {
        s_max[warp_idx] = warp_max;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_max = (lane_idx < WPB) ? s_max[lane_idx] : -CUDART_INF_F;
        warp_reduce_max_f32<WS>(block_max);

        if (lane_idx == 0) {
            // [FIX A] Legacy fallback should never output INF as reciprocal intercept.
            // INF will cause 0*INF = NaN and poison the population.
            // Use safe_max to prevent division by zero, and 1.0 as fallback when max is too small.
            const float safe_max = fmaxf(block_max, NormConstants::RANGE_MIN_ABS);
            const float reciprocal = 1.0f / safe_max;

            // Use 1.0 (no normalization) if block_max is too small or non-positive.
            const uint32_t mask = -(uint32_t)(block_max > NormConstants::INTERCEPT_MIN);
            const uint32_t recip_bits = __float_as_uint(reciprocal) & mask;
            const uint32_t one_bits   = __float_as_uint(1.0f) & ~mask;
            d_rcp_b[obj_idx] = __uint_as_float(recip_bits | one_bits);
        }
    }
}

// =================================================================================== //
//                      RESCALE VERSION SPECIFIC KERNELS
// =================================================================================== //

// =================================================================================== //
/**
 * @brief CUDA Kernel: Fused Shift + Range Computation (Rescale Version)
 * 
 * [FIX #4] Unified NaN handling: NaN values are excluded from min/max computation
 * [FIX I] Changed to exclude ALL non-finite values (NaN AND INF) from range computation
 *         Previously only excluded NaN, which caused INF to pollute range calculation
 * [FIX J] Clamp shifted values to >= 0 to prevent negative ASF values
 */
// =================================================================================== //
template <const int WPB>
__global__ void shift_and_compute_ranges_kernel(
    float* __restrict__ d_ndsfv,          // update: (M, N_nds) - input: raw fitness; output: shifted fitness (subtract ideal)
    const float* __restrict__ d_ipt,      // input: (M,) - ideal points
    float* __restrict__ d_obj_ranges,     // output: (M,) - per-objective dynamic ranges (protected by abs+rel floors)
    int M,                                // input: scalar - number of objectives
    int N_nds                             // input: scalar - population size (N_prior + N_last)
) {
    __shared__ float s_min[WPB];
    __shared__ float s_max[WPB];

    const int obj_idx  = blockIdx.x;
    const int tidx     = threadIdx.x;
    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    const float ideal = d_ipt[obj_idx];
    const int row_offset = obj_idx * N_nds;

    float local_min = CUDART_INF_F;
    float local_max = -CUDART_INF_F;

    for (int col = tidx; col < N_nds; col += blockDim.x) {
        const int idx = row_offset + col;

        float val = d_ndsfv[idx];
        val -= ideal;
        
        // [FIX J] Clamp to non-negative (only for finite values)
        // This prevents negative ASF values from numerical errors or historical ideal points
        if (isfinite(val)) {
            val = fmaxf(val, 0.0f);
        }
        
        d_ndsfv[idx] = val;

        // [FIX I] Exclude ALL non-finite values (NaN AND INF) from range computation
        // Previously: if (!isnan(val)) - this allowed INF to pollute the range
        // If INF participates: range becomes INF, then gets clamped to RANGE_MIN_ABS,
        // causing that objective to be scaled by 1e6, severely distorting extreme point selection
        if (isfinite(val)) {
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }

    #pragma unroll
    for (int offset = WS / 2; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_xor_sync(0xFFFFFFFF, local_min, offset));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }

    if (lane_idx == 0) {
        s_min[warp_idx] = local_min;
        s_max[warp_idx] = local_max;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_min = (lane_idx < WPB) ? s_min[lane_idx] : CUDART_INF_F;
        float block_max = (lane_idx < WPB) ? s_max[lane_idx] : -CUDART_INF_F;

        #pragma unroll
        for (int offset = WS / 2; offset > 0; offset >>= 1) {
            block_min = fminf(block_min, __shfl_xor_sync(0xFFFFFFFF, block_min, offset));
            block_max = fmaxf(block_max, __shfl_xor_sync(0xFFFFFFFF, block_max, offset));
        }

        if (lane_idx == 0) {
            float range = block_max - block_min;
            
            // [FIX B] Handle edge case where all values are non-finite
            // (block_max = -INF, block_min = +INF after excluding non-finite values)
            // This would cause range = -INF - INF = -INF
            if (!isfinite(range) || range < 0.0f) {
                range = NormConstants::RANGE_MIN_ABS;
            } else {
                // [FIX I] abs_max calculation also uses only finite values (block_max/block_min are finite here)
                const float abs_max = fmaxf(fabsf(block_max), fabsf(block_min));
                const float rel_floor = NormConstants::RANGE_MIN_REL * abs_max;
                const float floor = fmaxf(NormConstants::RANGE_MIN_ABS, rel_floor);
                range = fmaxf(range, floor);
            }
            
            d_obj_ranges[obj_idx] = range;
        }
    }
}

// =================================================================================== //
/**
 * @brief CUDA Kernel: Stage 1 - Block-level ASF with Dynamic Range Normalization (Rescale)
 * 
 * [FIX #4] Unified NaN handling: NaN fitness values are treated as +INF
 * [FIX #9] Shared memory optimization for inv_range broadcast
 * [FIX B] Added protection against INF * 0 -> NaN in ASF computation
 * [FIX F] ASF penalty now uses centralized constant NormConstants::ASF_PENALTY
 */
// =================================================================================== //
template <const int BPT, const int M, const int WPB>
__global__ void compute_asf_rescale_kernel(
    const float* __restrict__ d_ndsfv,     // input: (M, N_nds) - translated fitness matrix
    const float* __restrict__ d_obj_ranges,// input: (M,) - per-objective dynamic ranges (protected)
    float* d_expts_blk_min,                // output: (num_blks_nds, M) - block-level minimum ASF values
    int*   d_expts_blk_amin,               // output: (num_blks_nds, M) - block-level argmin indices
    int N_nds                              // input: scalar - N_prior + N_last
) {
    __shared__ float s_min_val[WPB];
    __shared__ int   s_min_idx[WPB];
    
    // [FIX #9] Shared memory for inv_range broadcast - reduces global memory traffic
    __shared__ float s_inv_range[M];

    const int obj_idx  = blockIdx.y;
    const int tidx     = threadIdx.x;
    const int glb_idx  = blockIdx.x * blockDim.x + tidx;
    const int col_base = glb_idx * BPT;

    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    // [FIX #9] Cooperative load of inv_range into shared memory
    // [FIX B] Also validate inv_range to prevent 1/INF = 0 issues
    if (tidx < M) {
        float range = d_obj_ranges[tidx];
        // Ensure range is finite and positive
        if (!isfinite(range) || range <= 0.0f) {
            range = NormConstants::RANGE_MIN_ABS;
        }
        s_inv_range[tidx] = 1.0f / range;
    }
    __syncthreads();

    float loc_min = CUDART_INF_F;
    int   loc_min_idx = -1;

    if (col_base < N_nds) {
        float lane_max[BPT];
        #pragma unroll
        for (int i = 0; i < BPT; ++i) lane_max[i] = -CUDART_INF_F;

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            // [FIX F] Use centralized ASF penalty constant
            const float penalty = (m == obj_idx) ? 1.0f : NormConstants::ASF_PENALTY;
            const int row_offset = m * N_nds;
            // [FIX #9] Read from shared memory instead of global
            const float inv_range_m = s_inv_range[m];

            #pragma unroll
            for (int i = 0; i < BPT; ++i) {
                const int col_idx = col_base + i;
                if (col_idx < N_nds) {
                    float val = d_ndsfv[row_offset + col_idx];
                    
                    // [FIX #4] Unified NaN handling: NaN -> +INF
                    // [FIX B] Prevent INF * 0 -> NaN when inv_range is 0 or val is INF
                    if (isnan(val)) {
                        val = CUDART_INF_F;
                    }
                    
                    float asf_val;
                    // [FIX B] If val is not finite (INF), force asf_val to INF (worst)
                    // This prevents INF * 0 = NaN when inv_range_m is 0
                    if (!isfinite(val)) {
                        asf_val = CUDART_INF_F;
                    } else {
                        asf_val = val * inv_range_m * penalty;
                        // [FIX B] Final guard: if result is NaN due to edge cases, use INF
                        if (!isfinite(asf_val)) {
                            asf_val = CUDART_INF_F;
                        }
                    }
                    
                    lane_max[i] = fmaxf(lane_max[i], asf_val);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < BPT; ++i) {
            const int col_idx = col_base + i;
            if (col_idx < N_nds) {
                const bool is_better = (lane_max[i] < loc_min) ||
                                       (lane_max[i] == loc_min && col_idx < loc_min_idx);
                loc_min = is_better ? lane_max[i] : loc_min;
                loc_min_idx = is_better ? col_idx : loc_min_idx;
            }
        }
    }

    warp_reduce_packed_min<WS>(loc_min, loc_min_idx);

    if (lane_idx == 0) {
        s_min_val[warp_idx] = loc_min;
        s_min_idx[warp_idx] = loc_min_idx;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_min_val = CUDART_INF_F;
        int   block_min_idx = -1;

        if (lane_idx < WPB) {
            block_min_val = s_min_val[lane_idx];
            block_min_idx = s_min_idx[lane_idx];
        }

        warp_reduce_packed_min<WS>(block_min_val, block_min_idx);

        if (lane_idx == 0) {
            const int out_idx = blockIdx.x * M + obj_idx;
            d_expts_blk_min[out_idx]  = block_min_val;
            d_expts_blk_amin[out_idx] = block_min_idx;
        }
    }
}

// ================================================================================= //
/**
 * @brief CUDA Kernel: Compute ill-conditioning flag and pinv threshold on device (Rescale)
 * 
 * [FIX C] Added check for NaN/INF singular values - force fallback if sigma is not finite
 */
// ================================================================================= //
__global__ void compute_illcond_and_threshold_kernel(
    const float* __restrict__ d_sigma,     // input: (M,) - singular values from SVD (descending)
    int M,                                 // input: scalar - number of objectives
    int* d_illcond_flag,                   // output: (1,) - ill-conditioning flag
    float* d_pinv_threshold                // output: (1,) - pinv threshold = TAU_REL * sigma_max
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float sigma_max = d_sigma[0];
    const float sigma_min = d_sigma[M - 1];

    // [FIX C] Check for NaN/INF singular values first - force fallback if not finite
    // IEEE 754: NaN comparisons always return false, so we must check explicitly
    if (!isfinite(sigma_max) || !isfinite(sigma_min)) {
        *d_illcond_flag = 1;
        // [FIX C] Set a safe pinv_threshold to avoid NaN propagation
        *d_pinv_threshold = NormConstants::TAU_ABS;
        return;
    }

    const float pinv_thresh = NormConstants::TAU_REL * sigma_max;
    *d_pinv_threshold = pinv_thresh;

    const bool is_illcond = (sigma_min <= NormConstants::TAU_ABS) ||
                            (sigma_max > NormConstants::TAU_ABS && sigma_min <= pinv_thresh);

    *d_illcond_flag = is_illcond ? 1 : 0;
}

// ================================================================================= //
/**
 * @brief CUDA Kernel: Compute pseudo-inverse (Rescale - Relative Threshold from Device)
 */
// ================================================================================= //
__global__ void cal_pinv_sigma_rel_kernel(
    const float* __restrict__ d_sigma,           // input: (M,) - singular values from SVD
    float*       __restrict__ d_sigma_cross,     // output: (M,) - pseudo-inverse singular values (masked)
    int M,                                       // input: scalar - number of objectives
    const float* __restrict__ d_pinv_threshold   // input: (1,) - pinv threshold stored on device
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= M) return;

    const float threshold = *d_pinv_threshold;
    float val = d_sigma[tidx];
    float reciprocal = 1.0f / val;
    uint32_t mask = -(uint32_t)(val > threshold);
    uint32_t result_bits = __float_as_uint(reciprocal) & mask;
    d_sigma_cross[tidx]  = __uint_as_float(result_bits);
}

// ============================================================================================ //
/**
 * @brief CUDA Kernel: Find row maximum and compute reciprocal (Rescale - No INF)
 * 
 * [FIX #4] Unified NaN handling: NaN values are excluded from max computation
 */
// ============================================================================================ //
template <const int BPT, const int WPB>
__global__ void find_intercepts_rescale_kernel(
    const float* __restrict__ d_ndsfv,     // input: (M, N_nds) - translated fitness matrix
    float* __restrict__ d_rcp_b,           // output: (M,) - reciprocal of row maxima (1.0 if max too small)
    int M,                                 // input: scalar - number of objectives
    int N_nds                              // input: scalar - N_prior + N_last
) {
    __shared__ float s_max[WPB];

    const int obj_idx  = blockIdx.y;
    const int tidx     = threadIdx.x;
    const int warp_idx = tidx / WS;
    const int lane_idx = tidx % WS;

    float warp_max = -CUDART_INF_F;

    for (int col_idx = tidx * BPT; col_idx < N_nds; col_idx += blockDim.x * BPT) {
        #pragma unroll
        for (int i = 0; i < BPT; ++i) {
            int current_col = col_idx + i;
            if (current_col < N_nds) {
                float val = d_ndsfv[obj_idx * N_nds + current_col];
                // [FIX #4] Unified NaN handling: exclude NaN from max computation
                val = isnan(val) ? -CUDART_INF_F : val;
                warp_max = fmaxf(warp_max, val);
            }
        }
    }

    warp_reduce_max_f32<WS>(warp_max);

    if (lane_idx == 0) {
        s_max[warp_idx] = warp_max;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float block_max = (lane_idx < WPB) ? s_max[lane_idx] : -CUDART_INF_F;
        warp_reduce_max_f32<WS>(block_max);

        if (lane_idx == 0) {
            const float safe_max = fmaxf(block_max, NormConstants::RANGE_MIN_ABS);
            const float reciprocal = 1.0f / safe_max;
            const uint32_t mask = -(uint32_t)(block_max > NormConstants::INTERCEPT_MIN);
            const uint32_t recip_bits = __float_as_uint(reciprocal) & mask;
            const uint32_t one_bits   = __float_as_uint(1.0f) & ~mask;
            d_rcp_b[obj_idx] = __uint_as_float(recip_bits | one_bits);
        }
    }
}

// =================================================================================== //
//                     LEGACY VERSION HOST FUNCTIONS
// =================================================================================== //

// ================================================================================================= //
/**
 * @brief Host Function: Find Extreme Points (Legacy Version)
 * 
 * [FIX #6] Returns error code instead of void
 */
// ================================================================================================= //
int find_expts_legacy(
    float*       d_ndsfv,           // update: (M, N_nds) - translated fitness (shifted by ideal points) in-place
    const float* d_idpts,           // input: (M,) - ideal points
    float*       d_expts_blk_min,   // buffer: (num_blks_nds, M) - block-level minimum ASF values
    int*         d_expts_blk_amin,  // buffer: (num_blks_nds, M) - block-level argmin indices
    int*         d_expts_idx,       // buffer: (M,) - global extreme point indices
    float*       d_expts,           // output: (M, M) - extreme points matrix (ROW-MAJOR)
    int          M,                 // input: scalar - number of objectives
    int          N_nds,             // input: scalar - N_prior + N_last
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    constexpr int BPT = 4;
    constexpr int TS  = BLK_SIZE * BPT;
    const int num_blks_nds = (N_nds + TS - 1) / TS;

    // Stage 0: Translation (with FIX J clamping)
    {
        const bool is_vec4 = (N_nds % 4 == 0);
        if (is_vec4) {
            const int grid_x = (N_nds + (BLK_SIZE * 4) - 1) / (BLK_SIZE * 4);
            dim3 grid_trans(grid_x, M);
            translated_ndsfit_vec_kernel<<<grid_trans, BLK_SIZE, 0, exec_stream>>>(
                d_ndsfv, d_idpts, M, N_nds
            );
        } else {
            const int grid_x = (N_nds + BLK_SIZE - 1) / BLK_SIZE;
            dim3 grid_trans(grid_x, M);
            translated_ndsfit_kernel<<<grid_trans, BLK_SIZE, 0, exec_stream>>>(
                d_ndsfv, d_idpts, M, N_nds
            );
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 1: ASF computation
    // [FIX #6] Return error code for unsupported M
    {
        dim3 grid_asf(num_blks_nds, M);
        #define LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(M_VAL) \
            compute_asf_legacy_kernel<BPT, M_VAL, WPB> \
            <<<grid_asf, BLK_SIZE, 0, exec_stream>>>(d_ndsfv, d_expts_blk_min, d_expts_blk_amin, N_nds);

        switch (M) {
            case 1: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(1); break;
            case 2: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(2); break;
            case 3: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(3); break;
            case 4: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(4); break;
            case 5: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(5); break;
            case 6: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(6); break;
            case 7: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(7); break;
            case 8: LAUNCH_COMPUTE_ASF_LEGACY_KERNEL(8); break;
            default:
                fprintf(stderr, "ERROR (Normalization): Unsupported objective count M = %d (max supported: 8)\n", M);
                return NormError::UNSUPPORTED_M;
        }
        #undef LAUNCH_COMPUTE_ASF_LEGACY_KERNEL
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 2: Global reduction
    find_expts_phase2_kernel<WPB><<<M, BLK_SIZE, 0, exec_stream>>>(
        d_expts_blk_min, d_expts_blk_amin, d_expts_idx, num_blks_nds, M
    );
    CUDA_CHECK(cudaGetLastError());

    // Stage 3: Extract extreme points matrix
    find_expts_phase3_kernel<<<M, M, 0, exec_stream>>>(
        d_ndsfv, d_expts_idx, d_expts, M, N_nds
    );
    CUDA_CHECK(cudaGetLastError());
    
    return NormError::SUCCESS;
}

// =================================================================================================== //
/**
 * @brief Host Function: Perform SVD and check ill-conditioning (Legacy Version)
 * 
 * [FIX #1] Added SVD convergence check - returns SVD_NOT_CONVERGED if SVD fails
 * [FIX G] Explicit stream synchronization before cudaMemcpy for correctness
 *         under per-thread default stream mode (--default-stream per-thread)
 */
// =================================================================================================== //
int run_expts_svd_legacy(
    float*  d_expts,                // buffer: (M, M) - extreme points matrix (ROW-MAJOR; destroyed by SVD)
    int*    d_svdinfo,              // buffer: (1,) - SVD convergence info (0 = success)
    float*  d_U,                    // output: (M, M) - left singular vectors
    float*  d_sigma,                // output: (M,) - singular values (descending)
    float*  d_Vh,                   // output: (M, M) - right singular vectors transposed
    int*    d_illcond,              // buffer: (1,) - ill-conditioning flag (device)
    int     M,                      // input: scalar - number of objectives
    cudaMemPool_t exec_pool,        // input: scalar - CUDA memory pool
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    cusolverDnHandle_t cusolverH = HandlePool::instance().get_cusolver(exec_stream);

    int lwork_svd = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolverH, M, M, &lwork_svd));

    float* d_work_svd = nullptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_work_svd, lwork_svd * FLOAT_SIZE, exec_pool, exec_stream));

    CUSOLVER_CHECK(cusolverDnSgesvd(
        cusolverH, 'A', 'A', M, M,
        d_expts, M, d_sigma, d_U, M, d_Vh, M,
        d_work_svd, lwork_svd, nullptr, d_svdinfo
    ));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_work_svd, exec_stream));
    CUDA_CHECK(cudaMemsetAsync(d_illcond, 0, INT_SIZE, exec_stream));

    check_expt_illcond_legacy_kernel<<<1, 32, 0, exec_stream>>>(d_sigma, M, d_illcond);
    CUDA_CHECK(cudaGetLastError());

    // [FIX G] Explicit stream synchronization before cudaMemcpy
    // This is required for correctness under per-thread default stream mode.
    // Without this, cudaMemcpy may read stale data if compiled with --default-stream per-thread
    // because the implicit synchronization behavior of cudaMemcpy only applies to legacy default stream.
    CUDA_CHECK(cudaStreamSynchronize(exec_stream));

    // [FIX #1] Check SVD convergence info
    int h_svdinfo = 0;
    CUDA_CHECK(cudaMemcpy(&h_svdinfo, d_svdinfo, INT_SIZE, cudaMemcpyDeviceToHost));
    
    if (h_svdinfo != 0) {
        // SVD did not converge - signal caller to use fallback
        fprintf(stderr, "WARNING (Normalization): SVD did not converge (info=%d), using fallback\n", h_svdinfo);
        return 1;  // Return ill-conditioned flag to trigger fallback
    }

    int h_illcond = 0;
    CUDA_CHECK(cudaMemcpy(&h_illcond, d_illcond, INT_SIZE, cudaMemcpyDeviceToHost));
    return h_illcond;
}

// ==================================================================================================== //
/**
 * @brief Host Function: Compute normalization factors (Legacy Version)
 */
// ==================================================================================================== //
void compute_norm_factor_legacy(
    const float* d_ndsfv,           // input: (M, N_nds) - translated fitness matrix
    float* d_rcp_b,                 // output: (M,) - reciprocal intercepts (hyperplane solution or fallback)
    const float*  d_U,              // input: (M, M) - left singular vectors
    const float*  d_sigma,          // input: (M,) - singular values
    const float*  d_Vh,             // input: (M, M) - right singular vectors transposed
    float*  d_sigma_cross,          // buffer: (M,) - pseudo-inverse of sigma
    const float* d_ones,            // input: (M,) - vector of ones
    float*  d_u,                    // buffer: (M,) - intermediate: U^T * ones
    float*  d_s,                    // buffer: (M,) - intermediate: sigma_cross .* u
    int h_illcond,                  // input: scalar - ill-conditioning flag (host)
    int M,                          // input: scalar - number of objectives
    int N_nds,                      // input: scalar - N_prior + N_last
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    if (h_illcond == 1) {
        constexpr int BPT = 2, WPB = 8, BLK_SIZE = WPB * WS;
        dim3 grid_idx(1, M);
        find_intercepts_legacy_kernel<BPT, WPB><<<grid_idx, BLK_SIZE, 0, exec_stream>>>(d_ndsfv, d_rcp_b, M, N_nds);
        CUDA_CHECK(cudaGetLastError());
    } else {
        cublasHandle_t cublasH = HandlePool::instance().get_cublas(exec_stream);

        cal_pinv_sigma_abs_kernel<<<(M + 31) / 32, 32, 0, exec_stream>>>(d_sigma, d_sigma_cross, M);
        CUDA_CHECK(cudaGetLastError());

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublasH, CUBLAS_OP_T, M, M, &alpha, d_U, M, d_ones, 1, &beta, d_u, 1));

        hadamard_product_kernel<<<(M + 63) / 64, 64, 0, exec_stream>>>(d_sigma_cross, d_u, d_s, M);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasSgemv(cublasH, CUBLAS_OP_T, M, M, &alpha, d_Vh, M, d_s, 1, &beta, d_rcp_b, 1));
    }
}

// =================================================================================== //
//                      RESCALE VERSION HOST FUNCTIONS
// =================================================================================== //

// ================================================================================================= //
/**
 * @brief Host Function: Find Extreme Points with Dynamic Range Normalization (Rescale Version)
 * 
 * [FIX #6] Returns error code instead of void
 */
// ================================================================================================= //
int find_expts_rescale(
    float*       d_ndsfv,           // update: (M, N_nds) - fitness shifted by ideal points (in-place)
    const float* d_idpts,           // input: (M,) - ideal points
    float*       d_obj_ranges,      // buffer: (M,) - per-objective dynamic ranges (protected)
    float*       d_expts_blk_min,   // buffer: (num_blks_nds, M) - block-level minimum ASF values
    int*         d_expts_blk_amin,  // buffer: (num_blks_nds, M) - block-level argmin indices
    int*         d_expts_idx,       // buffer: (M,) - global extreme point indices
    float*       d_expts,           // output: (M, M) - extreme points matrix (ROW-MAJOR)
    int          M,                 // input: scalar - number of objectives
    int          N_nds,             // input: scalar - N_prior + N_last
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    constexpr int WPB = 8;
    constexpr int BLK_SIZE = WPB * WS;
    constexpr int BPT = 4;
    constexpr int TS  = BLK_SIZE * BPT;
    const int num_blks_nds = (N_nds + TS - 1) / TS;

    // Stage 0+0.5 (FUSED): Translation + Range computation (with FIX I and FIX J)
    {
        shift_and_compute_ranges_kernel<WPB><<<M, BLK_SIZE, 0, exec_stream>>>(
            d_ndsfv, d_idpts, d_obj_ranges, M, N_nds
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 1: Normalized ASF computation
    // [FIX #6] Return error code for unsupported M
    {
        dim3 grid_asf(num_blks_nds, M);
        #define LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(M_VAL) \
            compute_asf_rescale_kernel<BPT, M_VAL, WPB> \
            <<<grid_asf, BLK_SIZE, 0, exec_stream>>>(d_ndsfv, d_obj_ranges, d_expts_blk_min, d_expts_blk_amin, N_nds);

        switch (M) {
            case 1: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(1); break;
            case 2: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(2); break;
            case 3: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(3); break;
            case 4: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(4); break;
            case 5: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(5); break;
            case 6: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(6); break;
            case 7: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(7); break;
            case 8: LAUNCH_COMPUTE_ASF_RESCALE_KERNEL(8); break;
            default:
                fprintf(stderr, "ERROR (Normalization): Unsupported objective count M = %d (max supported: 8)\n", M);
                return NormError::UNSUPPORTED_M;
        }
        #undef LAUNCH_COMPUTE_ASF_RESCALE_KERNEL
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 2: Global reduction
    find_expts_phase2_kernel<WPB><<<M, BLK_SIZE, 0, exec_stream>>>(
        d_expts_blk_min, d_expts_blk_amin, d_expts_idx, num_blks_nds, M
    );
    CUDA_CHECK(cudaGetLastError());

    // Stage 3: Extract extreme points matrix
    find_expts_phase3_kernel<<<M, M, 0, exec_stream>>>(
        d_ndsfv, d_expts_idx, d_expts, M, N_nds
    );
    CUDA_CHECK(cudaGetLastError());
    
    return NormError::SUCCESS;
}

// =================================================================================================== //
/**
 * @brief Host Function: Perform SVD and compute ill-conditioning on device (Rescale Version)
 * 
 * [FIX #1] Added SVD convergence check - returns SVD_NOT_CONVERGED if SVD fails
 * [FIX G] Explicit stream synchronization before cudaMemcpy for correctness
 *         under per-thread default stream mode (--default-stream per-thread)
 */
// =================================================================================================== //
int run_expts_svd_rescale(
    float*  d_expts,                // buffer: (M, M) - extreme points matrix (ROW-MAJOR; destroyed by SVD)
    int*    d_svdinfo,              // buffer: (1,) - SVD convergence info (0 = success)
    float*  d_U,                    // output: (M, M) - left singular vectors
    float*  d_sigma,                // output: (M,) - singular values (descending)
    float*  d_Vh,                   // output: (M, M) - right singular vectors transposed
    int*    d_illcond,              // buffer: (1,) - ill-conditioning flag (device)
    float*  d_pinv_threshold,       // buffer: (1,) - pinv threshold on device (TAU_REL * sigma_max)
    int     M,                      // input: scalar - number of objectives
    cudaMemPool_t exec_pool,        // input: scalar - CUDA memory pool
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    cusolverDnHandle_t cusolverH = HandlePool::instance().get_cusolver(exec_stream);

    int lwork_svd = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolverH, M, M, &lwork_svd));

    float* d_work_svd = nullptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&d_work_svd, lwork_svd * FLOAT_SIZE, exec_pool, exec_stream));

    CUSOLVER_CHECK(cusolverDnSgesvd(
        cusolverH, 'A', 'A', M, M,
        d_expts, M, d_sigma, d_U, M, d_Vh, M,
        d_work_svd, lwork_svd, nullptr, d_svdinfo
    ));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_work_svd, exec_stream));

    // [FIX G] Explicit stream synchronization before cudaMemcpy
    // This is required for correctness under per-thread default stream mode.
    // Without this, cudaMemcpy may read stale data if compiled with --default-stream per-thread
    CUDA_CHECK(cudaStreamSynchronize(exec_stream));

    // [FIX #1] Check SVD convergence info first
    int h_svdinfo = 0;
    CUDA_CHECK(cudaMemcpy(&h_svdinfo, d_svdinfo, INT_SIZE, cudaMemcpyDeviceToHost));
    
    if (h_svdinfo != 0) {
        // SVD did not converge - signal caller to use fallback
        fprintf(stderr, "WARNING (Normalization): SVD did not converge (info=%d), using fallback\n", h_svdinfo);
        return 1;  // Return ill-conditioned flag to trigger fallback
    }

    compute_illcond_and_threshold_kernel<<<1, 32, 0, exec_stream>>>(
        d_sigma, M, d_illcond, d_pinv_threshold
    );
    CUDA_CHECK(cudaGetLastError());

    // [FIX G] Another sync needed after illcond kernel before reading result
    CUDA_CHECK(cudaStreamSynchronize(exec_stream));

    int h_illcond = 0;
    CUDA_CHECK(cudaMemcpy(&h_illcond, d_illcond, INT_SIZE, cudaMemcpyDeviceToHost));
    return h_illcond;
}

// ==================================================================================================== //
/**
 * @brief Host Function: Compute normalization factors (Rescale Version)
 */
// ==================================================================================================== //
void compute_norm_factor_rescale(
    const float* d_ndsfv,           // input: (M, N_nds) - translated fitness matrix
    float* d_rcp_b,                 // output: (M,) - reciprocal intercepts (hyperplane solution or fallback)
    const float*  d_U,              // input: (M, M) - left singular vectors
    const float*  d_sigma,          // input: (M,) - singular values
    const float*  d_Vh,             // input: (M, M) - right singular vectors transposed
    float*  d_sigma_cross,          // buffer: (M,) - pseudo-inverse of sigma
    const float* d_ones,            // input: (M,) - vector of ones
    float*  d_u,                    // buffer: (M,) - intermediate: U^T * ones
    float*  d_s,                    // buffer: (M,) - intermediate: sigma_cross .* u
    const float* d_pinv_threshold,  // input: (1,) - device pinv threshold (TAU_REL * sigma_max)
    int h_illcond,                  // input: scalar - ill-conditioning flag (host)
    int M,                          // input: scalar - number of objectives
    int N_nds,                      // input: scalar - N_prior + N_last
    cudaStream_t exec_stream        // input: scalar - CUDA execution stream
) {
    if (h_illcond == 1) {
        constexpr int BPT = 2, WPB = 8, BLK_SIZE = WPB * WS;
        dim3 grid_idx(1, M);
        find_intercepts_rescale_kernel<BPT, WPB><<<grid_idx, BLK_SIZE, 0, exec_stream>>>(d_ndsfv, d_rcp_b, M, N_nds);
        CUDA_CHECK(cudaGetLastError());
    } else {
        cublasHandle_t cublasH = HandlePool::instance().get_cublas(exec_stream);

        cal_pinv_sigma_rel_kernel<<<(M + 31) / 32, 32, 0, exec_stream>>>(d_sigma, d_sigma_cross, M, d_pinv_threshold);
        CUDA_CHECK(cudaGetLastError());

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublasH, CUBLAS_OP_T, M, M, &alpha, d_U, M, d_ones, 1, &beta, d_u, 1));

        hadamard_product_kernel<<<(M + 63) / 64, 64, 0, exec_stream>>>(d_sigma_cross, d_u, d_s, M);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasSgemv(cublasH, CUBLAS_OP_T, M, M, &alpha, d_Vh, M, d_s, 1, &beta, d_rcp_b, 1));
    }
}

// =================================================================================== //
//                     EXECUTE_NORMALIZATION ENTRY POINTS
// =================================================================================== //

// ============================================================================================================================================================================================= //
/**
 * @brief Host Function: Execute NSGA-III normalization pipeline (Legacy Version)
 * 
 * [FIX #6] Returns error code (NormError::SUCCESS on success, negative on failure)
 * [FIX E] Added sanitization of d_rcp_b via normalize_ndsfit
 */
// ============================================================================================================================================================================================= //
int execute_normalization(
    float*  d_ndsfv,                // update: (M, N_nds) - input: raw fitness; intermediate: shifted; output: normalized
    const float*  d_idpts,           // input: (M,) - ideal points
    float*  d_expts_blk_min,         // buffer: (num_blks_nds, M) - block-level min ASF values
    int*    d_expts_blk_amin,        // buffer: (num_blks_nds, M) - block-level argmin indices
    int*    d_expts_idx,             // buffer: (M,) - global extreme point indices
    float*  d_expts,                 // buffer: (M, M) - extreme points matrix (ROW-MAJOR; destroyed by SVD)

    float*  d_rcp_b,                 // buffer: (M,) - reciprocal intercepts

    int*    d_svdinfo,               // buffer: (1,) - SVD convergence info
    float*  d_U,                     // buffer: (M, M) - left singular vectors
    float*  d_sigma,                 // buffer: (M,) - singular values
    float*  d_Vh,                    // buffer: (M, M) - right singular vectors transposed
    int*    d_illcond,               // buffer: (1,) - ill-conditioning flag (device)
    float*  d_sigma_cross,           // buffer: (M,) - pseudo-inverse sigma
    const float*  d_ones,            // input: (M,) - vector of ones
    float*  d_u,                     // buffer: (M,) - intermediate: U^T * ones
    float*  d_s,                     // buffer: (M,) - intermediate: sigma_cross .* u
    int     M,                       // input: scalar - number of objectives
    int     N_nds,                   // input: scalar - N_prior + N_last
    cudaMemPool_t exec_pool,         // input: scalar - CUDA memory pool
    cudaStream_t exec_stream         // input: scalar - CUDA execution stream
) 
{
    // [FIX #6] Check return codes and propagate errors
    int ret = find_expts_legacy(d_ndsfv, d_idpts, d_expts_blk_min, d_expts_blk_amin,
                                d_expts_idx, d_expts, M, N_nds, exec_stream);
    if (ret != NormError::SUCCESS) {
        return ret;
    }

    int h_illcond = run_expts_svd_legacy(d_expts, d_svdinfo, d_U, d_sigma, d_Vh,
                                         d_illcond, M, exec_pool, exec_stream);
    // Note: h_illcond >= 0 is valid (0 = well-conditioned, 1 = ill-conditioned/SVD failed)

    compute_norm_factor_legacy(d_ndsfv, d_rcp_b, d_U, d_sigma, d_Vh, d_sigma_cross,
                               d_ones, d_u, d_s, h_illcond, M, N_nds, exec_stream);

    // [FIX E] normalize_ndsfit now includes sanitization of d_rcp_b
    normalize_ndsfit(d_ndsfv, d_rcp_b, M, N_nds, exec_stream);
    
    return NormError::SUCCESS;
}

// ============================================================================================================================================================================================= //
/**
 * @brief Host Function: Execute NSGA-III normalization pipeline (Rescale Version)
 * 
 * [FIX #6] Returns error code (NormError::SUCCESS on success, negative on failure)
 * [FIX E] Added sanitization of d_rcp_b via normalize_ndsfit
 */
// ============================================================================================================================================================================================= //
int execute_normalization_rescale(
    float*  d_ndsfv,                 // update: (M, N_nds) - input: raw fitness; intermediate: shifted; output: normalized
    const float*  d_idpts,            // input: (M,) - ideal points
    float*  d_obj_ranges,             // buffer: (M,) - per-objective dynamic ranges
    float*  d_expts_blk_min,          // buffer: (num_blks_nds, M) - block-level min ASF values
    int*    d_expts_blk_amin,         // buffer: (num_blks_nds, M) - block-level argmin indices
    int*    d_expts_idx,              // buffer: (M,) - global extreme point indices
    float*  d_expts,                  // buffer: (M, M) - extreme points matrix (ROW-MAJOR; destroyed by SVD)

    float*  d_rcp_b,                  // buffer: (M,) - reciprocal intercepts

    int*    d_svdinfo,                // buffer: (1,) - SVD convergence info
    float*  d_U,                      // buffer: (M, M) - left singular vectors
    float*  d_sigma,                  // buffer: (M,) - singular values
    float*  d_Vh,                     // buffer: (M, M) - right singular vectors transposed
    int*    d_illcond,                // buffer: (1,) - ill-conditioning flag (device)
    float*  d_pinv_threshold,         // buffer: (1,) - device pinv threshold (TAU_REL * sigma_max)
    float*  d_sigma_cross,            // buffer: (M,) - pseudo-inverse sigma
    const float*  d_ones,             // input: (M,) - vector of ones
    float*  d_u,                      // buffer: (M,) - intermediate: U^T * ones
    float*  d_s,                      // buffer: (M,) - intermediate: sigma_cross .* u
    int     M,                        // input: scalar - number of objectives
    int     N_nds,                    // input: scalar - N_prior + N_last
    cudaMemPool_t exec_pool,          // input: scalar - CUDA memory pool
    cudaStream_t exec_stream          // input: scalar - CUDA execution stream
)
{
    // [FIX #6] Check return codes and propagate errors
    int ret = find_expts_rescale(d_ndsfv, d_idpts, d_obj_ranges, d_expts_blk_min, d_expts_blk_amin,
                                 d_expts_idx, d_expts, M, N_nds, exec_stream);
    if (ret != NormError::SUCCESS) {
        return ret;
    }

    int h_illcond = run_expts_svd_rescale(d_expts, d_svdinfo, d_U, d_sigma, d_Vh,
                                          d_illcond, d_pinv_threshold, M, exec_pool, exec_stream);
    // Note: h_illcond >= 0 is valid (0 = well-conditioned, 1 = ill-conditioned/SVD failed)

    compute_norm_factor_rescale(d_ndsfv, d_rcp_b, d_U, d_sigma, d_Vh, d_sigma_cross,
                                d_ones, d_u, d_s, d_pinv_threshold, h_illcond, M, N_nds, exec_stream);

    // [FIX E] normalize_ndsfit now includes sanitization of d_rcp_b
    normalize_ndsfit(d_ndsfv, d_rcp_b, M, N_nds, exec_stream);
    
    return NormError::SUCCESS;
}
