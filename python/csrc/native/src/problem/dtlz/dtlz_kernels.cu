#include "math_constants.h"

#include "cuda_moea/core/cuda/cuda_utils.cuh"
#include "cuda_moea/core/cuda/cuda_warpreduce.cuh"
#include "cuda_moea/problem/dtlz/dtlz_kernels.cuh"
#include "cuda_moea/core/cuda/cuda_transpose.cuh"

// ========================================================================================================================= //
// ================ Helper Functions for Numerical Stability ================
/**
 * Clamp value to range [min_val, max_val]
 */
__device__ __forceinline__ float clamp_value(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, val));
}
/**
 * Safe cosine function that ensures non-negative results
 * Critical for numerical stability in DTLZ2-6
 */
__device__ __forceinline__ float safe_cos(float x) {
    return fmaxf(0.0f, cosf(x));
}
// ========================================================================================================================= //
/**
 * DTLZ1 Test Function Kernel with Numerical Protection (1D Thread Structure)
 */
template<const int WPB>
__global__ void dtlz1_func_kernel(
    const float* d_pop,    // input: (N, D) - d_pop
    float* d_fit_trans,    // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr int BLK_SIZE = WPB * WS;
    // Static shared memory for warp reduction
    __shared__ float sdata[WPB];

    int row_idx    = blockIdx.x;
    int tid        = threadIdx.x;
    if (row_idx >= N) return;

    // Calculate g-function domain: last k = D - M + 1 variables
    int sum_start  = M - 1;
    int sum_length = D - sum_start;

    // Parallel summation for g-function components
    float thread_sum = 0.0f;
    for (int i = tid; i < sum_length; i += BLK_SIZE) {
        int d = sum_start + i;
        float val = d_pop[row_idx * D + d] - 0.5f;
        // Rastrigin-like component: (xi-0.5)^2 - cos(20π(xi-0.5))
        thread_sum += val * val - cosf(20.0f * M_PI * val);
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum<float, WS>(thread_sum);

    // Store warp results in shared memory
    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final computation by thread 0
    if (tid == 0) {
        // Aggregate all warp results
        float total_sum = 0.0f;
        int num_warps = (BLK_SIZE + WS - 1) / WS;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += sdata[i];
        }

        // Complete g-function calculation
        float g = 100.0f * (static_cast<float>(sum_length) + total_sum);
        float scalar = 0.5f * (1.0f + g);

        // Load first M-1 decision variables for objective computation
        float x[MAX_M];
        for (int i = 0; i < M - 1; ++i) {
            x[i] = d_pop[row_idx * D + i];
        }

        // Compute M objectives with numerical protection
        for (int m = 0; m < M; ++m) {
            float f = scalar;

            // Product term with clamping for numerical stability
            int prod_length = M - 1 - m;
            for (int j = 0; j < prod_length; ++j) {
                // Clamp to [0,1] to prevent invalid values
                float x_clamped = clamp_value(x[j], 0.0f, 1.0f);
                f *= x_clamped;
            }

            // Diversity term (1 - xi) for i > 0 with protection
            if (m > 0) {
                int diversity_idx = M - 1 - m;
                if (diversity_idx < M - 1) {
                    // Ensure non-negative diversity term
                    float diversity_term = fmaxf(0.0f, 1.0f - x[diversity_idx]);
                    f *= diversity_term;
                }
            }

            d_fit_trans[row_idx * M + m] = f;
        }
    }
}

// ========================================================================================================================= //
/**
 * DTLZ2 Test Function Kernel with Numerical Protection (1D Thread Structure)
 */
template<const int WPB>
__global__ void dtlz2_func_kernel(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_fit_trans,       // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr int BLK_SIZE = WPB * WS;
    // Static shared memory for warp reduction
    __shared__ float sdata[WPB];

    int row_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (row_idx >= N) return;

    // Calculate g-function domain
    int sum_start = M - 1;
    int sum_length = D - sum_start;

    // Parallel summation for simple quadratic g-function
    float thread_sum = 0.0f;
    for (int i = tid; i < sum_length; i += BLK_SIZE) {
        int d = sum_start + i;
        float val = d_pop[row_idx * D + d] - 0.5f;
        thread_sum += val * val;
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum<float, WS>(thread_sum);

    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final computation by thread 0
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < WPB; ++i) {
            total_sum += sdata[i];
        }

        float g = total_sum;

        // Load first M-1 decision variables
        float x[MAX_M];
        for (int i = 0; i < M - 1; ++i) {
            x[i] = d_pop[row_idx * D + i];
        }

        // Compute objectives using trigonometric transformation with protection
        for (int m = 0; m < M; ++m) {
            float f = 1.0f + g;

            // Product of cosines for convergence with numerical protection
            for (int j = 0; j < M - 1 - m; ++j) {
                // Use safe_cos to ensure non-negative values
                f *= safe_cos(x[j] * M_PI / 2.0f);
            }

            // Sine term for diversity
            if (m > 0) {
                f *= sinf(x[M - 1 - m] * M_PI / 2.0f);
            }

            d_fit_trans[row_idx * M + m] = f;
        }
    }
}
// ========================================================================================================================= //
template<const int WPB>
__global__ void convex_dtlz2_func_kernel(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_fit_trans,       // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr int BLK_SIZE = WPB * WS;
    __shared__ float sdata[WPB];

    int row_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (row_idx >= N) return;

    // Calculate g-function domain
    int sum_start = M - 1;
    int sum_length = D - sum_start;

    // Parallel summation for quadratic g-function
    float thread_sum = 0.0f;
    for (int i = tid; i < sum_length; i += BLK_SIZE) {
        int d = sum_start + i;
        float val = d_pop[row_idx * D + d] - 0.5f;
        thread_sum += val * val;
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum<float, WS>(thread_sum);

    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final computation by thread 0
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < WPB; ++i) {
            total_sum += sdata[i];
        }

        float g = total_sum;

        // Load first M-1 decision variables
        float x[MAX_M];
        for (int i = 0; i < M - 1; ++i) {
            x[i] = d_pop[row_idx * D + i];
        }

        // Compute objectives using trigonometric transformation
        for (int m = 0; m < M; ++m) {
            float f = 1.0f + g;

            // Product of cosines for convergence
            for (int j = 0; j < M - 1 - m; ++j) {
                f *= safe_cos(x[j] * M_PI / 2.0f);
            }

            // Sine term for diversity
            if (m > 0) {
                f *= sinf(x[M - 1 - m] * M_PI / 2.0f);
            }

            // ============================================
            // Power transformation for CONVEX shape (Deb et al.)
            // First M-1 objectives: f_i <- f_i^4
            // Last objective:       f_M <- f_M^2
            // Pareto-optimal surface: f_M + sum(sqrt(f_i)) = 1
            // ============================================
            if (m < M - 1) {
                // First M-1 objectives: apply 4th power
                f = f * f;      // f^2
                f = f * f;      // f^4
            } else {
                // Last objective: apply 2nd power
                f = f * f;      // f^2
            }

            d_fit_trans[row_idx * M + m] = f;
        }
    }
}
// ========================================================================================================================= //
/**
 * DTLZ3 Test Function Kernel with Critical Numerical Protection (1D Thread Structure)
 */
template<const int WPB>
__global__ void dtlz3_func_kernel(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_fit_trans,       // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr int BLK_SIZE = WPB * WS;
    // Static shared memory for warp reduction
    __shared__ float sdata[WPB];

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= N) return;

    // Calculate g-function domain
    int sum_start = M - 1;
    int sum_length = D - sum_start;

    // Parallel summation for multimodal g-function (same as DTLZ1)
    float thread_sum = 0.0f;
    for (int i = tid; i < sum_length; i += BLK_SIZE) {
        int d = sum_start + i;
        float val = d_pop[row_idx * D + d] - 0.5f;
        thread_sum += val * val - cosf(20.0f * M_PI * val);
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum<float, WS>(thread_sum);

    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final computation by thread 0
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < WPB; ++i) {
            total_sum += sdata[i];
        }

        // Multimodal g-function
        float g = 100.0f * (static_cast<float>(sum_length) + total_sum);

        float x[MAX_M];
        for (int i = 0; i < M - 1; ++i) {
            x[i] = d_pop[row_idx * D + i];
        }

        // Same objective computation as DTLZ2 but with critical protection
        for (int m = 0; m < M; ++m) {
            float f = 1.0f + g;

            // CRITICAL: Use safe_cos to prevent negative values
            for (int j = 0; j < M - 1 - m; ++j) {
                f *= safe_cos(x[j] * M_PI / 2.0f);
            }

            if (m > 0) {
                f *= sinf(x[M - 1 - m] * M_PI / 2.0f);
            }

            d_fit_trans[row_idx * M + m] = f;
        }
    }
}

// ========================================================================================================================= //
/**
 * DTLZ4 Test Function Kernel with Numerical Protection (1D Thread Structure)
 */
template<const int WPB>
__global__ void dtlz4_func_kernel(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_fit_trans,       // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M,
    float alpha
) {
    constexpr int BLK_SIZE = WPB * WS;
    // Static shared memory for warp reduction
    __shared__ float sdata[WPB];

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (row_idx >= N) return;

    // Calculate g-function domain
    int sum_start = M - 1;
    int sum_length = D - sum_start;

    // Parallel summation for quadratic g-function
    float thread_sum = 0.0f;
    for (int i = tid; i < sum_length; i += BLK_SIZE) {
        int d = sum_start + i;
        float val = d_pop[row_idx * D + d] - 0.5f;
        thread_sum += val * val;
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum<float, WS>(thread_sum);

    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final computation by thread 0
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < WPB; ++i) {
            total_sum += sdata[i];
        }

        float g = total_sum;

        float x[MAX_M];
        for (int i = 0; i < M - 1; ++i) {
            x[i] = d_pop[row_idx * D + i];
        }

        // Compute objectives with alpha bias and protection
        for (int m = 0; m < M; ++m) {
            float f = 1.0f + g;

            // Apply x^α transformation for bias with safe_cos
            for (int j = 0; j < M - 1 - m; ++j) {
                // Use safe_cos for numerical stability with alpha power
                f *= safe_cos(powf(x[j], alpha) * M_PI / 2.0f);
            }

            if (m > 0) {
                f *= sinf(powf(x[M - 1 - m], alpha) * M_PI / 2.0f);
            }

            d_fit_trans[row_idx * M + m] = f;
        }
    }
}
// ========================================================================================================================= //
template<const int WPB>
__global__ void dtlz5_func_kernel(
    const float* __restrict__ d_pop,    // input: (N, D) - d_pop
    float* __restrict__ d_fit_trans,    // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr int BLK_SIZE   = WS * WPB;
    constexpr uint32_t FMASK = 0xFFFFFFFFu;

    int tidx = blockIdx.x;
    int tid = threadIdx.x;
    if (tidx >= N) return;
    // ========== Phase 1: Parallel g computation with shuffle reduction ==========
    int k = D - M + 1;
    int sum_start = M - 1;

    // Coalesced memory reads - each thread reads different elements
    float local_sum = 0.0f;
    for (int base = 0; base < k; base += BLK_SIZE) {
        int i = base + tid;
        if (i < k) {
            int d = sum_start + i;
            // Use __ldg for better cache behavior (read-only cache)
            float val = __ldg(&d_pop[tidx * D + d]) - 0.5f;
            local_sum += val * val;
        }
    }

    // Warp-level reduction using shuffle (no shared memory needed!)
    #pragma unroll
    for (int offset = WS/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(FMASK, local_sum, offset);
    }

    // Inter-warp reduction if block has multiple warps
    __shared__ float warp_sums[WPB];
    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    float g = 0.0f;
    if (warp_id == 0) {
        float wsum = (lane_id < WPB) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WS >> 1; offset > 0; offset /= 2) {
            wsum += __shfl_down_sync(FMASK, wsum, offset);
        }
        g = wsum;
    }

    // Broadcast g to all threads
    g = __shfl_sync(FMASK, g, 0);
    if (warp_id > 0) {
        // For multi-warp blocks, use shared memory for inter-warp broadcast
        __shared__ float g_shared;
        if (tid == 0) g_shared = g;
        __syncthreads();
        g = g_shared;
    }

    // ========== Phase 2: Compute theta in registers ==========
    // Each thread computes ALL theta values locally (avoids synchronization)
    float theta[MAX_M];

    // Load first M-1 decision variables (coalesced if threads load consecutively)
    theta[0] = __ldg(&d_pop[tidx * D + 0]);

    float denom = 2.0f * (1.0f + g);
    #pragma unroll
    for (int i = 1; i < M - 1 && i < MAX_M - 1; ++i) {
        float xi = __ldg(&d_pop[tidx * D + i]);
        theta[i] = (1.0f + 2.0f * g * xi) / denom;
        theta[i] = clamp_value(theta[i], 0.0f, 1.0f);
    }

    // ========== Phase 3: Parallel objective computation ==========
    // Each thread computes one or more objectives
    for (int m = tid; m < M; m += BLK_SIZE) {
        float f = 1.0f + g;

        // Product of cosines
        #pragma unroll 8
        for (int j = 0; j < M - 1 - m && j < MAX_M - 1; ++j) {
            f *= fmaxf(0.0f, __cosf(theta[j] * M_PI * 0.5f));
        }

        // Sine term for diversity
        if (m > 0) {
            f *= __sinf(theta[M - 1 - m] * M_PI * 0.5f);
        }

        // Coalesced write pattern
        d_fit_trans[tidx * M + m] = f;
    }
}

// ========================================================================================================================= //
/**
 * DTLZ6 Test Function Kernel with Numerical Protection (1D Thread Structure)
 */
template<const int WPB>
__global__ void dtlz6_func_kernel(
    const float* __restrict__ d_pop,    // input: (N, D) - d_pop
    float* __restrict__ d_fit_trans,    // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr uint32_t FMASK = 0xFFFFFFFFu;
    constexpr int BLK_SIZE   = WS * WPB;

    int tidx = blockIdx.x;
    int tid = threadIdx.x;
    if (tidx >= N) return;

     // Inter-warp reduction using shared memory
    __shared__ float warp_sums[WPB];
    __shared__ float g_value;        // Single shared variable for final g

    int warp_id = tid / WS;
    int lane_id = tid % WS;

    // ========== Phase 1: Parallel g computation with shuffle reduction ==========
    int k = D - M + 1;  // Number of variables in g-function
    int sum_start = M - 1;

    // Each thread computes part of the sum
    float local_sum = 0.0f;
    for (int base = 0; base < k; base += BLK_SIZE) {
        int i = base + tid;
        if (i < k) {
            int d = sum_start + i;
            // Use standard memory read (not __ldg) for compatibility
            float val = d_pop[tidx * D + d];
            // DTLZ6 uses x^0.1 in g-function
            local_sum += powf(val, 0.1f);
        }
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WS >> 1; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(FMASK, local_sum, offset);
    }

    // Each warp leader stores its sum
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by thread 0
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < WPB; ++i) {
            total_sum += warp_sums[i];
        }
        g_value = total_sum;  // Store in shared memory for all threads
    }
    __syncthreads();

    // All threads now have access to g
    float g = g_value;

    // ========== Phase 2: Compute theta transformation ==========
    // CRITICAL: Each thread needs all theta values for objective computation
    // So we compute them redundantly in each thread (small array, worth it)
    float theta[MAX_M];

    // First theta is just x[0]
    theta[0] = d_pop[tidx * D + 0];

    // Transform remaining variables according to DTLZ5/6 formula
    float denom = 2.0f * (1.0f + g);
    for (int i = 1; i < M - 1; ++i) {  // Note: M-1 total theta values (0 to M-2)
        if (i < MAX_M) {  // Bounds check for static array
            float xi = d_pop[tidx * D + i];
            theta[i] = (1.0f + 2.0f * g * xi) / denom;
        }
    }

    // ========== Phase 3: Parallel objective computation ==========
    // Each thread computes one or more objectives
    for (int m = tid; m < M; m += BLK_SIZE) {
        float f = 1.0f + g;

        // Product of cosines for convergence
        for (int j = 0; j < M - 1 - m; ++j) {
            // Use standard cosf for accuracy (not __cosf)
            // Apply safe_cos pattern from working version
            f *= fmaxf(0.0f, cosf(theta[j] * M_PI * 0.5f));
        }

        // Sine term for diversity (only for m > 0)
        if (m > 0) {
            int theta_idx = M - 1 - m;
            if (theta_idx >= 0 && theta_idx < M - 1) {  // Bounds check
                f *= sinf(theta[theta_idx] * M_PI * 0.5f);
            }
        }

        // Coalesced write pattern
        d_fit_trans[tidx * M + m] = f;
    }
}
// ========================================================================================================================= //
/**
 * DTLZ7 Test Function Kernel (1D Thread Structure)
 */
template<const int WPB>
__global__ void dtlz7_func_kernel(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_fit_trans,       // output: (N, M) - d_fit_trans
    int N,
    int D,
    int M
) {
    constexpr int BLK_SIZE = WS * WPB;
    // Static shared memory for warp reduction
    __shared__ float sdata[WPB];

    int row_idx    = blockIdx.x;
    int tid        = threadIdx.x;
    if (row_idx >= N) return;

    // Calculate g-function parameters
    int k = D - M + 1;
    int sum_start = D - k;

    // Parallel summation for linear g-function
    float thread_sum = 0.0f;
    for (int i = tid; i < k; i += BLK_SIZE) {
        int d = sum_start + i;
        thread_sum += d_pop[row_idx * D + d];
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum<float, WS>(thread_sum);

    int warp_id = tid / WS;
    int lane_id = tid % WS;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final computation by thread 0
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < WPB; ++i) {
            total_sum += sdata[i];
        }

        // Linear g-function
        float g = 1.0f + (9.0f / static_cast<float>(k)) * total_sum;

        // First M-1 objectives are directly the decision variables
        float f[MAX_M];
        for (int i = 0; i < M - 1; ++i) {
            f[i] = d_pop[row_idx * D + i];
            d_fit_trans[row_idx * M + i] = f[i];
        }

        // Compute h function for disconnected regions
        float h_sum = 0.0f;
        for (int i = 0; i < M - 1; ++i) {
            h_sum += (f[i] / (1.0f + g)) * (1.0f + sinf(3.0f * M_PI * f[i]));
        }
        float h = static_cast<float>(M) - h_sum;

        // Last objective
        d_fit_trans[row_idx * M + (M - 1)] = (1.0f + g) * h;
    }
}
// ========================================================================================================================= //
// CSDP: three objectives with ten inequality constraints, D = 7
template<const int WPB>
__global__ void csdp_func_kernel(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_fit_trans,       // output: (N, M) - d_fit_trans
    int N
) {

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    float x1 = d_pop[tidx * 7 + 0];
    float x2 = d_pop[tidx * 7 + 1];
    float x3 = d_pop[tidx * 7 + 2];
    float x4 = d_pop[tidx * 7 + 3];
    float x5 = d_pop[tidx * 7 + 4];
    float x6 = d_pop[tidx * 7 + 5];
    float x7 = d_pop[tidx * 7 + 6];

    float F     = 4.72 - 0.5   * x4 - 0.19    * x2 * x3;
    float V_MBP = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2;
    float V_FD  = 16.45 - 0.489 * x3 * x7 - 0.843   * x5 * x6;

    d_fit_trans[tidx * 3 + 0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7;
    d_fit_trans[tidx * 3 + 1] = F;
    d_fit_trans[tidx * 3 + 2] = 0.5f * (V_MBP + V_FD);
}
// ========================================================================================================================= //
/**
 * Unified compute function for all DTLZ test functions (1D Thread Structure)
 */
void compute_fv(
    float* d_pop,           // input: (N, D) - d_pop
    float* d_fit_trans,     // output: (N, M) - d_fit_trans
    float* d_fit,           // output: (M, N) - transposed fitness matrix
    int N,                  // input: scalar - population size
    int D,                  // input: scalar - number of decision variables
    int M,                  // input: scalar - number of objectives
    cudaStream_t fit_stream,// input: scalar - CUDA stream for fitness computation
    MOPType dtlz_type       // input: scalar - selected DTLZ problem type
)
{
    // Memory and kernel configuration
    constexpr int WPB  = 8;
    constexpr int BLK_SIZE = WPB * WS;
    // Grid and block dimensions - 1D STRUCTURE
    dim3 block_dtlz(BLK_SIZE);  // 1D block with BLK_SIZE threads
    dim3 grid_dtlz(N);            // 1D grid with N blocks

    // Launch appropriate kernel based on DTLZ type
    switch(dtlz_type) {
        case DTLZ1:
            dtlz1_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case DTLZ2:
            dtlz2_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case DTLZ3:
            dtlz3_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case DTLZ4:
            dtlz4_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M, 100.0f);
            break;

        case DTLZ5:
            dtlz5_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case DTLZ6:
            dtlz6_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case DTLZ7:
            dtlz7_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case CONVEX_DTLZ2:
            convex_dtlz2_func_kernel<WPB><<<grid_dtlz, block_dtlz, 0, fit_stream>>>(
                d_pop, d_fit_trans, N, D, M);
            break;

        case CSDP:
               csdp_func_kernel<WPB><<<((N + BLK_SIZE - 1) / BLK_SIZE), BLK_SIZE, 0, fit_stream>>>(d_pop, d_fit_trans, N);
            break;
        default:
            printf("Error: Invalid DTLZ type %d. Valid types are 1-7.\n", dtlz_type);
            return;
    }

    CUDA_CHECK(cudaGetLastError());

    transpose_matrix(d_fit_trans, d_fit, N, M, fit_stream);
}
// ========================================================================================================================= //
/**
 * C1-DTLZ1 Constraint Violation Kernel
 */
__global__ void c1_dtlz1_cv_kernel(
    const float* __restrict__ d_fit,  // input: (M, N) - d_fit
    float* __restrict__ d_cv,         // output: (N,) - d_cv
    const int N,
    const int M
)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    // Start with f_M / 0.6 (last objective with different normalization)
    float constraint = d_fit[(M - 1) * N + tidx] / 0.6f;

    // Add sum of f_i / 0.5 for i = 1, ..., M-1
    #pragma unroll
    for (int m = 0; m < M - 1; m++) {
        float f_m = d_fit[m * N + tidx];
        constraint += f_m / 0.5f;
    }

    // Subtract 1 to complete constraint formula
    float cv = constraint - 1.0f;

    // Constraint violation = max(constraint, 0)
    d_cv[tidx] = fmaxf(cv, 0.0f);
}
// ========================================================================================================================= //
/**
 * C1-DTLZ3 Constraint Violation Kernel
 */
__global__ void c1_dtlz3_cv_kernel(
    const float* __restrict__ d_fit,  // input: (M, N) - d_fit
    float* __restrict__ d_cv,         // output: (N,) - d_cv
    const float R,                    // input: scalar - outer sphere radius
    const int   N,
    const int   M
)
{
    const float R_sqr = R * R;

    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    // Compute sum of squared fitness values
    float fv_sqr = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float f_m = d_fit[m * N + tidx];
        fv_sqr += f_m * f_m;
    }

    // Infeasible region: 4 <= ||f(x)||_2 <= R
    float cv_inner = fv_sqr - 16.0f;
    float cv_outer = R_sqr - fv_sqr;

    float cv = fmaxf(cv_inner * cv_outer, 0.0f);

    d_cv[tidx] = cv;
}
// ========================================================================================================================= //
/**
 * C2-DTLZ2 Constraint Violation Kernel
 */
__global__ void c2_dtlz2_cv_kernel(
    const float* __restrict__ d_fit,  // input: (M, N) - fitness matrix
    float* __restrict__ d_cv,         // output: (N,) - constraint violation values
    const float R,                    // input: scalar - sphere radius
    const int N,                      // input: scalar - population size
    const int M                       // input: scalar - number of objectives
){
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    const float inv_sqrt_M = 1.0f / sqrtf((float)M);
    const float r2 = R * R;
    const float inv_r2 = 1.0f / r2;

    // Compute sum of squared fitness
    float sum_f2 = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float fm = d_fit[m * N + tidx];
        sum_f2 = fmaf(fm, fm, sum_f2);
    }

    // Union: take min of all sphere constraints
    float min_cv = CUDART_INF_F;

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float fm = d_fit[m * N + tidx];
        float diff = fm - 1.0f;
        float dist2 = fmaf(diff, diff, sum_f2 - fm * fm);
        float cv_m = dist2 * inv_r2 - 1.0f;
        min_cv = fminf(min_cv, cv_m);
    }

    // Center sphere
    float center_sum2 = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float fm = d_fit[m * N + tidx];
        float df = fm - inv_sqrt_M;
        center_sum2 = fmaf(df, df, center_sum2);
    }
    float center_cv = center_sum2 * inv_r2 - 1.0f;

    // Union → take min
    float g = fminf(min_cv, center_cv);

    // Violation = max(g, 0)
    d_cv[tidx] = fmaxf(g, 0.0f);
}
// ========================================================================================================================= //
/**
 * Convex-C2-DTLZ2 Constraint Violation Kernel (Single-Pass)
 */
__global__ void convex_c2_dtlz2_cv_kernel(
    const float* __restrict__ d_fit,  // input: (M, N) - d_fit
    float* __restrict__ d_cv,         // output: (N,) - d_cv
    const float R,                    // input: scalar - sphere radius
    const int N,
    const int M
)
{
    const float inv_M = 1.0f / static_cast<float>(M);
    const float inv_R_sqr = 1.0f / (R * R);

    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    // Single-pass computation of sum and sum of squares
    float sum    = 0.0f;
    float sum_sq = 0.0f;

    #pragma unroll
    for (int m = 0; m < M; m++) {
        float f_m = d_fit[m * N + tidx];
        sum    += f_m;
        sum_sq += f_m * f_m;
    }

    // Variance formula: Var = E[X²] - E[X]²
    float mean = sum * inv_M;
    float variance = sum_sq - static_cast<float>(M) * mean * mean;

    // Constraint: 1 - variance/R² >= 0
    float constraint = 1.0f - variance * inv_R_sqr;

    d_cv[tidx] = fmaxf(constraint, 0.0f);
}
// ========================================================================================================================= //
/**
 * C3-DTLZ1 Constraint Violation Kernel
 */
__global__ void c3_dtlz1_cv_kernel(
    const float* __restrict__ d_fit,  // input: (M, N) - d_fit
    float* __restrict__ d_cv,         // output: (N,) - d_cv
    const int N,
    const int M
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute 2 * Σ_{m=1}^{M-1} f_m
    float fv_sum = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float f_m = d_fit[m * N + idx];
        fv_sum += f_m;
    }
    fv_sum *= 2.0f;

    // Compute M constraints
    float cv = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float f_m = d_fit[m * N + idx];
        float constraint = f_m + 1.0f - fv_sum;
        constraint = fmaxf(constraint, 0.0f);
        cv += constraint;
    }

    d_cv[idx] = cv;
}
// ========================================================================================================================= //
/**
 * C3-DTLZ4 Constraint Violation Kernel
 */
__global__ void c3_dtlz4_cv_kernel(
    const float* __restrict__ d_fit,  // input: (M, N) - d_fit
    float* __restrict__ d_cv,         // output: (N,) - d_cv
    const int N,
    const int M
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute sum of squared fitness
    float fv_sqr = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float f_m = d_fit[m * N + idx];
        fv_sqr += f_m * f_m;
    }

    // Compute M weighted constraints
    float cv = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float f_m = d_fit[m * N + idx];
        float constraint = 0.75f * (f_m * f_m) - fv_sqr;
        constraint += 1.0f;
        cv += fmaxf(constraint, 0.0f);
    }

    d_cv[idx] = cv;
}
// ========================================================================================================================= //
// CSDP: ten inequality constraints, D = 7
__global__ void csdp_cv_kernel(
    const float* __restrict__ d_pop,  // input: (N, D) - d_pop
    float* __restrict__ d_cv,          // output: (N,) - d_cv
    const int N
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    float x1 = d_pop[tidx * 7 + 0];
    float x2 = d_pop[tidx * 7 + 1];
    float x3 = d_pop[tidx * 7 + 2];
    float x4 = d_pop[tidx * 7 + 3];
    float x5 = d_pop[tidx * 7 + 4];
    float x6 = d_pop[tidx * 7 + 5];
    float x7 = d_pop[tidx * 7 + 6];

    float F     = 4.72f - 0.5f * x4 - 0.19f * x2 * x3;
    float V_MBP = 10.58f - 0.674f * x1 * x2 - 0.67275f * x2;
    float V_FD  = 16.45f - 0.489f * x3 * x7 - 0.843f * x5 * x6;

    float g1  = 1.16f - 0.3717f * x2 * x4 - 0.0092928f * x3 - 1.0f;
    float g2  = 0.261f - 0.0159f * x1 * x2 - 0.06486f * x1 - 0.019f * x2 * x7 + 0.0144f * x3 * x5 + 0.0154464f * x6 - 0.32f;
    float g3  = 0.214f + 0.00817f * x5 - 0.045195f * x1 - 0.0135168f * x1 + 0.03099f * x2 * x6 - 0.018f * x2 * x7 + 0.007176f * x3 + 0.023232f * x3 - 0.00364f * x5 * x6 - 0.018f * x2 * x2 - 0.32f;
    float g4  = 0.74f - 0.61f * x2 - 0.031296f * x3 - 0.031872f * x7 + 0.227f * x2 * x2 - 0.32f;
    float g5  = 28.98f + 3.818f * x3 - 4.2f * x1 * x2 + 1.27296f * x6 - 2.68065f * x7 - 32.0f;
    float g6  = 33.86f + 2.95f * x3 - 5.057f * x1 * x2 - 3.795f * x2 - 3.4431f * x7 + 1.45728f - 32.0f;
    float g7  = 46.36f - 9.9f * x2 - 4.4505f * x1 - 32.0f;
    float g8  = F - 4.0f;
    float g9  = V_MBP - 9.9f;
    float g10 = V_FD - 15.7f;

    g1  = fmaxf(g1, 0.0f);
    g2  = fmaxf(g2, 0.0f);
    g3  = fmaxf(g3, 0.0f);
    g4  = fmaxf(g4, 0.0f);
    g5  = fmaxf(g5, 0.0f);
    g6  = fmaxf(g6, 0.0f);
    g7  = fmaxf(g7, 0.0f);
    g8  = fmaxf(g8, 0.0f);
    g9  = fmaxf(g9, 0.0f);
    g10 = fmaxf(g10, 0.0f);

    d_cv[tidx] = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10;
}
// ========================================================================================================================= //
/**
 * Unified CV dispatch for constraint DTLZ variants
 */
void compute_cv(
    const float* d_fit,       // input: (M, N) - d_fit
    float* d_cv,              // output: (N,) - d_cv
    int N,
    int M,
    cudaStream_t stream,
    MOPType c_dtlz_type
)
{
    constexpr int BLK_SIZE = 256;
    int grid_size = (N + BLK_SIZE - 1) / BLK_SIZE;

    dim3 block(BLK_SIZE);
    dim3 grid(grid_size);

    switch(c_dtlz_type) {
        case C1_DTLZ1:
            c1_dtlz1_cv_kernel<<<grid, block, 0, stream>>>(d_fit, d_cv, N, M);
            break;
        case C1_DTLZ3:
            {
                float R = 15.0f;
                switch(M) {
                    case 2:  R = 6.0f;   break;
                    case 3:  R = 5.0f;   break;
                    case 5:  R = 12.5f;  break;
                    case 8:  break;
                    case 10: break;
                    case 15: break;
                    default:
                        printf("Error CV: Invalid M %d, please check.\n", M);
                        return;
                }
                c1_dtlz3_cv_kernel<<<grid, block, 0, stream>>>(d_fit, d_cv, R, N, M);
            }
            break;

        case C2_DTLZ2:
            {
                float R = 1.0f;
                switch(M) {
                    case 2:  R = 0.15f;  break;
                    case 3:  R = 0.40f;  break;
                    case 5:  break;
                    case 8:  break;
                    case 10: break;
                    case 15: break;
                    default:
                        printf("Error CV: Invalid M %d, please check.\n", M);
                        return;
                }
                c2_dtlz2_cv_kernel<<<grid, block, 0, stream>>>(d_fit, d_cv, R, N, M);
                break;
            }
        case C2_CONVEX_DTLZ2:
            {
                float R = 0.225f;
                switch(M) {
                    case 3:  R = 0.20f;  break;
                    case 5:  R = 0.225f; break;
                    case 8:  R = 0.26f;  break;
                    case 10: R = 0.26f;  break;
                    case 15: R = 0.27f;  break;
                    default:
                        printf("Error CV: Invalid M %d, please check.\n", M);
                        return;
                }
                convex_c2_dtlz2_cv_kernel<<<grid, block, 0, stream>>>(d_fit, d_cv, R, N, M);
                break;
            }
        case C3_DTLZ1:
            c3_dtlz1_cv_kernel<<<grid, block, 0, stream>>>(d_fit, d_cv, N, M);
            break;
        case C3_DTLZ4:
            c3_dtlz4_cv_kernel<<<grid, block, 0, stream>>>(d_fit, d_cv, N, M);
            break;
        default:
            printf("Error: Invalid Constraint DTLZ type %d, please check input expression.\n", c_dtlz_type);
            return;
    }

    CUDA_CHECK(cudaGetLastError());
}

void compute_csdp_cv(
    const float* d_pop,       // input: (N, D) - d_pop
    float* d_cv,              // output: (N,) - d_cv
    int N,
    cudaStream_t stream
)
{
    constexpr int BLK_SIZE = 256;
    int grid_size = (N + BLK_SIZE - 1) / BLK_SIZE;
    dim3 block(BLK_SIZE);
    dim3 grid(grid_size);
    csdp_cv_kernel<<<grid, block, 0, stream>>>(d_pop, d_cv, N);
}
// ========================================================================================================================= //
// ==================================================== Unconstrained DTLZ ================================================ //
void compute_dtlz1(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ1);
}
void compute_dtlz2(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ2);
}
void compute_dtlz3(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ3);
}
void compute_dtlz4(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ4);
}
void compute_dtlz5(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ5);
}
void compute_dtlz6(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ6);
}
void compute_dtlz7(
    float* d_pop, float* d_fit_trans, float* d_fit,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ7);
}
// ====================================== Constrained DTLZ ================================================================ //
void compute_c1_dtlz1(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ1);
    compute_cv(d_fit, d_cv, N, M, fit_stream, C1_DTLZ1);
}
void compute_c1_dtlz3(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ3);
    compute_cv(d_fit, d_cv, N, M, fit_stream, C1_DTLZ3);
}
void compute_c2_dtlz2(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ2);
    compute_cv(d_fit, d_cv, N, M, fit_stream, C2_DTLZ2);
}
void compute_c2_convex_dtlz2(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, CONVEX_DTLZ2);
    compute_cv(d_fit, d_cv, N, M, fit_stream, C2_CONVEX_DTLZ2);
}
void compute_c3_dtlz1(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ1);
    compute_cv(d_fit, d_cv, N, M, fit_stream, C3_DTLZ1);
}
void compute_c3_dtlz4(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, DTLZ4);
    compute_cv(d_fit, d_cv, N, M, fit_stream, C3_DTLZ4);
}
void compute_csdp(
    float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv,
    int N, int D, int M, cudaStream_t fit_stream) {
    compute_fv(d_pop, d_fit_trans, d_fit, N, D, M, fit_stream, CSDP);
    compute_csdp_cv(d_pop, d_cv, N, fit_stream);
}

// ======================================================================================================================================================= //
//                                              MOEAStdTestEvaluator Implementation                                                                         //
// ======================================================================================================================================================= //
namespace {

bool is_constrained_mop(MOPType mop_type) {
    return mop_type >= C1_DTLZ1;
}

MOPType constrained_base_mop(MOPType mop_type) {
    switch (mop_type) {
        case C1_DTLZ1:        return DTLZ1;
        case C1_DTLZ3:        return DTLZ3;
        case C2_DTLZ2:        return DTLZ2;
        case C2_CONVEX_DTLZ2: return CONVEX_DTLZ2;
        case C3_DTLZ1:        return DTLZ1;
        case C3_DTLZ4:        return DTLZ4;
        default:              return DTLZ1;
    }
}

} // namespace

void MOEAStdTestEvaluator::set_iteration_context(int _n_iter, int _N_iter) {
    n_iter = _n_iter;
    N_iter = (_N_iter > 0) ? _N_iter : 1;
}

bool MOEAStdTestEvaluator::is_cv_active(int eval_iter) const {
    if (config.cv_activation_ratio <= 0.0f) {
        return true;
    }
    if (eval_iter < 0 || N_iter <= 0) {
        return false;
    }
    const float progress = static_cast<float>(eval_iter + 1) / static_cast<float>(N_iter);
    return progress >= config.cv_activation_ratio;
}

bool MOEAStdTestEvaluator::is_cv_activation_switch_node(int target_iter) const {
    if (target_iter < 0) {
        return false;
    }
    if (last_prepared_target_iter == target_iter) {
        return false;
    }
    if (target_iter == 0) {
        return is_cv_active(0);
    }
    return !is_cv_active(target_iter - 1) && is_cv_active(target_iter);
}

void MOEAStdTestEvaluator::evaluate(cudaStreamSync cuda_streams, const PopData& d_pop, float* d_cv, float* d_fv) {
    cudaStream_t& fcal_stream = cuda_streams.fcal_stream;

    // Synchronize exec_stream → fcal_stream (pop/off updated on exec_stream)
    cuda_streams.wait_execstream_done_and_execute(fcal_stream);

    const int D = d_auxdata.D;
    MOPType mop_type = d_auxdata.mop_type;

    const bool is_constrained = is_constrained_mop(mop_type);

    if (!is_constrained) {
        // Unconstrained: compute fv, zero out cv
        compute_fv(d_pop.d_pop, d_auxdata.d_fv_trans, d_fv, N, D, M, fcal_stream, mop_type);
        CUDA_CHECK(cudaMemsetAsync(d_cv, 0, static_cast<size_t>(N) * sizeof(float), fcal_stream));
    } else if (mop_type == CSDP) {
        // CSDP: special handling (cv depends on d_pop, not d_fv)
        compute_fv(d_pop.d_pop, d_auxdata.d_fv_trans, d_fv, N, D, M, fcal_stream, CSDP);
        compute_csdp_cv(d_pop.d_pop, d_cv, N, fcal_stream);
    } else {
        // Constrained DTLZ: compute fv first, then cv from d_fv
        compute_fv(d_pop.d_pop, d_auxdata.d_fv_trans, d_fv, N, D, M, fcal_stream, constrained_base_mop(mop_type));
        compute_cv(d_fv, d_cv, N, M, fcal_stream, mop_type);
    }

    if (is_constrained && !is_cv_active(n_iter)) {
        CUDA_CHECK(cudaMemsetAsync(d_cv, 0, static_cast<size_t>(N) * sizeof(float), fcal_stream));
    }

    // Save fitness values to host if enabled
    saveDeviceArrayToBin<float>("h_fv", d_fv, static_cast<size_t>(M) * N, enable_h_save);
    saveDeviceArrayToBin<float>("h_cv", d_cv, N, enable_h_save);

    // Signal fcal_stream done → exec_stream can proceed
    cuda_streams.wait_fcalstream_done_and_execute(cuda_streams.exec_stream);
}

bool MOEAStdTestEvaluator::prepare_parent_cv(
    cudaStreamSync cuda_streams,
    const PopData& d_parent,
    const float* d_fv,
    float* d_cv,
    int target_iter,
    int n_active)
{
    if (!is_constrained_mop(d_auxdata.mop_type)) {
        return false;
    }
    if (!is_cv_activation_switch_node(target_iter)) {
        return false;
    }

    cudaStream_t& fcal_stream = cuda_streams.fcal_stream;
    cuda_streams.wait_execstream_done_and_execute(fcal_stream);

    if (d_auxdata.mop_type == CSDP) {
        compute_csdp_cv(d_parent.d_pop, d_cv, n_active, fcal_stream);
    } else {
        compute_cv(d_fv, d_cv, n_active, M, fcal_stream, d_auxdata.mop_type);
    }

    last_prepared_target_iter = target_iter;
    printf(
        "[StdTest-CV] parent CV activated at target_iter=%d, N_iter=%d, ratio=%.6f\n",
        target_iter,
        N_iter,
        config.cv_activation_ratio);

    cuda_streams.wait_fcalstream_done_and_execute(cuda_streams.exec_stream);
    return true;
}
