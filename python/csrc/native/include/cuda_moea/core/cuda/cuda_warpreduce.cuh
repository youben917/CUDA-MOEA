#pragma once

/**
 * Collection of warp-level reduction functions for CUDA
 * These functions perform collective operations across all threads in a warp (typically 32 threads)
 * using warp shuffle instructions for efficient inter-thread communication without shared memory
 */

/**
 * Performs a sum reduction across all threads in a warp
 * 
 * @tparam T Data type to reduce (must be supported by shuffle instructions)
 * @tparam WS Size of the warp (typically 32)
 * @param val Input value from current thread
 * @return Sum of all values across the warp (only valid in thread 0)
 * 
 * Algorithm explanation:
 * - Uses __shfl_down_sync to communicate between threads
 * - __shfl_down_sync(FMASK, val, offset) copies 'val' from thread (threadIdx + offset)
 * - Performs tree-like reduction: each iteration halves the number of active threads
 * - Example with 8 threads (simplified):
 *   Initial: [1, 2, 3, 4, 5, 6, 7, 8]
 *   Step 1 (offset=4): [6, 8, 10, 12, 5, 6, 7, 8] (threads 0-3 get values from threads 4-7)
 *   Step 2 (offset=2): [16, 20, 10, 12, 5, 6, 7, 8] (threads 0-1 get values from threads 2-3)
 *   Step 3 (offset=1): [36, 20, 10, 12, 5, 6, 7, 8] (thread 0 gets value from thread 1)
 *   Final result: 36 in thread 0
 */
template<typename T, const int WS>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    // Mask indicating which threads participate in the shuffle operation
    // 0xFFFFFFFFu means all 32 threads participate
    uint32_t FMASK = 0xFFFFFFFFu;
    
    // Unroll the loop at compile time for better performance
    #pragma unroll
    for (int offset = WS>>1; offset > 0; offset >>= 1) {
        // __shfl_down_sync: Each thread receives value from thread (current_thread_id + offset)
        // Threads with (current_thread_id + offset >= warp_size) receive undefined values
        // Only threads in the lower half of each iteration get valid results
        val += __shfl_down_sync(FMASK, val, offset);
    }
    return val; // Final result is only valid in thread 0
}

/**
 * Performs a bitwise OR reduction across all threads in a warp
 * 
 * @tparam T Data type to reduce (typically integer types)
 * @tparam WS Size of the warp
 * @param val Input value from current thread
 * @return Bitwise OR of all values across the warp (only valid in thread 0)
 * 
 * Uses the same tree reduction pattern as sum, but with bitwise OR operation
 */
template<typename T, const int WS>
__device__ __forceinline__ T warp_reduce_or(T val) {
    uint32_t FMASK = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = WS>>1; offset > 0; offset >>= 1) {
        // Perform bitwise OR with value from thread (current_thread_id + offset)
        val |= __shfl_down_sync(FMASK, val, offset);
    }
    return val;
}

/**
* Finds the minimum value across all threads in a warp
* 
* @tparam WS Size of the warp
* @param val Reference to input/output float value - will contain minimum on return (only valid in thread 0)
* 
* Key changes from XOR version:
* - Uses __shfl_down_sync instead of __shfl_xor_sync
* - Creates a tree reduction pattern where only thread 0 gets the final result
* - More efficient when only one thread needs the result
* - Result is only valid in thread 0, not all threads
* 
* Algorithm explanation:
* - Uses tree-like reduction: each iteration halves the number of active threads
* - __shfl_down_sync(FMASK, val, offset) gets value from thread (threadIdx + offset)
* - Example with 4 threads and values [1, 4, 2, 3]:
*   Step 1 (offset=2): [min(1,2)=1, min(4,3)=3, 2, 3] (threads 0-1 get values from threads 2-3)
*   Step 2 (offset=1): [min(1,3)=1, 4, 2, 3] (thread 0 gets value from thread 1)
*   Final: thread 0 has the minimum value 1, others have their intermediate values
*/
template<const int WS>
__device__ __forceinline__ void warp_reduce_min_f32(float &val) {
   uint32_t FMASK = 0xFFFFFFFFu;
   #pragma unroll
   for (int offset = WS >> 1; offset > 0; offset >>= 1) {
       // __shfl_down_sync: Each thread receives value from thread (current_thread_id + offset)
       // Threads with (current_thread_id + offset >= warp_size) receive undefined values
       // Only threads in the lower half of each iteration get valid results
       float other = __shfl_down_sync(FMASK, val, offset);
       val = fminf(val, other); // Keep the minimum of the two values
   }
   // Final result is only valid in thread 0
}

/**
* Finds the maximum value across all threads in a warp
* 
* @tparam WS Size of the warp
* @param val Reference to input/output float value - will contain maximum on return (only valid in thread 0)
* 
* Same tree reduction pattern as min reduction, but finds maximum
* Result is only valid in thread 0, not all threads
* 
* Algorithm explanation:
* - Uses tree-like reduction with __shfl_down_sync
* - Example with 4 threads and values [1, 4, 2, 3]:
*   Step 1 (offset=2): [max(1,2)=2, max(4,3)=4, 2, 3] (threads 0-1 get values from threads 2-3)
*   Step 2 (offset=1): [max(2,4)=4, 4, 2, 3] (thread 0 gets value from thread 1)
*   Final: thread 0 has the maximum value 4, others have their intermediate values
*/
template<const int WS>
__device__ __forceinline__ void warp_reduce_max_f32(float &val) {
   uint32_t FMASK = 0xFFFFFFFFu;
   #pragma unroll
   for (int offset = WS >> 1; offset > 0; offset >>= 1) {
       // __shfl_down_sync: Get value from thread (current_thread_id + offset)
       // Creates a tree reduction pattern where only thread 0 needs the final result
       float other = __shfl_down_sync(FMASK, val, offset);
       val = fmaxf(val, other); // Keep the maximum of the two values
   }
   // Final result is only valid in thread 0
}

/**
 * Finds the minimum value and its associated index across all threads in a warp
 * This is an "argmin" operation that returns both the minimum value and which thread had it
 * 
 * @tparam WS Size of the warp
 * @param val Reference to input/output value - will contain minimum on return
 * @param idx Reference to input/output index - will contain index of minimum on return
 * 
 * Uses __shfl_down_sync with tree reduction pattern (result only valid in thread 0)
 * Tie-breaking: when values are equal, prefer the smaller index
 */
template<const int WS>
__device__ __forceinline__ void warp_reduce_packed_min(float &val, int &idx) {
    uint32_t FMASK = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = WS >> 1; offset >= 1; offset >>= 1) {
        // Get value and index from thread (current_thread_id + offset)
        float temp_val = __shfl_down_sync(FMASK, val, offset);
        int   temp_idx = __shfl_down_sync(FMASK, idx, offset);
        
        // Update if we found a smaller value, OR if values are equal but index is smaller
        // This ensures deterministic behavior when multiple threads have the same minimum value
        if (temp_val < val || (temp_val == val && temp_idx < idx)) {
            val = temp_val;
            idx = temp_idx;
        }
    }
    // Result (minimum value and its index) is only valid in thread 0
}

/**
 * Specialized version of packed minimum reduction for NSGA-III niching process
 * Finds minimum value along with both individual index and column index
 * 
 * @tparam WS Size of the warp
 * @param val Reference to input/output value
 * @param indiv_idx Reference to input/output individual index
 * @param col_idx Reference to input/output column index (represents column in d_rho_last_niches)
 * 
 * Note: There's a bug in the original code - it uses 'temp_idx' in the comparison
 * but should use 'temp_indiv_idx'. This has been corrected in the comment.
 */
template<const int WS>
__device__ __forceinline__ void warp_reduce_packed_min(float &val, int &indiv_idx, int &col_idx) {
    uint32_t FMASK = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = WS >> 1; offset >= 1; offset >>= 1) {
        // Shuffle all three values from the partner thread
        float temp_val       = __shfl_down_sync(FMASK, val,       offset);
        int   temp_indiv_idx = __shfl_down_sync(FMASK, indiv_idx, offset);
        int   temp_col_idx   = __shfl_down_sync(FMASK, col_idx,   offset);
        
        // BUG FIX: Original code used 'temp_idx' but should be 'temp_indiv_idx'
        // Update if we found a smaller value, OR if equal value but smaller individual index
        if (temp_val < val || (temp_val == val && temp_indiv_idx < indiv_idx)) {
            val       = temp_val;
            indiv_idx = temp_indiv_idx;
            col_idx   = temp_col_idx;
        }
    }
    // Results are only valid in thread 0
}

/**
 * SHUFFLE INSTRUCTION DETAILED EXPLANATION:
 * 
 * __shfl_down_sync(FMASK, val, offset):
 * - Each thread receives 'val' from thread (threadIdx.x + offset)
 * - Threads where (threadIdx.x + offset >= warp_size) receive undefined values
 * - Creates a "down-shift" communication pattern
 * - Used in tree reductions where only thread 0 needs the final result
 * - More efficient for reductions that don't need the result in all threads
 * 
 * __shfl_xor_sync(FMASK, val, offset):
 * - Each thread receives 'val' from thread (threadIdx.x XOR offset)
 * - Creates a butterfly network communication pattern
 * - Thread pairs exchange values symmetrically
 * - Used when all threads need the final result (like broadcast reductions)
 * - Example: with offset=1, thread 0↔1, thread 2↔3, thread 4↔5, etc.
 * 
 * Both functions require:
 * - FMASK: Specifies which threads participate (0xFFFFFFFFu = all threads)
 * - Proper synchronization to avoid race conditions
 * - All participating threads must execute the same shuffle instruction
 */