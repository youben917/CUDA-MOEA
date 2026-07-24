#pragma once

#include <cublas_v2.h>

#include <iostream>
#include <string>
#include <fstream>  // Add this for std::ofstream
#include <vector>   // Add this for std::vector

#include <cstdio>
#include <cstdint>   // for uint32_t, uint64_t
#include <cstring>   // for std::memcpy
#include <cstdlib>


#define CUDA_CHECK(err) \
    do { \
        cudaError_t error = (err); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

  
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = (call); \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "CUSOLVER error %s:%d code %d\n", __FILE__, __LINE__, (int)status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static float cuda_timecounter(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

#define CUBLAS_CHECK(status)                                                        \
    do {                                                                            \
        cublasStatus_t _status = (status);                                          \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "CUBLAS error: %s at %s:%d\n",                          \
                    cublasGetErrorString(_status), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN_ERROR";
    }
}

size_t get_device_mem_threshold(int device);

// ================================================================================================== //
// ============================================================================
// Template function with default stream parameter
// ============================================================================
template<typename T>
void saveDeviceArrayToBin(
    const std::string& array_name, 
    T* device_ptr, 
    size_t length,
    bool enable,
    cudaStream_t stream = 0,
    const char* file = nullptr, 
    int line = 0
) {
    if (!enable) return;
    
    // Synchronize the stream to ensure all prior operations are complete
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        std::cerr << "[saveDeviceArrayToBin] cudaStreamSynchronize failed";
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << " - " << cudaGetErrorString(sync_err) << std::endl;
        return;
    }
    
    // Allocate host memory and copy from device
    std::vector<T> host_data(length);
    cudaError_t err = cudaMemcpy(host_data.data(), device_ptr, 
                                  length * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[saveDeviceArrayToBin] cudaMemcpy failed";
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << " - " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Write to binary file
    std::string filename = array_name + ".bin";
    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file) {
        std::cerr << "[saveDeviceArrayToBin] Cannot create file: " << filename;
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << std::endl;
        return;
    }

    out_file.write(reinterpret_cast<const char*>(host_data.data()), 
                   host_data.size() * sizeof(T));
    out_file.close();
}

// ============================================================================
// Variadic preprocessor helper: supports both 4 and 5 arguments
// ============================================================================
// Helper to count arguments
#define GET_INST_5(_1, _2, _3, _4, _5, NAME, ...) NAME

// Helper with 4 arguments (stream defaults to 0)
#define SAVE_DEVICE_ARRAY_4(array_name, device_ptr, length, enable) \
    saveDeviceArrayToBin(array_name, device_ptr, length, enable, 0, __FILE__, __LINE__)

// Helper with 5 arguments (explicit stream)
#define SAVE_DEVICE_ARRAY_5(array_name, device_ptr, length, enable, stream) \
    saveDeviceArrayToBin(array_name, device_ptr, length, enable, stream, __FILE__, __LINE__)

// Unified helper: automatically selects correct version based on argument count
#define saveDeviceArrayToBin(...) \
    GET_INST_5(__VA_ARGS__, SAVE_DEVICE_ARRAY_5, SAVE_DEVICE_ARRAY_4)(__VA_ARGS__)
    
// ================================================================================================== //
// ================================================================================================== //
// ========== saveHostVectorToBin with debug info and enable flag ==========
template<typename T>
void saveHostVectorToBin(
    const std::string& array_name, 
    const std::vector<T>& vec, 
    size_t length, 
    bool enable,
    const char* file = nullptr,
    int line = 0
) {
    // If enable is false, skip execution
    if (!enable) return;
    
    if (vec.size() < length) {
        std::cerr << "[saveHostVectorToBin] Vector size (" << vec.size()
                  << ") < requested length (" << length << ")";
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << std::endl;
        return;
    }

    std::string filename = array_name + ".bin";
    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file) {
        std::cerr << "[saveHostVectorToBin] Cannot create file: " << filename;
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << std::endl;
        return;
    }

    out_file.write(reinterpret_cast<const char*>(vec.data()), length * sizeof(T));
    out_file.close();
}

// ================================================================================================== //
// ========== saveHostIntToBin with debug info and enable flag ==========
template<typename T>
void saveHostIntToBin(
    const std::string& array_name, 
    const T& value, 
    bool enable,
    const char* file = nullptr,
    int line = 0
) {
    // If enable is false, skip execution
    if (!enable) return;

    std::string filename = array_name + ".bin";
    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file) {
        std::cerr << "[saveHostIntToBin] Cannot create file: " << filename;
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << std::endl;
        return;
    }

    out_file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out_file.good()) {
        std::cerr << "[saveHostIntToBin] Write failed for file: " << filename;
        if (file) std::cerr << " at " << file << ":" << line;
        std::cerr << std::endl;
    }
    out_file.close();
}
// =========================================================================================//
static inline unsigned long long host_pack_float_int(float val, int idx) {
    uint32_t val_bits;
    std::memcpy(&val_bits, &val, sizeof(float));
    uint32_t idx_bits = static_cast<uint32_t>(idx);
    return (static_cast<unsigned long long>(val_bits) << 32) | idx_bits;
}
