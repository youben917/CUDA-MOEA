#pragma once

#include "gpu_adaptation.cuh"
#include "cuda_utils.cuh"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using ull = unsigned long long;
using ll  = long long;
using uint = unsigned int;

// Data type size constants
constexpr size_t HALF_SIZE   = sizeof(half);
constexpr size_t FLOAT_SIZE  = sizeof(float);
constexpr size_t FLOAT2_SIZE = sizeof(float2);
constexpr size_t FLOAT4_SIZE = sizeof(float4);
constexpr size_t INT_SIZE    = sizeof(int);
constexpr size_t INT2_SIZE   = sizeof(int2);
constexpr size_t LL_SIZE     = sizeof(long long);
constexpr size_t ULL_SIZE    = sizeof(unsigned long long);
constexpr size_t UINT_SIZE   = sizeof(unsigned int);

// GPU architecture configurations (select one)
// // RTX 4060
// constexpr int SM_COUNT           = gpu::models::rtx4060::SM_COUNT;
// constexpr int MAX_THREADS_PER_SM = gpu::models::rtx4060::MAX_THREADS_PER_SM;

// // RTX 4090
// constexpr int SM_COUNT           = gpu::models::rtx4090::SM_COUNT;
// constexpr int MAX_THREADS_PER_SM = gpu::models::rtx4090::MAX_THREADS_PER_SM;

// RTX 5090
// constexpr int SM_COUNT           = gpu::models::rtx5090::SM_COUNT;
// constexpr int MAX_THREADS_PER_SM = gpu::models::rtx5090::MAX_THREADS_PER_SM;

// RTX 6000 pro
constexpr int SM_COUNT           = gpu::models::rtx6000pro::SM_COUNT;
constexpr int MAX_THREADS_PER_SM = gpu::models::rtx6000pro::MAX_THREADS_PER_SM;

// // H800
// constexpr int SM_COUNT           = gpu::models::h800::SM_COUNT;
// constexpr int MAX_THREADS_PER_SM = gpu::models::h800::MAX_THREADS_PER_SM;

// Warp size and maximum objectives
constexpr int WS    = 32;  // NVIDIA warp size
constexpr int MAX_M = 8;   // Maximum number of objectives supported

constexpr float FEASIBLE_CV  = 0.0f;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
