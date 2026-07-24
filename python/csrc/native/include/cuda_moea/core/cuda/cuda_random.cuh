#pragma once

#include <curand.h>
#include "cuda_globals.cuh"
// =============================================================================================== //
// =============================================================================================== //
void launch_random_uniform_kernel(
    float* d_output,                 
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset, 
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_boolean_kernel(
    int* d_output,
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_int_kernel(
    int* d_output,
    int count,
    int lb,
    int ub,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_float_kernel(
    float* d_output,                  
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset, 
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_float2_kernel(
    float2* d_output,
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_float4_kernel(
    float4* d_output,
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_int2_kernel(
    int2* d_output,
    int count,
    int lb,
    int ub,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_bounded_uniform_kernel(
    float* d_output,
    int count,
    float lb,
    float ub,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_bounded_float2_kernel(
    float2* d_output,
    int count,
    float lb,
    float ub,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);    
// =============================================================================================== //
void launch_random_uint32_kernel(
    unsigned int* d_output,
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
// =============================================================================================== //
void launch_random_uint2_kernel(
    uint2* d_output,
    int count,
    ull rnd_seed,
    ull& glb_rnd_offset,
    const int block_size,
    cudaStream_t stream);
