#pragma once

#include "cuda_vecops.cuh"

__device__ __forceinline__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ float atomicMinf32(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ void atomicMinFloatInt(unsigned long long* address, float new_val, int new_idx) {
    unsigned long long* addr = address;
    unsigned long long old   = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        float current_val;
        int current_idx;
        constexpr int KWarpSize = 32;
        unpackFloatIndex<KWarpSize>(assumed, current_val, current_idx);
        if (new_val < current_val || (new_val == current_val && new_idx < current_idx)) {
            unsigned long long new_packed = packFloatIndex<KWarpSize>(new_val, new_idx);
            old = atomicCAS(addr, assumed, new_packed);
        } else {
            break;
        }
    } while (assumed != old);
}

