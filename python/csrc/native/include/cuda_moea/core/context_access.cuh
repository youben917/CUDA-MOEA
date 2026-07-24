#pragma once

#include "cuda_moea/core/cuda_context.cuh"

#include "cuda_moea/core/cuda/cuda_manager.cuh"

namespace cuda_moea {

struct CudaContext::Impl {
    cudaStreamSync streams;
    cudaMemPools pools;
    RNDManager random;
    int device_id = 0;

    explicit Impl(const CudaConfig& config)
        : random(config.seed, 0), device_id(config.device_id) {}
};

namespace detail {

struct ContextAccess {
    static CudaContext::Impl& get(CudaContext& context) {
        return *context.impl_;
    }

    static const CudaContext::Impl& get(const CudaContext& context) {
        return *context.impl_;
    }
};

} // namespace detail
} // namespace cuda_moea
