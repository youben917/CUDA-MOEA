#pragma once

#include <memory>

#include <cuda_runtime.h>

namespace cuda_moea {

namespace detail {
struct ContextAccess;
}

struct CudaConfig {
    int device_id = 0;
    float evaluation_pool_ratio = 0.25f;
    float execution_pool_ratio = 0.75f;
    int memory_pool_policy = 1;
    unsigned long long seed = 2887ULL;
    bool enable_warmup = true;
};

class CudaContext {
public:
    explicit CudaContext(const CudaConfig& config = {});
    ~CudaContext();

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&&) = delete;
    CudaContext& operator=(CudaContext&&) = delete;

    cudaStream_t evaluation_stream() const noexcept;
    cudaStream_t execution_stream() const noexcept;
    cudaMemPool_t evaluation_pool() const noexcept;
    cudaMemPool_t execution_pool() const noexcept;

    unsigned long long random_seed() const noexcept;
    unsigned long long& random_offset() noexcept;

    void wait_evaluation_on_execution();
    void wait_execution_on_evaluation();
    void synchronize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend struct detail::ContextAccess;
};

} // namespace cuda_moea
