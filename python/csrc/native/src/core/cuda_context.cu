#include "cuda_moea/core/cuda_context.cuh"

#include <mutex>
#include <stdexcept>
#include <utility>

#include "cuda_moea/core/cuda/cuda_utils.cuh"
#include "cuda_moea/core/cuda/cuda_warmup.cuh"
#include "cuda_moea/selection/nsga3/kernels/normalization.cuh"
#include "cuda_moea/core/context_access.cuh"

namespace cuda_moea {
namespace {

std::mutex library_mutex;
int library_users = 0;

void acquire_cuda_libraries()
{
    std::lock_guard<std::mutex> lock(library_mutex);
    if (library_users == 0) {
        try {
            CuBlasManager::initialize();
            CuSolverManager::initialize();
        } catch (...) {
            CuSolverManager::cleanup();
            CuBlasManager::cleanup();
            throw;
        }
    }
    ++library_users;
}

void release_cuda_libraries()
{
    std::lock_guard<std::mutex> lock(library_mutex);
    if (library_users > 0 && --library_users == 0) {
        cleanup_normalization_handle_pool();
        CuSolverManager::cleanup();
        CuBlasManager::cleanup();
    }
}

} // namespace

CudaContext::CudaContext(const CudaConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
    if (config.evaluation_pool_ratio <= 0.0f ||
        config.execution_pool_ratio <= 0.0f) {
        throw std::invalid_argument("CUDA memory-pool ratios must be positive");
    }

    CUDA_CHECK(cudaSetDevice(config.device_id));
    bool libraries_acquired = false;
    try {
        acquire_cuda_libraries();
        libraries_acquired = true;
        impl_->streams.create();
        impl_->pools.create(
            impl_->streams,
            config.evaluation_pool_ratio,
            config.execution_pool_ratio,
            config.memory_pool_policy,
            config.device_id);
        if (config.enable_warmup) {
            warmup_cuda_libraries(*this);
        }
    } catch (...) {
        impl_->pools.destroy();
        impl_->streams.destroy();
        if (libraries_acquired) {
            release_cuda_libraries();
        }
        throw;
    }
}

CudaContext::~CudaContext()
{
    if (!impl_) return;
    cudaSetDevice(impl_->device_id);
    if (impl_->streams.fcal_stream) {
        cudaStreamSynchronize(impl_->streams.fcal_stream);
    }
    if (impl_->streams.exec_stream) {
        cudaStreamSynchronize(impl_->streams.exec_stream);
    }
    release_cuda_libraries();
    impl_->pools.destroy();
    impl_->streams.destroy();
}

cudaStream_t CudaContext::evaluation_stream() const noexcept {
    return impl_->streams.fcal_stream;
}

cudaStream_t CudaContext::execution_stream() const noexcept {
    return impl_->streams.exec_stream;
}

cudaMemPool_t CudaContext::evaluation_pool() const noexcept {
    return impl_->pools.fcal_pool;
}

cudaMemPool_t CudaContext::execution_pool() const noexcept {
    return impl_->pools.exec_pool;
}

unsigned long long CudaContext::random_seed() const noexcept {
    return impl_->random.get_seed();
}

unsigned long long& CudaContext::random_offset() noexcept {
    return impl_->random.get_offset_ref();
}

void CudaContext::wait_evaluation_on_execution() {
    impl_->streams.wait_fcalstream_done_and_execute(impl_->streams.exec_stream);
}

void CudaContext::wait_execution_on_evaluation() {
    impl_->streams.wait_execstream_done_and_execute(impl_->streams.fcal_stream);
}

void CudaContext::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(impl_->streams.fcal_stream));
    CUDA_CHECK(cudaStreamSynchronize(impl_->streams.exec_stream));
}

} // namespace cuda_moea
