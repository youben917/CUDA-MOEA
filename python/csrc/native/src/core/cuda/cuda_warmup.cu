#include "cuda_moea/core/cuda/cuda_warmup.cuh"

#include <algorithm>
#include <stdexcept>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuda_moea/core/cuda/cuda_utils.cuh"
#include "cuda_moea/core/cuda_context.cuh"
#include "cuda_moea/core/device_buffer.cuh"

namespace cuda_moea {
namespace {

class ScopedCublasHandle {
public:
    explicit ScopedCublasHandle(cudaStream_t stream)
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream));
        CUBLAS_CHECK(
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    }

    ~ScopedCublasHandle()
    {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    ScopedCublasHandle(const ScopedCublasHandle&) = delete;
    ScopedCublasHandle& operator=(const ScopedCublasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

class ScopedCusolverHandle {
public:
    explicit ScopedCusolverHandle(cudaStream_t stream)
    {
        CUSOLVER_CHECK(cusolverDnCreate(&handle_));
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream));
    }

    ~ScopedCusolverHandle()
    {
        if (handle_) {
            cusolverDnDestroy(handle_);
        }
    }

    ScopedCusolverHandle(const ScopedCusolverHandle&) = delete;
    ScopedCusolverHandle& operator=(const ScopedCusolverHandle&) = delete;

    cusolverDnHandle_t get() const noexcept { return handle_; }

private:
    cusolverDnHandle_t handle_ = nullptr;
};

} // namespace

void warmup_cuda_libraries(CudaContext& context)
{
    constexpr int matrix_size = 4;
    constexpr int element_count = matrix_size * matrix_size;

    const cudaStream_t stream = context.execution_stream();
    const cudaMemPool_t pool = context.execution_pool();

    DeviceBuffer<float> matrix(element_count, pool, stream);
    DeviceBuffer<float> singular_values(matrix_size, pool, stream);
    DeviceBuffer<float> left_vectors(element_count, pool, stream);
    DeviceBuffer<float> right_vectors(element_count, pool, stream);
    DeviceBuffer<float> input_vector(matrix_size, pool, stream);
    DeviceBuffer<float> output_vector(matrix_size, pool, stream);
    DeviceBuffer<int> solver_info(1, pool, stream);

    float host_matrix[element_count];
    float host_vector[matrix_size];
    for (int i = 0; i < element_count; ++i) {
        host_matrix[i] = static_cast<float>(i + 1);
    }
    std::fill_n(host_vector, matrix_size, 1.0f);

    printf("[Warmup] Starting CUDA libraries warmup...\n"); 
    
    CUDA_CHECK(cudaMemcpyAsync(
        matrix.data(),
        host_matrix,
        sizeof(host_matrix),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        input_vector.data(),
        host_vector,
        sizeof(host_vector),
        cudaMemcpyHostToDevice,
        stream));

    ScopedCublasHandle cublas(stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(
        cublas.get(),
        CUBLAS_OP_N,
        matrix_size,
        matrix_size,
        &alpha,
        matrix.data(),
        matrix_size,
        input_vector.data(),
        1,
        &beta,
        output_vector.data(),
        1));

    ScopedCusolverHandle cusolver(stream);
    int workspace_size = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(
        cusolver.get(),
        matrix_size,
        matrix_size,
        &workspace_size));
    DeviceBuffer<float> workspace(
        static_cast<std::size_t>(workspace_size),
        pool,
        stream);

    CUSOLVER_CHECK(cusolverDnSgesvd(
        cusolver.get(),
        'A',
        'A',
        matrix_size,
        matrix_size,
        matrix.data(),
        matrix_size,
        singular_values.data(),
        left_vectors.data(),
        matrix_size,
        right_vectors.data(),
        matrix_size,
        workspace.data(),
        workspace_size,
        nullptr,
        solver_info.data()));

    int host_solver_info = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &host_solver_info,
        solver_info.data(),
        sizeof(host_solver_info),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (host_solver_info != 0) {
        throw std::runtime_error(
            "CUDA warmup failed: cuSOLVER SVD did not converge");
    }
}

} // namespace cuda_moea
