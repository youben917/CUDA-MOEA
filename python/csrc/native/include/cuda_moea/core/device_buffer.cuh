#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

namespace cuda_moea {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    DeviceBuffer(
        std::size_t count,
        cudaMemPool_t pool,
        cudaStream_t stream)
    {
        allocate(count, pool, stream);
    }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        move_from(std::move(other));
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    void allocate(
        std::size_t count,
        cudaMemPool_t pool,
        cudaStream_t stream)
    {
        release();
        if (count == 0) return;

        T* ptr = nullptr;
        const cudaError_t status = cudaMallocFromPoolAsync(
            reinterpret_cast<void**>(&ptr),
            count * sizeof(T),
            pool,
            stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        data_ = ptr;
        size_ = count;
        stream_ = stream;
    }

    void release() noexcept {
        if (data_) {
            cudaFreeAsync(data_, stream_);
        }
        data_ = nullptr;
        size_ = 0;
        stream_ = nullptr;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return data_ == nullptr; }

private:
    void move_from(DeviceBuffer&& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        stream_ = other.stream_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.stream_ = nullptr;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    cudaStream_t stream_ = nullptr;
};

} // namespace cuda_moea
