#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA

#include <device_launch_parameters.h>

#include "backends/cuda/cuda_runtime.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

#include "backends/device_runtime.hpp"

namespace EC::Dev
{





bool CudaDeviceRuntime::isAvailable(int idx) const{
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return false;
    return idx >= 0 && idx < n;
}

void CudaDeviceRuntime::setCurrentDevice(DI dev)  {
    cudaCheck(cudaSetDevice(dev.id()), "cudaSetDevice failed");
}

void CudaDeviceRuntime::synchronizeDevice(DI dev)  {
    setCurrentDevice(dev);
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

void* CudaDeviceRuntime::allocate(DI dev, size_t bytes, MemoryType kind)  {
    setCurrentDevice(dev);
    void* ptr = nullptr;
    switch (kind) {
        case MemoryType::Device:
            cudaCheck(cudaMalloc(&ptr, bytes), "cudaMalloc failed");
            break;
        case MemoryType::Unified:
            cudaCheck(cudaMallocManaged(&ptr, bytes), "cudaMallocManaged failed");
            break;
        case MemoryType::PinnedHost:
            cudaCheck(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc failed");
            break;
        case MemoryType::Host:
            throw std::runtime_error("Host memory should not be allocated by CudaDeviceRuntime");
    }
    return ptr;
}

void CudaDeviceRuntime::deallocate(DI dev, void* ptr, MemoryType kind) {
    if (!ptr) return;
    setCurrentDevice(dev);
    switch (kind) {
        case MemoryType::Device:
        case MemoryType::Unified:
            cudaCheck(cudaFree(ptr), "cudaFree failed");
            break;
        case MemoryType::PinnedHost:
            cudaCheck(cudaFreeHost(ptr), "cudaFreeHost failed");
            break;
        case MemoryType::Host:
            throw std::runtime_error("Host memory should not be deallocated by CudaDeviceRuntime");
    }
}

void* CudaDeviceRuntime::allocateAsync(DI dev, size_t bytes, MemoryType kind, IStream stream) {
    setCurrentDevice(dev);
    if (kind != MemoryType::Device) {
        return allocate(dev, bytes, kind);
    }
#if CUDART_VERSION >= 11020
    void* ptr = nullptr;
    cudaCheck(cudaMallocAsync(&ptr, bytes, toCudaStream(stream)), "cudaMallocAsync failed");
    return ptr;
#else
    return allocate(dev, bytes, kind);
#endif
}

void CudaDeviceRuntime::deallocateAsync(DI dev, void* ptr, MemoryType kind, IStream stream) {
    if (!ptr) return;
    setCurrentDevice(dev);
    if (kind != MemoryType::Device) {
        deallocate(dev, ptr, kind);
        return;
    }
#if CUDART_VERSION >= 11020
    cudaCheck(cudaFreeAsync(ptr, toCudaStream(stream)), "cudaFreeAsync failed");
#else
    deallocate(dev, ptr, kind);
#endif
}

IStream CudaDeviceRuntime::createStream(DI dev, int priority) {
    setCurrentDevice(dev);
    cudaStream_t s = nullptr;

    int least = 0, greatest = 0;
    cudaCheck(cudaDeviceGetStreamPriorityRange(&least, &greatest),
                "cudaDeviceGetStreamPriorityRange failed");

    int cuda_priority = 0;
    if (priority < 0) cuda_priority = greatest;
    else if (priority > 0) cuda_priority = least;
    else cuda_priority = 0;

    cudaCheck(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, cuda_priority),
                "cudaStreamCreateWithPriority failed");
    return IStream(reinterpret_cast<void*>(s), dev);
}

void CudaDeviceRuntime::destroyStream(IStream stream) {
    if (!stream.valid()) return;
    auto dev = stream.device();
    setCurrentDevice(dev);
    cudaCheck(cudaStreamDestroy(toCudaStream(stream)), "cudaStreamDestroy failed");
}

void CudaDeviceRuntime::synchronizeStream(IStream stream)  {
    cudaCheck(cudaStreamSynchronize(toCudaStream(stream)), "cudaStreamSynchronize failed");
}

IEvent CudaDeviceRuntime::createEvent(DI dev, bool timing )  {
    setCurrentDevice(dev);
    cudaEvent_t ev = nullptr;
    unsigned flags = timing ? cudaEventDefault : cudaEventDisableTiming;
    cudaCheck(cudaEventCreateWithFlags(&ev, flags), "cudaEventCreateWithFlags failed");
    return IEvent(reinterpret_cast<void*>(ev), dev);
}

void CudaDeviceRuntime::destroyEvent(IEvent event) {
    if (!event.valid()) return;
    auto dev = event.device();
    setCurrentDevice(dev);
    cudaCheck(cudaEventDestroy(toCudaEvent(event)), "cudaEventDestroy failed");
}

void CudaDeviceRuntime::recordEvent(IEvent event, IStream stream) {
    cudaCheck(cudaEventRecord(toCudaEvent(event), toCudaStream(stream)),
                "cudaEventRecord failed");
}

bool CudaDeviceRuntime::queryEvent(IEvent event) {
    auto err = cudaEventQuery(toCudaEvent(event));
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    cudaCheck(err, "cudaEventQuery failed");
    return false;
}

void CudaDeviceRuntime::synchronizeEvent(IEvent event) {
    cudaCheck(cudaEventSynchronize(toCudaEvent(event)), "cudaEventSynchronize failed");
}

void CudaDeviceRuntime::waitEvent(IStream stream, IEvent event) {
    cudaCheck(cudaStreamWaitEvent(toCudaStream(stream), toCudaEvent(event), 0),
                "cudaStreamWaitEvent failed");
}

void CudaDeviceRuntime::memcpyAsync(void* dst, DI dst_dev,
                    const void* src, DI src_dev,
                    size_t bytes, IStream stream) {
    (void)src_dev;
    setCurrentDevice(dst_dev);

    cudaMemcpyKind kind;
    if (src_dev.type() == DeviceType::CPU && dst_dev.type() == DeviceType::CUDA) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_dev.type() == DeviceType::CUDA && dst_dev.type() == DeviceType::CPU) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_dev.type() == DeviceType::CUDA && dst_dev.type() == DeviceType::CUDA) {
        kind = cudaMemcpyDeviceToDevice;
    } else {
        throw std::runtime_error("unsupported memcpy direction for cuda runtime");
    }

    cudaCheck(cudaMemcpyAsync(dst, src, bytes, kind, toCudaStream(stream)),
                "cudaMemcpyAsync failed");
}

}

#endif