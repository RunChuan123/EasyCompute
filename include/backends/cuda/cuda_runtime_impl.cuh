#ifdef EC_ENABLE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include "device_runtime.hpp"

namespace EC::Dev
{


static inline cudaStream_t toCudaStream(StreamHandle s) {
    return reinterpret_cast<cudaStream_t>(s.impl());
}

static inline cudaEvent_t toCudaEvent(EventHandle e) {
    return reinterpret_cast<cudaEvent_t>(e.impl());
}

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

class CudaDeviceRuntime final : public IDeviceRuntime {
public:
    DeviceType type() const override { return DeviceType::CUDA; }

    int deviceCount() const override {
        int n = 0;
        cudaCheck(cudaGetDeviceCount(&n), "cudaGetDeviceCount failed");
        return n;
    }

    bool isAvailable(int idx) const override {
        int n = 0;
        cudaError_t err = cudaGetDeviceCount(&n);
        if (err != cudaSuccess) return false;
        return idx >= 0 && idx < n;
    }

    void setCurrentDevice(Device dev) override {
        cudaCheck(cudaSetDevice(dev.index), "cudaSetDevice failed");
    }

    void synchronizeDevice(Device dev) override {
        setCurrentDevice(dev);
        cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    void* allocate(Device dev, size_t bytes, MemoryKind kind) override {
        setCurrentDevice(dev);
        void* ptr = nullptr;
        switch (kind) {
            case MemoryKind::Device:
                cudaCheck(cudaMalloc(&ptr, bytes), "cudaMalloc failed");
                break;
            case MemoryKind::Unified:
                cudaCheck(cudaMallocManaged(&ptr, bytes), "cudaMallocManaged failed");
                break;
            case MemoryKind::HostPinned:
                cudaCheck(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc failed");
                break;
            case MemoryKind::Host:
                throw std::runtime_error("Host memory should not be allocated by CudaDeviceRuntime");
        }
        return ptr;
    }

    void deallocate(Device dev, void* ptr, MemoryKind kind) override {
        if (!ptr) return;
        setCurrentDevice(dev);
        switch (kind) {
            case MemoryKind::Device:
            case MemoryKind::Unified:
                cudaCheck(cudaFree(ptr), "cudaFree failed");
                break;
            case MemoryKind::HostPinned:
                cudaCheck(cudaFreeHost(ptr), "cudaFreeHost failed");
                break;
            case MemoryKind::Host:
                throw std::runtime_error("Host memory should not be deallocated by CudaDeviceRuntime");
        }
    }

    void* allocateAsync(Device dev, size_t bytes, MemoryKind kind, StreamHandle stream) override {
        setCurrentDevice(dev);
        if (kind != MemoryKind::Device) {
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

    void deallocateAsync(Device dev, void* ptr, MemoryKind kind, StreamHandle stream) override {
        if (!ptr) return;
        setCurrentDevice(dev);
        if (kind != MemoryKind::Device) {
            deallocate(dev, ptr, kind);
            return;
        }
#if CUDART_VERSION >= 11020
        cudaCheck(cudaFreeAsync(ptr, toCudaStream(stream)), "cudaFreeAsync failed");
#else
        deallocate(dev, ptr, kind);
#endif
    }

    StreamHandle createStream(Device dev, int priority = 0) override {
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
        return StreamHandle(reinterpret_cast<void*>(s), dev);
    }

    void destroyStream(StreamHandle stream) override {
        if (!stream.valid()) return;
        auto dev = stream.device();
        setCurrentDevice(dev);
        cudaCheck(cudaStreamDestroy(toCudaStream(stream)), "cudaStreamDestroy failed");
    }

    void synchronizeStream(StreamHandle stream) override {
        cudaCheck(cudaStreamSynchronize(toCudaStream(stream)), "cudaStreamSynchronize failed");
    }

    EventHandle createEvent(Device dev, bool timing = false) override {
        setCurrentDevice(dev);
        cudaEvent_t ev = nullptr;
        unsigned flags = timing ? cudaEventDefault : cudaEventDisableTiming;
        cudaCheck(cudaEventCreateWithFlags(&ev, flags), "cudaEventCreateWithFlags failed");
        return EventHandle(reinterpret_cast<void*>(ev), dev);
    }

    void destroyEvent(EventHandle event) override {
        if (!event.valid()) return;
        auto dev = event.device();
        setCurrentDevice(dev);
        cudaCheck(cudaEventDestroy(toCudaEvent(event)), "cudaEventDestroy failed");
    }

    void recordEvent(EventHandle event, StreamHandle stream) override {
        cudaCheck(cudaEventRecord(toCudaEvent(event), toCudaStream(stream)),
                  "cudaEventRecord failed");
    }

    bool queryEvent(EventHandle event) override {
        auto err = cudaEventQuery(toCudaEvent(event));
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        cudaCheck(err, "cudaEventQuery failed");
        return false;
    }

    void synchronizeEvent(EventHandle event) override {
        cudaCheck(cudaEventSynchronize(toCudaEvent(event)), "cudaEventSynchronize failed");
    }

    void waitEvent(StreamHandle stream, EventHandle event) override {
        cudaCheck(cudaStreamWaitEvent(toCudaStream(stream), toCudaEvent(event), 0),
                  "cudaStreamWaitEvent failed");
    }

    void memcpyAsync(void* dst, Device dst_dev,
                     const void* src, Device src_dev,
                     size_t bytes, StreamHandle stream) override {
        (void)src_dev;
        setCurrentDevice(dst_dev);

        cudaMemcpyKind kind;
        if (src_dev.type == DeviceType::CPU && dst_dev.type == DeviceType::CUDA) {
            kind = cudaMemcpyHostToDevice;
        } else if (src_dev.type == DeviceType::CUDA && dst_dev.type == DeviceType::CPU) {
            kind = cudaMemcpyDeviceToHost;
        } else if (src_dev.type == DeviceType::CUDA && dst_dev.type == DeviceType::CUDA) {
            kind = cudaMemcpyDeviceToDevice;
        } else {
            throw std::runtime_error("unsupported memcpy direction for cuda runtime");
        }

        cudaCheck(cudaMemcpyAsync(dst, src, bytes, kind, toCudaStream(stream)),
                  "cudaMemcpyAsync failed");
    }
};

void registerBackends() {
    DeviceManager::instance().registerRuntime(std::make_unique<CudaDeviceRuntime>());
    // 以后还能注册 CpuDeviceRuntime / AscendDeviceRuntime
}

#endif