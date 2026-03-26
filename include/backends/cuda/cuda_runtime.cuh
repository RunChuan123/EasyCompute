#pragma once
#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include "device_runtime.hpp"
#include "device_manager.hpp"

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
    // CudaDeviceRuntime(DI dev)
    DI device() const override { return DI::cuda(); }

    int deviceCount() const override {
        int n = 0;
        cudaCheck(cudaGetDeviceCount(&n), "cudaGetDeviceCount failed");
        return n;
    }

    bool isAvailable(int idx) const override ;

    void setCurrentDevice(DI dev) override;

    void synchronizeDevice(DI dev) override ;
    void* allocate(DI dev, size_t bytes, MemoryType kind) override ;

    void deallocate(DI dev, void* ptr, MemoryType kind) override ;

    void* allocateAsync(DI dev, size_t bytes, MemoryType kind, StreamHandle stream) override ;

    void deallocateAsync(DI dev, void* ptr, MemoryType kind, StreamHandle stream) override ;

    StreamHandle createStream(DI dev, int priority = 0) override;

    void destroyStream(StreamHandle stream) override;

    void synchronizeStream(StreamHandle stream) override;

    EventHandle createEvent(DI dev, bool timing = false) override;

    void destroyEvent(EventHandle event) override;

    void recordEvent(EventHandle event, StreamHandle stream) override;

    bool queryEvent(EventHandle event) override ;

    void synchronizeEvent(EventHandle event) override;

    void waitEvent(StreamHandle stream, EventHandle event) override ;

    void memcpyAsync(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, StreamHandle stream) override ;
};

void registerBackends() {
    DM::get_instance().registerRuntime(std::make_unique<CudaDeviceRuntime>());
    // 以后还能注册 CpuDeviceRuntime / AscendDeviceRuntime
}
}

#endif