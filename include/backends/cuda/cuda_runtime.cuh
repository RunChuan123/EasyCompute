#pragma once
#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include "backends/device_runtime.hpp"
#include "backends/device_manager.hpp"

namespace EC::Dev
{


static inline cudaStream_t toCudaStream(IStream s) {
    return reinterpret_cast<cudaStream_t>(s.impl());
}

static inline cudaEvent_t toCudaEvent(IEvent e) {
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

    void* allocateAsync(DI dev, size_t bytes, MemoryType kind, IStream stream) override ;

    void deallocateAsync(DI dev, void* ptr, MemoryType kind, IStream stream) override ;

    IStream createStream(DI dev, int priority = 0) override;

    void destroyStream(IStream stream) override;

    void synchronizeStream(IStream stream) override;

    IEvent createEvent(DI dev, bool timing = false) override;

    void destroyEvent(IEvent event) override;

    void recordEvent(IEvent event, IStream stream) override;

    bool queryEvent(IEvent event) override ;

    void synchronizeEvent(IEvent event) override;

    void waitEvent(IStream stream, IEvent event) override ;

    void memcpyAsync(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, IStream stream) override ;
};

inline void registerCUDABackends() {
    DM::get_instance().registerRuntime(std::make_unique<CudaDeviceRuntime>());
    // 以后还能注册 CpuDeviceRuntime / AscendDeviceRuntime
}
}

#endif