#pragma once

#include <memory>
#include <unordered_map>

#include "tensor/device.hpp"
#include "abstract.hpp"
#include "device_runtime.hpp"


namespace EC{
namespace Dev{
    

struct DeviceManager {
public:

    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;
    
    static DeviceManager& get_instance() {
        static DeviceManager dm;
        return dm;
    }

    void registerRuntime(std::unique_ptr<IDeviceRuntime> runtime) {
        runtimes_[runtime->device().type()] = std::move(runtime);
    }

    IDeviceRuntime& runtime(DeviceType dev) {
        auto it = runtimes_.find(dev);
        if (it == runtimes_.end()) {
            throw std::runtime_error("runtime not registered");
        }
        return *(it->second);
    }

    void* allocate(DI dev, size_t bytes, MemoryType kind) {
        return runtime(dev.type()).allocate(dev, bytes, kind);
    }

    void deallocate(DI dev, void* ptr,  MemoryType kind) {
        runtime(dev.type()).deallocate(dev, ptr, kind);
    }

    void* allocateAsync(DI dev, size_t bytes, MemoryType kind, StreamHandle stream) {
        return runtime(dev.type()).allocateAsync(dev, bytes, kind, stream);
    }

    void deallocateAsync(DI dev, void* ptr, MemoryType kind, StreamHandle stream) {
        runtime(dev.type()).deallocateAsync(dev, ptr, kind, stream);
    }

    StreamHandle createStream(DI dev, int priority = 0) {
        return runtime(dev.type()).createStream(dev, priority);
    }

    void destroyStream(StreamHandle stream) {
        return runtime(stream.device().type()).destroyStream(stream);
    }

    EventHandle createEvent(DI dev, bool timing = false) {
        return runtime(dev.type()).createEvent(dev, timing);
    }

    void recordEvent(EventHandle ev, StreamHandle stream) {
        runtime(ev.device().type()).recordEvent(ev, stream);
    }

    bool queryEvent(EventHandle ev) {
        return runtime(ev.device().type()).queryEvent(ev);
    }

    void destroyEvent(EventHandle ev) {
        return runtime(ev.device().type()).destroyEvent(ev);
    }

    void waitEvent(StreamHandle stream, EventHandle ev) {
        runtime(stream.device().type()).waitEvent(stream, ev);
    }

    void memcpyAsync(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, StreamHandle stream) {
        runtime(dst_dev.type()).memcpyAsync(dst, dst_dev, src, src_dev, bytes, stream);
    }

    void synchronize(StreamHandle stream) {
        runtime(stream.device().type()).synchronizeStream(stream);
    }

    void synchronize(EventHandle ev) {
        runtime(ev.device().type()).synchronizeEvent(ev);
    }

    void synchronize(DI dev) {
        runtime(dev.type()).synchronizeDevice(dev);
    }

private:
    DeviceManager() = default;
    std::unordered_map<DeviceType, std::unique_ptr<IDeviceRuntime>> runtimes_;
};


}
using DM = Dev::DeviceManager;
}
