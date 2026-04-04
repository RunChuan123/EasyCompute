#pragma once

#include <memory>
#include <iostream>
#include <unordered_map>

#include "tensor/device.hpp"
#include "abstract.hpp"
#include "device_runtime.hpp"
#include "util/err.hpp"
#include "util/logger.hpp"


namespace EC{
namespace Dev{
    
/**
 * 设备管理层，设备注册到 runtimes，并提供相关接口
 */
struct DeviceManager {
public:

    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;
    
    static DeviceManager& get_instance() {
        static DeviceManager* dm = new DeviceManager();
        return *dm;
    }

    void registerRuntime(std::unique_ptr<IDeviceRuntime> runtime) {
        runtimes_[runtime->device().type()] = std::move(runtime);
        LOG_INFO("runtime: ",runtime->name(),"register success!");
    }

    IDeviceRuntime& runtime(DeviceType dev) {
        if (runtimes_.empty() && runtimes_.bucket_count() == 0) 
            LOG_ERROR("DeviceManager contains no device");
        auto it = runtimes_.find(dev);
        if (it == runtimes_.end()) 
            LOG_WARN("target runtime not registered");
        return *(it->second);
    }


    void* allocate(DI dev, size_t bytes, MemoryType kind) {
        return runtime(dev.type()).allocate(dev, bytes, kind);
    }

    void deallocate(DI dev, void* ptr,  MemoryType kind) {
        runtime(dev.type()).deallocate(dev, ptr, kind);
    }

    void* allocateAsync(DI dev, size_t bytes, MemoryType kind, IStream stream) {
        return runtime(dev.type()).allocateAsync(dev, bytes, kind, stream);
    }

    void deallocateAsync(DI dev, void* ptr, MemoryType kind, IStream stream) {
        runtime(dev.type()).deallocateAsync(dev, ptr, kind, stream);
    }

    IStream createStream(DI dev, int priority = 0) {
        return runtime(dev.type()).createStream(dev, priority);
    }

    void destroyStream(IStream stream) {
        return runtime(stream.device().type()).destroyStream(stream);
    }

    IEvent createEvent(DI dev, bool timing = false) {
        return runtime(dev.type()).createEvent(dev, timing);
    }

    void recordEvent(IEvent ev, IStream stream) {
        runtime(ev.device().type()).recordEvent(ev, stream);
    }

    bool queryEvent(IEvent ev) {
        return runtime(ev.device().type()).queryEvent(ev);
    }

    void destroyEvent(IEvent ev) {
        return runtime(ev.device().type()).destroyEvent(ev);
    }

    void waitEvent(IStream stream, IEvent ev) {
        runtime(stream.device().type()).waitEvent(stream, ev);
    }

    void memcpy(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, IStream stream) {
        if(dst_dev.type() == DeviceType::CUDA || src_dev.type() == DeviceType::CUDA){
            runtime(DeviceType::CUDA).memcpy(dst, dst_dev, src, src_dev, bytes, stream);
        }
        if(false){
            runtime(DeviceType::CPU).memcpy(dst, dst_dev, src, src_dev, bytes, stream);
        }
    }

    void memcpyAsync(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, IStream stream) {
        if(dst_dev.type() == DeviceType::CUDA || src_dev.type() == DeviceType::CUDA){
            runtime(DeviceType::CUDA).memcpyAsync(dst, dst_dev, src, src_dev, bytes, stream);
        }else{
            runtime(DeviceType::CPU).memcpyAsync(dst, dst_dev, src, src_dev, bytes, stream);
        }
    }

    void synchronize(IStream stream) {
        runtime(stream.device().type()).synchronizeStream(stream);
    }

    void synchronize(IEvent ev) {
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
