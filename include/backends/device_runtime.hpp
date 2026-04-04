#pragma once

#include <cstddef>

#include "tensor/device.hpp"
#include "abstract.hpp"

namespace EC::Dev
{
/**
 * 设备抽象层
 */
class IDeviceRuntime {
public:
    virtual std::string name();

    virtual ~IDeviceRuntime() = default;

    virtual DI device() const = 0;
    virtual int deviceCount() const = 0;
    virtual bool isAvailable(int idx) const = 0;

    virtual void setCurrentDevice(DI dev) = 0;
    virtual void synchronizeDevice(DI dev) = 0;

    virtual void* allocate(DI dev, size_t bytes, MemoryType kind) = 0;
    virtual void deallocate(DI dev, void* ptr, MemoryType kind) = 0;

    virtual void* allocateAsync(DI dev, size_t bytes, MemoryType kind, IStream stream) {
        (void)stream;
        return allocate(dev, bytes, kind);
    }

    virtual void deallocateAsync(DI dev, void* ptr, MemoryType kind, IStream stream) {
        (void)stream;
        deallocate(dev, ptr, kind);
    }

    virtual IStream createStream(DI dev, int priority = 0) = 0;
    virtual void destroyStream(IStream stream) = 0;
    virtual void synchronizeStream(IStream stream) = 0;

    virtual IEvent createEvent(DI dev, bool timing = false) = 0;
    virtual void destroyEvent(IEvent event) = 0;
    virtual void recordEvent(IEvent event, IStream stream) = 0;
    virtual bool queryEvent(IEvent event) = 0;
    virtual void synchronizeEvent(IEvent event) = 0;
    virtual void waitEvent(IStream stream, IEvent event) = 0;

    virtual void memcpyAsync(void* dst, DI dst_dev,
                             const void* src, DI src_dev,
                             size_t bytes, IStream stream) = 0;
    virtual void memcpy(void* dst, DI dst_dev,
                             const void* src, DI src_dev,
                             size_t bytes, IStream stream) = 0;
};

}