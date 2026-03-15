#pragma once

#include <cstddef>

#include "tensor/device.hpp"
#include "abstract.hpp"

namespace EC::Dev
{
class IDeviceRuntime {
public:
    virtual ~IDeviceRuntime() = default;

    virtual DI device() const = 0;
    virtual int deviceCount() const = 0;
    virtual bool isAvailable(int idx) const = 0;

    virtual void setCurrentDevice(DI dev) = 0;
    virtual void synchronizeDevice(DI dev) = 0;

    virtual void* allocate(DI dev, size_t bytes, MemoryKind kind) = 0;
    virtual void deallocate(DI dev, void* ptr, MemoryKind kind) = 0;

    virtual void* allocateAsync(DI dev, size_t bytes, MemoryKind kind, StreamHandle stream) {
        (void)stream;
        return allocate(dev, bytes, kind);
    }

    virtual void deallocateAsync(DI dev, void* ptr, MemoryKind kind, StreamHandle stream) {
        (void)stream;
        deallocate(dev, ptr, kind);
    }

    virtual StreamHandle createStream(DI dev, int priority = 0) = 0;
    virtual void destroyStream(StreamHandle stream) = 0;
    virtual void synchronizeStream(StreamHandle stream) = 0;

    virtual EventHandle createEvent(DI dev, bool timing = false) = 0;
    virtual void destroyEvent(EventHandle event) = 0;
    virtual void recordEvent(EventHandle event, StreamHandle stream) = 0;
    virtual bool queryEvent(EventHandle event) = 0;
    virtual void synchronizeEvent(EventHandle event) = 0;
    virtual void waitEvent(StreamHandle stream, EventHandle event) = 0;

    virtual void memcpyAsync(void* dst, DI dst_dev,
                             const void* src, DI src_dev,
                             size_t bytes, StreamHandle stream) = 0;
};

}