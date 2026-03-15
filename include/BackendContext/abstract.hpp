#include "tensor/device.hpp"

class StreamHandle {
public:
    StreamHandle() = default;
    explicit StreamHandle(void* impl, Device device) : impl_(impl), device_(device) {}

    void* impl() const { return impl_; }
    Device device() const { return device_; }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    void* impl_{nullptr};
    Device device_;
};

class EventHandle {
public:
    EventHandle() = default;
    explicit EventHandle(void* impl, Device device) : impl_(impl), device_(device) {}

    void* impl() const { return impl_; }
    Device device() const { return device_; }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    void* impl_{nullptr};
    Device device_;
};

class IDeviceRuntime {
public:
    virtual ~IDeviceRuntime() = default;

    virtual DeviceType type() const = 0;
    virtual bool isAvailable(int index) const = 0;
    virtual int deviceCount() const = 0;

    virtual void setCurrentDevice(int index) = 0;
    virtual void synchronizeDevice(int index) = 0;

    virtual void* allocate(size_t bytes, int device_index, MemoryKind kind) = 0;
    virtual void deallocate(void* ptr, int device_index, MemoryKind kind) = 0;

    virtual void memcpyH2D(void* dst, const void* src, size_t bytes, StreamHandle stream) = 0;
    virtual void memcpyD2H(void* dst, const void* src, size_t bytes, StreamHandle stream) = 0;
    virtual void memcpyD2D(void* dst, const void* src, size_t bytes, StreamHandle stream) = 0;
    virtual void memcpyAsync(void* dst, Device dst_dev,
                             const void* src, Device src_dev,
                             size_t bytes,
                             StreamHandle stream) = 0;

    virtual StreamHandle createStream(Device device, int priority = 0) = 0;
    virtual void destroyStream(StreamHandle stream) = 0;
    virtual void synchronizeStream(StreamHandle stream) = 0;

    virtual EventHandle createEvent(Device device, bool timing = false) = 0;
    virtual void destroyEvent(EventHandle event) = 0;
    virtual void recordEvent(EventHandle event, StreamHandle stream) = 0;
    virtual bool queryEvent(EventHandle event) = 0;
    virtual void synchronizeEvent(EventHandle event) = 0;
    virtual void waitEvent(StreamHandle stream, EventHandle event) = 0;

    virtual size_t freeMemory(int device_index) const = 0;
    virtual size_t totalMemory(int device_index) const = 0;
};