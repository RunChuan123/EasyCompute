#pragma once

#include <cstddef>

#include "tensor/device.hpp"

namespace EC::Dev
{
class StreamHandle {
public:
    StreamHandle() = default;
    explicit StreamHandle(void* impl, DI device=DI::cpu()) : impl_(impl), device_(device) {}

    void set(void* impl_ptr=nullptr){impl_ = impl_ptr;}
    void reset(){impl_ = nullptr;}
    bool is_default();
    bool is_blockind();
    void* impl() const { return impl_; }
    DI device() const { return device_; }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    void* impl_{nullptr};
    DI device_;
};

class EventHandle {
public:
    EventHandle() = default;
    explicit EventHandle(void* impl, DI device) : impl_(impl), device_(device) {}
    void set(void* impl_ptr=nullptr){impl_ = impl_ptr;}
    void reset(){impl_ = nullptr;}

    void* impl() const { return impl_; }
    DI device() const { return device_; }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    void* impl_{nullptr};
    DI device_;
};


}