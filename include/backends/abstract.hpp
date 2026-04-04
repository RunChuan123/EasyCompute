#pragma once

#include <cstddef>

#include "tensor/device.hpp"
#include "backends/plan.hpp"
#include "runtime/task_executor.hpp"

namespace EC::Dev
{

/**
 * 任务提交队列
 * 每个设备都有包括 CPU
 */
class IStream {
public:
    IStream() = default;
    explicit IStream(void* impl, DI device=DI::cpu()) : impl_(impl), device_(device) {}

    void set(void* impl_ptr=nullptr){impl_ = impl_ptr;}
    void reset(){impl_ = nullptr;}

    virtual void submit( Task* task) const =0;
    virtual void wait_event(IEvent ev) =0;
    virtual IEvent record_event() = 0; 
    virtual void synchronize() = 0;


    bool is_default();
    bool is_blocking();
    void* impl() const { return impl_; }
    DI device() const { return device_; }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    void* impl_{nullptr};
    DI device_;
};

/**
 *  任务依赖
 *  一个标记
 */
class IEvent {
public:
    IEvent() = default;
    explicit IEvent(void* impl, DI device) : impl_(impl), device_(device) {}
    void set(void* impl_ptr=nullptr){impl_ = impl_ptr;}
    void reset(){impl_ = nullptr;}

    virtual bool query() const =0;
    virtual void synchronize() = 0;

    void* impl() const { return impl_; }
    DI device() const { return device_; }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    void* impl_{nullptr};
    DI device_;
};

struct DevAllocator{
public:
    virtual void* allocate(size_t bytes,DI device)=0;
    virtual void deallocate(void* ptr,size_t bytes,DI device,void* ctx) = 0;
};


}