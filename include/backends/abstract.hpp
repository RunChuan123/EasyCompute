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
    virtual ~IStream() = default;

    virtual void submit( TS::Task* task) const =0;
    virtual void wait_event(IEvent* ev) =0;
    virtual IEvent* record_event(){return nullptr;} 
    virtual void synchronize() = 0;
    virtual DI device() const =0;
    // 未来可能会用到
    // bool is_default();
    // bool is_blocking();
};

/**
 *  任务依赖
 *  一个标记
 */
class IEvent {
public:
    virtual ~IEvent() = default;

    virtual bool query() const =0;
    virtual void synchronize() = 0;
    virtual void set_complete(){}


};

struct DevAllocator{
public:
    virtual void* allocate(size_t bytes,DI device)=0;
    virtual void deallocate(void* ptr,size_t bytes,DI device,void* ctx) = 0;
};


}