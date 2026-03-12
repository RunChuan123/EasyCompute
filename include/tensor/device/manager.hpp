#pragma once

#include <memory>

#include "device.hpp"

namespace EC::Device{

struct DeviceContext{
     virtual ~DeviceContext() = default;

    // ========== 所有设备通用的核心接口 ==========
    // 1. 获取设备类型/ID
    virtual DeviceType type() const = 0;
    virtual int device_id() const = 0;

    // 2. 内存分配/释放（通用接口，子类各自实现）
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;

    // 3. 同步（通用语义：等待所有操作完成）
    virtual void sync() = 0;

    // ========== 设备特有接口（类型安全的向下转换） ==========
    // CUDA特有接口：子类实现，基类返回nullptr
    virtual cudaStream_t get_compute_stream(int stream_idx = 0) { return nullptr; }
    virtual cudaStream_t get_h2d_stream() { return nullptr; }
    virtual cudaStream_t get_mem_stream() { return nullptr; }
    virtual void record_event(cudaStream_t stream, cudaEvent_t& event) {}
    virtual void wait_event(cudaStream_t stream, cudaEvent_t event) {}

    // CPU特有接口：子类实现，基类空实现
    virtual void set_cpu_affinity(int core_id) {}
};

struct DeviceManager {
    static std::shared_ptr<DevContext> get_current_ctx() {
        static thread_local int current_device = 0;
        return DevContext::get_instance(current_device);
    }

    static void set_current_device(int device_id) {
        thread_local int current_device = 0;
        current_device = device_id;
    }
};

}