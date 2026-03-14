#pragma once

#include <memory>

#include "nv/context.cuh"
#include "cpu/context.hpp"
#include "device.hpp"


namespace EC{
namespace Dev{

class DeviceManager {
public:
    static thread_local std::unordered_map<int,std::shared_ptr<NVContext>> current_cuda_ctx_;
    // 线程局部存储：当前 CPU 上下文（默认初始化）
    static thread_local std::shared_ptr<CPUContext> current_cpu_ctx_;
    // 设置当前线程的目标设备上下文
    static void set_current_device(DeviceType type, int device_id = 0) {
        switch (type) {
            case DeviceType::CUDA: {
#ifdef CUDA_ENABLED
                current_cuda_ctx_[device_id] = NVContext::get_instance(device_id);
                // 可选：设置 CUDA 设备（关联上下文）
                cudaSetDevice(device_id);
#else
                throw std::runtime_error("CUDA is not enabled in this build");
#endif
                break;
            }
            case DeviceType::CPU: {
                current_cpu_ctx_ = CPUContext::get_instance(0);
                break;
            }
            default:
                throw std::invalid_argument("Unsupported DeviceType");
        }
    }



    static std::shared_ptr<NVContext> get_current_cuda_context(int id=0) {
        if(current_cuda_ctx_.find(id) == current_cuda_ctx_.end()) set_current_device(DeviceType::CUDA,id);
        
        return current_cuda_ctx_[id];
    }

    static std::shared_ptr<CPUContext> get_current_cpu_context(int id=0) {
        if (!current_cpu_ctx_) {
            id=0;
            current_cpu_ctx_ = CPUContext::get_instance(id);
        }
        return current_cpu_ctx_;
    }
    template <typename T>
    static std::shared_ptr<T> get_current_context() {
        if constexpr (std::is_same_v<T, NVContext>) {
            return get_current_cuda_context();
        } else if constexpr (std::is_same_v<T, CPUContext>) {
            return get_current_cpu_context();
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported context type");
        }
    }
    static void clear_current_context(DeviceType type) {
        if (type == DeviceType::CUDA) {
            current_cuda_ctx_.clear();
        } else if (type == DeviceType::CPU) {
            current_cpu_ctx_.reset();
        }
    }
};

// 初始化线程局部存储
thread_local std::unordered_map<int,std::shared_ptr<NVContext>> DeviceManager::current_cuda_ctx_ = {};
thread_local std::shared_ptr<CPUContext> DeviceManager::current_cpu_ctx_ = nullptr;

}
using DM = Dev::DeviceManager;

}