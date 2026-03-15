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
    static DeviceManager& instance() {
        static DeviceManager dm;
        return dm;
    }

    void registerRuntime(std::unique_ptr<IDeviceRuntime> runtime) {
        runtimes_[runtime->device()] = std::move(runtime);
    }

    IDeviceRuntime& runtime(DI dev) {
        auto it = runtimes_.find(dev);
        if (it == runtimes_.end()) {
            throw std::runtime_error("runtime not registered");
        }
        return *(it->second);
    }

    void* allocate(DI dev, size_t bytes, MemoryKind kind) {
        return runtime(dev).allocate(dev, bytes, kind);
    }

    void deallocate(DI dev, void* ptr,  MemoryKind kind) {
        runtime(dev).deallocate(dev, ptr, kind);
    }

    void* allocateAsync(DI dev, size_t bytes, MemoryKind kind, StreamHandle stream) {
        return runtime(dev).allocateAsync(dev, bytes, kind, stream);
    }

    void deallocateAsync(DI dev, void* ptr, MemoryKind kind, StreamHandle stream) {
        runtime(dev).deallocateAsync(dev, ptr, kind, stream);
    }

    StreamHandle createStream(DI dev, int priority = 0) {
        return runtime(dev).createStream(dev, priority);
    }

    EventHandle createEvent(DI dev, bool timing = false) {
        return runtime(dev).createEvent(dev, timing);
    }

    void recordEvent(EventHandle ev, StreamHandle stream) {
        runtime(ev.device()).recordEvent(ev, stream);
    }

    bool queryEvent(EventHandle ev) {
        return runtime(ev.device()).queryEvent(ev);
    }

    void waitEvent(StreamHandle stream, EventHandle ev) {
        runtime(stream.device()).waitEvent(stream, ev);
    }

    void memcpyAsync(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, StreamHandle stream) {
        runtime(dst_dev).memcpyAsync(dst, dst_dev, src, src_dev, bytes, stream);
    }

    void synchronize(StreamHandle stream) {
        runtime(stream.device()).synchronizeStream(stream);
    }

    void synchronize(EventHandle ev) {
        runtime(ev.device()).synchronizeEvent(ev);
    }

    void synchronize(DI dev) {
        runtime(dev).synchronizeDevice(dev);
    }

private:
    DeviceManager() = default;
    std::unordered_map<DI, std::unique_ptr<IDeviceRuntime>> runtimes_;
};
}
}

// class DeviceManager {
// public:
//     static thread_local std::unordered_map<int,std::shared_ptr<NVContext>> current_cuda_ctx_;
//     // 线程局部存储：当前 CPU 上下文（默认初始化）
//     static thread_local std::shared_ptr<CPUContext> current_cpu_ctx_;
//     // 设置当前线程的目标设备上下文
//     static void set_current_device(DeviceType type, int device_id = 0) {
//         switch (type) {
//             case DeviceType::CUDA: {
// #ifdef CUDA_ENABLED
//                 current_cuda_ctx_[device_id] = NVContext::get_instance(device_id);
//                 // 可选：设置 CUDA 设备（关联上下文）
//                 cudaSetDevice(device_id);
// #else
//                 throw std::runtime_error("CUDA is not enabled in this build");
// #endif
//                 break;
//             }
//             case DeviceType::CPU: {
//                 current_cpu_ctx_ = CPUContext::get_instance(0);
//                 break;
//             }
//             default:
//                 throw std::invalid_argument("Unsupported DeviceType");
//         }
//     }



//     static std::shared_ptr<NVContext> get_current_cuda_context(int id=0) {
//         if(current_cuda_ctx_.find(id) == current_cuda_ctx_.end()) set_current_device(DeviceType::CUDA,id);
        
//         return current_cuda_ctx_[id];
//     }

//     static std::shared_ptr<CPUContext> get_current_cpu_context(int id=0) {
//         if (!current_cpu_ctx_) {
//             id=0;
//             current_cpu_ctx_ = CPUContext::get_instance(id);
//         }
//         return current_cpu_ctx_;
//     }
//     template <typename T>
//     static std::shared_ptr<T> get_current_context() {
//         if constexpr (std::is_same_v<T, NVContext>) {
//             return get_current_cuda_context();
//         } else if constexpr (std::is_same_v<T, CPUContext>) {
//             return get_current_cpu_context();
//         } else {
//             static_assert(!std::is_same_v<T, T>, "Unsupported context type");
//         }
//     }
//     static void clear_current_context(DeviceType type) {
//         if (type == DeviceType::CUDA) {
//             current_cuda_ctx_.clear();
//         } else if (type == DeviceType::CPU) {
//             current_cpu_ctx_.reset();
//         }
//     }
// };

// // 初始化线程局部存储
// thread_local std::unordered_map<int,std::shared_ptr<NVContext>> DeviceManager::current_cuda_ctx_ = {};
// thread_local std::shared_ptr<CPUContext> DeviceManager::current_cpu_ctx_ = nullptr;

// }
// using DM = Dev::DeviceManager;

// }









    // Buffer h2d(int device_id = 0,bool async=true)const{
    //     if (device.type() != DeviceType::CPU) {
    //         throw std::runtime_error("h2d() only support CPU Buffer");
    //     }
    //     Buffer newb;
    //     auto ctx = DM::get_current_cuda_context(device_id);
    //     newb.ptr = ctx->allocate(nbytes,true,ctx->custom_streams["allocate_cuda"]);
    //     if (newb.ptr == nullptr) {
    //         throw std::runtime_error("CUDA allocate memory failed");
    //     }
    //     newb.dtype = this->dtype;
    //     newb.device = Device::cuda(device_id);  
    //     newb.nbytes = nbytes;
    //     newb.align = this->align;
    //     if(this->ptr != nullptr){
    //         cudaMemcpyAsync(newb.ptr,this->ptr,nbytes,cudaMemcpyHostToDevice,ctx->custom_streams["h2d" + std::to_string(device.id())]);
    //     }
    //     if(!async){ctx->sync_stream(ctx->custom_streams["h2d" + std::to_string(device.id())]);}
    //     return newb;
    // }

    // Buffer d2h(bool async = true){
    //     if (device.type() != DeviceType::CUDA) {
    //         throw std::runtime_error("d2h() only support CUDA Buffer");
    //     }
    //     Buffer newb;
    //     // 对齐
    //     if (posix_memalign(&newb.ptr, align, nbytes) != 0) {
    //         throw std::runtime_error("CPU allocate memory failed (posix_memalign)");
    //     }

    //     newb.dtype = this->dtype;
    //     newb.device = Device::cpu();
    //     newb.nbytes = nbytes;
    //     newb.align = this->align;

    //     if (this->ptr != nullptr ) {

    //         cudaSetDevice(device.id());
    //         // 异步拷贝
    //         CUDA_CHECK(cudaMemcpyAsync(newb.ptr, this->ptr, nbytes,cudaMemcpyDeviceToHost, 
    //             DM::get_current_cuda_context(device.id())->custom_streams["d2h" + std::to_string(device.id())]));
    //         if(!async){ DM::get_current_cuda_context(device.id())->sync_stream("h2d" + std::to_string(device.id()));}
    //     }

    //     return newb; // 移动构造
    // }

    // void h2d_(int device_id=0,bool async = true){
    //     if (device.type() != DeviceType::CPU) {
    //         throw std::runtime_error("h2d_() only support CPU Buffer");
    //     }
    //     if (this->ptr == nullptr) {this->device=Device::cuda(this->device.id());return;}
    //     auto cuda_ctx = DM::get_current_cuda_context(device_id);
    //     void* cuda_ptr = cuda_ctx->allocate(nbytes,true,cuda_ctx->custom_streams["h2d" + std::to_string(device_id)]);
    //     CUDA_CHECK(cudaMemcpyAsync(cuda_ptr, this->ptr,nbytes, 
    //                               cudaMemcpyHostToDevice, cuda_ctx->custom_streams["h2d"+ std::to_string(device_id)]));
    //     if(!async){cuda_ctx->sync_stream("h2d"+ std::to_string(device_id));}
    //     DM::get_current_cpu_context()->deallocate(this->ptr,align);
    //     this->ptr = cuda_ptr;
    //     this->device = Device::cuda(device_id);
    // }

    // void d2h_(bool async = true){
    //     if (device.type() != DeviceType::CUDA) {
    //         throw std::runtime_error("h2d_() only support CUDA Buffer");
    //     }
    //     auto cuda_ctx = DM::get_current_cuda_context(device.id());
    //     cudaStream_t s = cuda_ctx->custom_streams["d2h"+ std::to_string(device.id())];
    //     if(nbytes == 0){this->device = Device::cpu();return;}
        
    //     void* cpu_ptr = DM::get_current_cpu_context()->allocate(nbytes,true);
    //     CUDA_CHECK(cudaMemcpyAsync(cpu_ptr, this->ptr,nbytes, 
    //                               cudaMemcpyDeviceToHost,s ));
    //     this->ptr = cpu_ptr;
    //     if(!async){cuda_ctx->sync_stream(s);}
    //     return;
    // }