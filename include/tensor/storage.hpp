#pragma once 

#include <cstddef>
#include <mutex>
#include <cstring>

#include "dtype.hpp"
#include "util/err.hpp"
#include "device.hpp"
#include "backends/device_manager.hpp"


namespace EC::AT
{
  

struct Storage{
    void* ptr = nullptr;
    size_t nbytes = 0;
    DI device = DI::cpu();
    size_t align = 64;
    bool owns_memory = false;

    // host mem mirror

    void* host_ptr = nullptr;
    bool host_owns_memory = false;
    bool host_valid = false; // mirror 是否跟 ptr 一致
    bool host_dirty = false; // mirror 是否被修改过，是否需要刷回 device

    mutable std::mutex mtx_;

    Storage() = default;
    explicit Storage(size_t bytes,DI dev =DI::cpu(),size_t align_=64):nbytes(bytes),device(dev),align(align_){}
    bool allocated()const{return ptr != nullptr;}
    bool empty()const{return nbytes == 0;}
    void allocate() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (ptr != nullptr || nbytes == 0) return;

//         switch (device.type()) {
//             case DeviceType::CPU: {
// #ifdef __cpp_aligned_new
//                 ptr = ::operator new(nbytes, std::align_val_t(align));
// #else
//                 ptr = std::malloc(nbytes);
// #endif
//                 if (!ptr) throw std::bad_alloc();
//                 owns_memory = true;
//                 break;
//             }

//             case DeviceType::CUDA: {
        auto& dm = Dev::DeviceManager::get_instance();
        MemoryType mt = MemoryType::Host;
        if (device.type() ==  DeviceType::CUDA){
            mt = MemoryType::Device;
        }
        ptr = dm.allocate(device, nbytes, mt);
        if (!ptr) throw BufferException("Storage allocate CUDA failed");
        owns_memory = true;
            //     break;
            // }

            // default:
            //     throw BufferException("Storage allocate: unsupported device");
        // }
    }
    void allocateAsync(Dev::StreamHandle stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (ptr != nullptr || nbytes == 0) return;

        switch (device.type()) {
            case DeviceType::CPU: {
                allocate();
                break;
            }

            case DeviceType::CUDA: {
                auto& dm = Dev::DeviceManager::get_instance();
                ptr = dm.allocateAsync(device, nbytes, MemoryType::Device, stream);
                if (!ptr) throw BufferException("Storage allocateAsync CUDA failed");
                owns_memory = true;
                break;
            }

            default:
                throw BufferException("Storage allocateAsync: unsupported device");
        }
    }

    void allocate_host(){
        if(host_ptr || nbytes == 0) return;
#ifdef __cpp_aligned_new
        host_ptr = ::operator new(nbytes,std::align_val_t(align));
#else
        host_ptr = std::malloc(nbytes);
#endif
        if(!host_ptr)throw std::bad_alloc();
        host_owns_memory = true;
    }

    void ensure_host_mirror(){
        std::lock_guard<std::mutex> lock(mtx_);
         if (!ptr && nbytes > 0) {
            throw BufferException("ensure_host_mirror: device storage not allocated");
        }
        if(!host_ptr){
            allocate_host();
        }
        if(device.type() == DeviceType::CPU){
            // 直接镜像为数据副本
            if(!host_valid){
                std::memcpy(host_ptr,ptr,nbytes);
                host_valid = true;
                host_dirty = false;
            }return;
        }
        if(host_dirty){
            host_valid = true;
            return;
        }
        if(host_valid)return;

        // WARN: 谨记 auto 不会推到为引用！！！
        auto& dm = DM::get_instance();
        auto s = dm.createStream(device,0);
        dm.memcpyAsync(host_ptr,DI::cpu(),ptr,device,nbytes,s);
        dm.synchronize(s);
        dm.destroyStream(s);
        host_valid = true;
    }
    void flush_host_to_device_if_needed(){
        std::lock_guard<std::mutex> lock(mtx_);
        if(!host_dirty) return;
         if (!host_ptr) throw BufferException("flush_host_to_device_if_needed: no host mirror");
        if (!ptr) throw BufferException("flush_host_to_device_if_needed: no device storage");
        if(device.type() == DeviceType::CPU){
            std::memcpy(ptr,host_ptr,nbytes);
            host_valid = true;
            host_dirty = false;
            return;
        }
        auto& dm = DM::get_instance();
        auto s = dm.createStream(device, 0);
        dm.memcpyAsync(ptr, device, host_ptr, DI::cpu(),nbytes,s);
        dm.synchronize(s);
        dm.destroyStream(s);

        host_valid = true;
        host_dirty = false;
    }

    void mark_host_dirty(){
        std::lock_guard<std::mutex> lock(mtx_);
        host_valid = true;
        host_dirty = true;
    }

    void invalidate_host(){
        std::lock_guard<std::mutex> lock(mtx_);
        host_valid = false;
        host_dirty = false;
    }

    void release() noexcept {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!ptr || !owns_memory) return;

        try {
            if(ptr && owns_memory){
                switch (device.type()) {
                    case DeviceType::CPU: {
    #ifdef __cpp_aligned_new
                        ::operator delete(ptr, std::align_val_t(align));
    #else
                        std::free(ptr);
    #endif
                        break;
                    }

                    case DeviceType::CUDA: {
                        auto& dm = Dev::DeviceManager::get_instance();
                        dm.deallocate(device, ptr, MemoryType::Device);
                        break;
                    }

                    default:
                        break;
                }
            }
            if(host_ptr && host_owns_memory){
#ifdef __cpp_aligned_new
            ::operator delete(host_ptr, std::align_val_t(align));
#else
            std::free(host_ptr);
#endif
            }
        } catch (...) {
        }

        ptr = nullptr;
        host_ptr = nullptr;
        owns_memory = false;
        host_owns_memory = false;
        host_valid = false;
        host_dirty = false;
        nbytes = 0;
        
    }

    ~Storage() {
        release();
    }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& o) noexcept {
        *this = std::move(o);
    }

    Storage& operator=(Storage&& o) noexcept {
        if (this == &o) return *this;
        release();

        std::lock_guard<std::mutex> lock(o.mtx_);

        ptr = o.ptr;
        nbytes = o.nbytes;
        device = o.device;
        align = o.align;
        owns_memory = o.owns_memory;

        host_ptr = o.host_ptr;
        host_owns_memory = o.host_owns_memory;
        host_valid = o.host_valid;
        host_dirty = o.host_dirty;

        o.ptr = nullptr;
        o.host_ptr = nullptr;
        o.nbytes = 0;
        o.owns_memory = false;
        o.host_owns_memory = false;
        o.host_valid = false;
        o.host_dirty = false;
        return *this;
    }
};

} 
