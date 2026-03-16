#pragma once 

#include <cstddef>

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

    Storage() = default;
    explicit Storage(size_t bytes,DI dev =DI::cpu(),size_t align_=64):nbytes(bytes),device(dev),align(align_){}
    bool allocated()const{return ptr != nullptr;}
    bool empty()const{return nbytes == 0;}
    void allocate() {
        if (ptr != nullptr || nbytes == 0) return;

        switch (device.type()) {
            case DeviceType::CPU: {
#ifdef __cpp_aligned_new
                ptr = ::operator new(nbytes, std::align_val_t(align));
#else
                ptr = std::malloc(nbytes);
#endif
                if (!ptr) throw std::bad_alloc();
                owns_memory = true;
                break;
            }

            case DeviceType::CUDA: {
                auto& dm = Dev::DeviceManager::get_instance();
                ptr = dm.allocate(device, nbytes, MemoryType::Device);
                if (!ptr) throw BufferException("Storage allocate CUDA failed");
                owns_memory = true;
                break;
            }

            default:
                throw BufferException("Storage allocate: unsupported device");
        }
    }
    void allocateAsync(Dev::StreamHandle stream) {
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

    void release() noexcept {
        if (!ptr || !owns_memory) return;

        try {
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
        } catch (...) {
        }

        ptr = nullptr;
        nbytes = 0;
        owns_memory = false;
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

        ptr = o.ptr;
        nbytes = o.nbytes;
        device = o.device;
        align = o.align;
        owns_memory = o.owns_memory;

        o.ptr = nullptr;
        o.nbytes = 0;
        o.owns_memory = false;
        return *this;
    }
};

} 
