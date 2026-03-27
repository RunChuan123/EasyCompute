#pragma once

#include <cstddef>
#include <string>
#include <exception>

#include "dtype.hpp"
#include "util/err.hpp"
#include "device.hpp"
#include "storage.hpp"


namespace EC::AT {

    // 不持有真实数据，是数据视图
struct Buffer {
    std::shared_ptr<Storage> storage;
    size_t offset_bytes{0};
    size_t visible_bytes{0};
    DType dtype{DType::f32};
    DI device{DI::cpu()};
    bool is_contiguous{true};

    Buffer() = default;

    Buffer(std::shared_ptr<Storage> st,
           size_t offset,
           size_t bytes,
           DType dt,
           DI dev,
           bool contiguous = true)
        : storage(st),
          offset_bytes(offset),
          visible_bytes(bytes),
          dtype(dt),
          device(dev),
          is_contiguous(contiguous) {}

    static std::shared_ptr<Buffer> make(size_t bytes,
                                        DType dt = DType::f32,
                                        DI dev = DI::cpu(),
                                        size_t align = 64) {
        auto st = std::make_shared<Storage>(bytes, dev, align);
        st->allocate();

        return std::make_shared<Buffer>(st, 0, bytes, dt, dev, true);
    }

    static std::shared_ptr<Buffer> make_unallocated(size_t bytes,
                                                    DType dt = DType::f32,
                                                    DI dev = DI::cpu(),
                                                    size_t align = 64) {
        auto st = std::make_shared<Storage>(bytes, dev, align);
        return std::make_shared<Buffer>(st, 0, bytes, dt, dev, true);
    }

    bool valid() const {
        return storage != nullptr;
    }

    bool allocated() const {
        return storage && storage->allocated();
    }

    bool empty() const {
        return visible_bytes == 0;
    }

    size_t nbytes() const {
        return visible_bytes;
    }

    size_t capacity_bytes() const {
        return storage ? storage->nbytes : 0;
    }

    void allocate() {
        if (!storage) {
            throw BufferException("Buffer allocate failed: null storage");
        }
        storage->allocate();
    }

    void allocateAsync(Dev::StreamHandle stream) {
        if (!storage) {
            throw BufferException("Buffer allocateAsync failed: null storage");
        }
        storage->allocateAsync(stream);
    }

    void flush_host_to_device_if_needed() const {
        if (!storage) return;
        storage->flush_host_to_device_if_needed();
    }

    void ensure_host_mirror() const {
        if (!storage) throw BufferException("Buffer::ensure_host_mirror null storage");
        storage->ensure_host_mirror();
    }

    void mark_host_dirty() const {
        if (!storage) return;
        storage->mark_host_dirty();
    }

    void invalidate_host() const {
        if (!storage) return;
        storage->invalidate_host();
    }

    void release() noexcept {
        // Buffer 自己不主动 release 底层内存。
        // 真正释放由 shared_ptr<Storage> 的生命周期决定。
        storage.reset();
        offset_bytes = 0;
        visible_bytes = 0;
        is_contiguous = true;
    }

    void* raw_ptr() {
        return storage ? storage->ptr : nullptr;
    }

    const void* raw_ptr() const {
        return storage ? storage->ptr : nullptr;
    }

    void* data_ptr() {
        if (!storage || !storage->ptr) return nullptr;
        return static_cast<void*>(static_cast<char*>(storage->ptr) + offset_bytes);
    }

    const void* data_ptr() const {
        if (!storage || !storage->ptr) return nullptr;
        return static_cast<const void*>(static_cast<const char*>(storage->ptr) + offset_bytes);
    }

    void* host_data_ptr() const {
        if (!storage || !storage->host_ptr) return nullptr;
        return static_cast<void*>(static_cast<char*>(storage->host_ptr) + offset_bytes);
    }

    template<typename T>
    T& operator[] (size_t idx){
        if(idx >= visible_bytes) throw std::out_of_range("Buffer View index out of range");
        char* base = static_cast<char*> (data_ptr());
        return *reinterpret_cast<T*>(base+idx * size_dtype(dtype));
    }
    template<typename T>
    const T& operator[] (size_t idx) const{
        if(idx >= visible_bytes) throw std::out_of_range("Buffer View index out of range");
        const char* base = static_cast<const char*> (data_ptr());
        return *reinterpret_cast<const T*>(base+idx * size_dtype(dtype));
    }

    Buffer view(size_t offset, size_t bytes) const {
        if (!storage) {
            throw BufferException("Buffer::view failed: null storage");
        }
        if (offset + bytes > visible_bytes) {
            throw BufferException("Buffer::view failed: out of range");
        }

        return Buffer{
            storage,
            offset_bytes + offset,
            bytes,
            dtype,
            device,
            is_contiguous
        };
    }
};

}