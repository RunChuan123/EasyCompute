#pragma once

#include <cstddef>
#include <exception>

#include "dtype.hpp"
#include "device/device.hpp"

namespace EC {

// TODO : check memory status if valid;
struct Buffer{
    void* ptr = nullptr;
    size_t nbytes;
    DType dtype;
    Device device;
    size_t dev_id = 0;
    size_t align = 64;
    size_t offset_bytes = 0;

    Buffer()=default;
    explicit Buffer(size_t bytes,DType dt=DType::f32,Device dev=Device::CPU,size_t dev_id = 0,size_t align_=64):nbytes(bytes),dtype(dt),device(dev),align(align_){
        switch (device){
        case Device::CPU:{
#ifdef __cpp_aligned_new
            ptr = ::operator new(bytes,::std::align_val_t(align));
#else
            ptr = std::malloc(bytes);
#endif
            if(!ptr) throw ::std::bad_alloc();
        }break;
        case Device::NV_GPU:{
            
            break;
        }
        
        default:
            throw BufferException("unknow device to allocate memory!");
        }
    }
    ~Buffer() { release(); }    

    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    Buffer& operator=(Buffer&& o) noexcept{
        if(this == &o) return *this;
        release();
        ptr=o.ptr;
        dtype=o.dtype;
        device=o.device;
        nbytes=o.nbytes;
        align = o.align;
        o.ptr=nullptr;
        o.nbytes=0;

        return *this;
    }

    // 释放原数据
    void release()noexcept{
        if(ptr){
#ifdef __cpp_aligned_new
            ::operator delete(ptr,::std::align_val_t(align));
#else
            std::free(ptr);
#endif  
        }
    }
    
    // 获取数据指针
    void* data_ptr(){return ptr;}
    const void* data_ptr()const{return ptr;}


};

}