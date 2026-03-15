#pragma once

#include <cstddef>
#include <string>
#include <exception>

#include "dtype.hpp"
#include "util/err.hpp"
#include "util/check_cuda.cuh"
#include "device.hpp"
#include "function.cuh"
#include "device/manager.cuh"

namespace EC {

// TODO : check memory status if valid;
struct Buffer{
    void* ptr = nullptr;
    size_t nbytes;
    DType dtype;
    DI device;
    bool is_contiguous = true;
    size_t align = 64;
    size_t offset_bytes = 0;

    Buffer()=default;
    explicit Buffer(size_t bytes,DType dt=DType::f32,DI dev=DI::cpu(),size_t align_=64):nbytes(bytes),dtype(dt),device(dev),align(align_){
        switch (device.type()){
        case DeviceType::CPU:{
#ifdef __cpp_aligned_new
            ptr = ::operator new(bytes,::std::align_val_t(align));
#else
            ptr = std::malloc(bytes);
#endif
            if(!ptr) throw ::std::bad_alloc();
        }break;
        case DeviceType::CUDA:{ptr = DM::get_current_cuda_context(device.id())->allocate(nbytes,true,DM::get_current_cuda_context()->default_stream);}break;
        default:
            throw BufferException("unknow device to allocate memory!");
        }
    }
    // 释放原数据
    void release()noexcept{
        switch(device.type()){
            case DeviceType::CPU :{
                DM::get_current_cpu_context(device.id())->deallocate(ptr,align);
            } return;
            case DeviceType::CUDA:{
                DM::get_current_cuda_context(device.id())->deallocate(ptr);
            } return;
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

    Buffer(Buffer&& o) noexcept {
        ptr = o.ptr;
        dtype = o.dtype;
        device = o.device;
        nbytes = o.nbytes;
        align = o.align;
        o.ptr = nullptr;
        o.nbytes = 0;
    }



    // 获取数据指针
    void* data_ptr(){return ptr;}
    const void* data_ptr()const{return ptr;}
    Buffer move() noexcept {return std::move(*this);}


};

}