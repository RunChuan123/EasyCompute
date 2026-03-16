#pragma once

#include <cstddef>
#include <string>
#include <exception>

#include "dtype.hpp"
#include "util/err.hpp"
#include "util/check_cuda.cuh"
#include "device.hpp"


namespace EC {

struct BufferDesc{
    size_t nbytes = 0;
    DType dtype = DType::f32;
    DI device = DI::cpu();
    bool is_contituous = true;
    size_t align = 64;
    size_t offset_bytes = 0;
}

// TODO : check memory status if valid;
struct Buffer{
    void* ptr = nullptr;
    // size_t nbytes;
    // DType dtype = DType::f32;
    // DI device = DI::cpu();
    // bool is_contiguous = true;
    // size_t align = 64;
    // size_t offset_bytes = 0;
    BufferDesc desc;
    bool owns_memorys = false;
    bool allocated() const{return ptr != nullptr;}

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