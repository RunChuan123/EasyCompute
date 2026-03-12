#pragma once

#include <cstdint>

namespace EC
{
enum class DeviceType :uint8_t{
    CPU=0,
    CUDA,

    NumDevice
};

enum class MemoryType : uint8_t {
    Host,
    PinnedHost,
    Device,
    Unified,
};

struct Device{
    DeviceType type_;
    int index_ = 0;
    MemoryType memtype_;

    static Device cpu(){
        return {DeviceType::CPU,0,MemoryType::Host};
    }
    static Device cuda(int idx=0){
        return {DeviceType::CUDA,idx,MemoryType::Device};
    }
    bool is_cpu() const {return type_ == DeviceType::CPU;}
    bool is_cuda() const {return type_ == DeviceType::CUDA;}

    DeviceType type()const{return type_;}
};

struct DeviceContext{
    virtual ~DeviceContext() = default;
    virtual Device device() const = 0;
};


} // namespace Ec
