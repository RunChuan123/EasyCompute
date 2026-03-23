#pragma once

#include <cstdint>
#include <sstream>
#include <string>

namespace EC
{
    
enum class DeviceType :uint8_t{
    CPU=0,
    CUDA,
    ASCEND,

    NumDevice
};

enum class MemoryType : uint8_t {
    Host,
    PinnedHost,
    Device,
    Unified,
};

struct MemoryKind{
    DeviceType type_;
    int index_ = 0;
    MemoryType memtype_;

    static MemoryKind cpu(){
        return {DeviceType::CPU,0,MemoryType::Host};
    }
    static MemoryKind cuda(int idx=0){
        return {DeviceType::CUDA,idx,MemoryType::Device};
    }
    bool in_cpu() const {return type_ == DeviceType::CPU;}
    bool in_cuda() const {return type_ == DeviceType::CUDA;}
    bool in_ascend() const { return type_ == DeviceType::ASCEND; }

    DeviceType type()const{return type_;}
    int id()const{return index_;}
    MemoryType memtype()const{return memtype_;}
};

struct DeviceIdentification{
    DeviceType type_;
    int index_ = 0;

    static DeviceIdentification cpu(){
        return {DeviceType::CPU,0};
    }
    static DeviceIdentification cuda(int idx=0){
        return {DeviceType::CUDA,idx};
    }
    bool is_cpu() const {return type_ == DeviceType::CPU;}
    bool is_cuda() const {return type_ == DeviceType::CUDA;}
    bool is_ascend() const { return type_ == DeviceType::ASCEND; }

    DeviceType type()const{return type_;}
    int id()const{return index_;}
    std::string to_string()const{
        std::ostringstream oss;
        switch (type_)
        {
        case DeviceType::CPU:
            oss << "cpu:";
            break;
        case DeviceType::CUDA:
            oss << "cuda:";
            break;
        case DeviceType::ASCEND:
            oss << "ascend:";
            break;
        default:
            break;
        }
        oss << index_;
        return oss.str();
    }
    inline bool operator==(const DeviceIdentification& rhs){
        if(rhs.type_ == type_ && rhs.index_ == index_)return true;
        return false;
    }
};

using DI = DeviceIdentification;
}


