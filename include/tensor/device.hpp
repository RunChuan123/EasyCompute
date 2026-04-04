#pragma once

#include <cstdint>
#include <sstream>
#include <string>

namespace EC
{
    
/**
 * 设备类型
 */
enum class DeviceType :uint8_t{
    CPU=0,
    CUDA,
    ASCEND,

    NumDevice
};
/**
 * 内存类型
 */
enum class MemoryType : uint8_t {
    Host,
    PinnedHost,
    Device,
    Unified,
};

/**
 * 解释内存
 */
struct MemoryKind{
    DeviceType type_;
    int index_ = 0;
    MemoryType memtype_;

    static MemoryKind cpu(){return {DeviceType::CPU,0,MemoryType::Host};}

    static MemoryKind cuda(int cuda_device_idx=0){
        return {DeviceType::CUDA,cuda_device_idx,MemoryType::Device};
    }

    DeviceType devtype()const{return type_;}
    int id()const{return index_;}
    MemoryType memtype()const{return memtype_;}

};

/**
 * 设备标识，设备类型+index
 */
struct DeviceIdentification{
    DeviceType type_;
    int index_ = 0;

    static DeviceIdentification cpu(){return {DeviceType::CPU,0};}

    static DeviceIdentification cuda(int cuda_deviec_idx=0){return {DeviceType::CUDA,cuda_deviec_idx};}


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
            oss << "unkonw dev";
            break;
        }
        oss << index_;
        return oss.str();
    }
    inline bool operator==(const DeviceIdentification& rhs) const {
        if(rhs.type_ == type_ && rhs.index_ == index_)return true;
        return false;
    }

};

using DI = DeviceIdentification;

}