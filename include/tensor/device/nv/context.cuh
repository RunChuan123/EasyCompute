#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <iostream>
#include <sstream>

#include "util/err.hpp"
#include "util/logger.hpp"

namespace EC::Device
{
    
struct NVContext{
public:
    int device_id;
    cudaDeviceProp device_prop; 
    cudaStream_t default_stream = nullptr;
    std::unordered_map<std::string,cudaStream_t> custom_streams;
    cudaMemPool_t mem_pool = nullptr;
    std::mutex ctx_mutex;

    static std::shared_ptr<NVContext> get_instance(int dev_id){
        static std::unordered_map<int,std::shared_ptr<NVContext>> ctx_map;
        static std::mutex global_mutex;
        std::lock_guard<std::mutex> lock(global_mutex);
        if(ctx_map.find(dev_id) == ctx_map.end()){
            auto ctx = std::make_shared<NVContext>();
            ctx->init(dev_id);
            ctx_map[dev_id] = ctx;
        }
        return ctx_map[dev_id];
    }

    bool init(int dev_id){
        std::lock_guard<std::mutex> lock(ctx_mutex);
        // 已经初始化
        if(device_id !=-1){
            return true;
        }
        int device_count;
        std::ostringstream oss;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if(err != cudaSuccess || dev_id < 0 || dev_id >= device_count){
            oss << "NVContext: device_id invalid " << dev_id << " / " << device_count;
            throw DeviceException(oss.str());
            return false;
        }
        device_id = dev_id;
        err = cudaSetDevice(device_id);
        if(err != cudaSuccess){
            oss << "NVContext: change device failed / "<< device_id<< " " << cudaGetErrorString(err);
            throw DeviceException(oss.str());
            return false;
        }
        // 可以在后续改伟创建自定义默认流以供异步执行算子优化
        default_stream = cudaStreamPerThread;

        // 这是什么？
        // 初始化显存池（ML框架核心优化：减少显存碎片）
        // // 仅CUDA 11.2+支持，按需开启
        int version_;
        cudaRuntimeGetVersion(&version_);
        if(version_ >= 11020) {
            cudaMemPoolProps pool_props{};
            pool_props.allocType = cudaMemAllocationTypePinned;
            pool_props.handleTypes = cudaMemHandleTypeNone;
            cudaDeviceGetDefaultMemPool(&mem_pool, device_id);
            cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, 0);
        }

        LOG_DEBUG("NVContext: device ", device_prop.name," ,id: " ,device_id," init success | current gpu memory: ",device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        return true;
    }

    void* allocate()
    






};




} // namespace EC
