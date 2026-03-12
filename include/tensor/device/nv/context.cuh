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
#include "tensor/device/manager.cuh"



namespace EC::Dev
{
    
struct NVContext{
public:
    int device_id_;
    DeviceType type_;
    cudaDeviceProp device_prop; 
    cudaStream_t default_stream = nullptr;
    
    std::unordered_map<std::string,cudaStream_t> custom_streams;
    cudaMemPool_t mem_pool = nullptr;
    std::mutex ctx_mutex;

    DeviceType type()const{return type_;}
    int device_id() const {return device_id_;}
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
        if(device_id_ !=-1){
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
        device_id_ = dev_id;
        err = cudaSetDevice(device_id_);
        if(err != cudaSuccess){
            oss << "NVContext: change device failed / "<< device_id_<< " " << cudaGetErrorString(err);
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
            cudaDeviceGetDefaultMemPool(&mem_pool, device_id_);
            cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, 0);
        }

        LOG_INFO("NVContext: device ", device_prop.name," ,id: " ,device_id_," init success | current gpu memory: ",device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        return true;
    }

    void* allocate(size_t size,bool use_pool=true,cudaStream_t s = cudaStreamPerThread){
        std::lock_guard<std::mutex> lock(ctx_mutex);
        if(device_id_==-1 || size == 0) return nullptr;
        cudaSetDevice(device_id_);
        void* ptr = nullptr;
        if(use_pool && mem_pool != nullptr){
            cudaMallocFromPoolAsync(&ptr,size,mem_pool,s);
        }else{
            cudaMallocAsync(&ptr,size,s);
        }
        return ptr;
    }

    void deallocate(void* ptr){
        if(ptr == nullptr)return;
        std::lock_guard<std::mutex> lock(ctx_mutex);
        cudaSetDevice(device_id_);
        cudaFreeAsync(ptr,default_stream);
    }

    void sync_stream(cudaStream_t s = nullptr){
        std::lock_guard<std::mutex> lock(ctx_mutex);
        cudaSetDevice(device_id_);
        if(s == nullptr) s = default_stream;
        cudaStreamSynchronize(s);
    }

    cudaStream_t create_custom_stream(std::string& name,bool non_blocking = true){
        std::lock_guard<std::mutex> lock(ctx_mutex);
        if(custom_streams.find(name) != custom_streams.end()) return custom_streams[name];
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream,non_blocking ? cudaStreamNonBlocking : cudaStreamDefault);
        custom_streams[name] = stream;
        return stream;
    }

    void destroy_custom_stream(const std::string& stream_name) {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        auto it = custom_streams.find(stream_name);
        if (it != custom_streams.end()) {
            cudaStreamDestroy(it->second);
            custom_streams.erase(it);
        }
    }
    void record_event(cudaStream_t stream, cudaEvent_t& event) {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        cudaSetDevice(device_id_);
        cudaEventRecord(event, stream);
    }

    void wait_event(cudaStream_t stream, cudaEvent_t event) {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        cudaSetDevice(device_id_);
        cudaStreamWaitEvent(stream, event, 0);
    }
    
    ~NVContext(){
        std::lock_guard<std::mutex> lock(ctx_mutex);
        if(device_id_ == -1) return;
        for(auto& [_,stream] : custom_streams)cudaStreamDestroy(stream);
        custom_streams.clear();
        device_id_=-1;
    }

    NVContext(const NVContext&) = delete;
    NVContext& operator=(const NVContext&)=delete;
    NVContext(NVContext&&) = default;
    NVContext& operator=(NVContext&&) = default;
private:
    NVContext() = default;

};


} // namespace EC
