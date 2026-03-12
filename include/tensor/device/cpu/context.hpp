#pragma once

#include <mutex>
#include <memory>
#include <unordered_map>

#include "../device.hpp"

namespace EC::Dev
{

struct CPUContext {

    // TODO
    void* mem_pool = nullptr;
    std::mutex ctx_mutex;


    DeviceType type() const { return DeviceType::CPU; }

    static std::shared_ptr<CPUContext> get_instance(int dev_id){
        static std::shared_ptr<CPUContext> ctx;
        static std::mutex global_mutex;
        std::lock_guard<std::mutex> lock(global_mutex);
        if(ctx.get() == nullptr) ctx = std::make_shared<CPUContext>();
        return ctx;
    }

    void* allocate(size_t size,bool use_pool=true){
        std::lock_guard<std::mutex> lock(ctx_mutex);
        void* ptr = malloc(size);
        return ptr;
    }

    void deallocate(void* ptr){
        free(ptr);
    }

    ~CPUContext()=default;
    CPUContext(const CPUContext&) = delete;
    CPUContext& operator=(const CPUContext&)=delete;
    CPUContext(CPUContext&&) = default;
    CPUContext& operator=(CPUContext&&) = default;
private:
    CPUContext() = default;

};
} // namespace EC::Device
