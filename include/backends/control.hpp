/**
 * control 层负责接收 ready 的 task 并分发到不同的 istream 执行
 * Scheduler 只决定 device/hint，Control 决定 stream
 */

#pragma once

#include <memory>

#include <mutex>
#include <unordered_map>

#include "runtime/task.hpp"
#include "util/logger.hpp"
#include "runtime/task_scheduler.hpp"
#include "backends/cpu/exec.hpp"

#ifdef EC_ENABLE_CUDA
#include "backends/cuda/exec.cuh"
#endif


namespace EC::Dev
{
/**
 * 第一版全走 default_stream
 */
struct Control {
public:
    static Control& get_instance(){
        static Control instance;
        return instance;
    }

    Control(const Control&) = delete;
    Control& operator=(const Control&) = delete;

    bool dispatch(std::shared_ptr<TS::Task> task){
        if (!task) {
            LOG_ERROR("dispatch failed: task is null");
            return false;
        }

        auto stream = select_stream(task);
        if (!stream) {
            LOG_ERROR("dispatch failed: no stream selected for task ", task->task_name);
            return false;
        }

        task->status.store(TS::TaskStatus::Submitted, std::memory_order_release);
        stream->submit(task);
        return true;
    }
    std::shared_ptr<IStream> select_stream(std::shared_ptr<TS::Task> task){
        if(!task) return nullptr;
        switch (task->exec_device.type()){
            case DeviceType::CPU:
                return select_cpu_stream(task);
            case DeviceType::CUDA:
                return select_cuda_stream(task);
            default:{
                LOG_ERROR("unsupported device for task ", task->task_name);
                return nullptr;
            }
        }
    }
    std::shared_ptr<IStream> default_stream(DI dev) {
        switch (dev.type()) {
            case DeviceType::CPU:
                return cpu_default_;
            case DeviceType::CUDA:
                return cuda_default_;
            default:
                return nullptr;
        }
    }


private:
    Control(){
        cpu_default_ = CPU::default_cpu_stream();
#ifdef EC_ENABLE_CUDA
        cuda_default_ = CUDA::default_cuda_stream();
#endif   
    }
    std::shared_ptr<IStream> select_cpu_stream(std::shared_ptr<TS::Task> task) {
        (void)task;
        return cpu_default_;
    }
    std::shared_ptr<IStream> select_cuda_stream(std::shared_ptr<TS::Task> task) {
#ifdef EC_ENABLE_CUDA
        // 第一版先全走 default，后面再按 stream_hint 拆
        (void)task;
        return cuda_default_;
#else
        LOG_ERROR("CUDA disabled, but task requests CUDA device");
        return nullptr;
#endif
    }

    std::shared_ptr<IStream> cpu_default_;
    std::shared_ptr<IStream> cuda_default_;

};
    





} // namespace EC::Dev
