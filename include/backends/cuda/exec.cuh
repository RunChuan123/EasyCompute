#pragma once
#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA
#include <cuda_runtime.h>
#include "backends/abstract.hpp"
#include "runtime/task_executor.hpp"
#include "util/logger.hpp"

namespace EC::Dev::CUDA
{

struct CUDAEvent final:public IEvent{
    cudaEvent_t cuda_event_;
    CUDAEvent(){
        // 初始化成员变量
        cudaEventCreate(&cuda_event_);
    }
    ~CUDAEvent(){
        cudaEventDestroy(cuda_event_);
    }
    bool query() const override{
        cudaError_t err = cudaEventQuery(cuda_event_);
        return (err == cudaSuccess);
    }
    void synchronize() override {
        cudaEventSynchronize(cuda_event_);
    }

};


struct CUDAStreamState {
public:
    std::mutex mtx_stm;
    std::condition_variable cv_stm;
    std::deque<TS::Task*> queue_stm;
    bool running{false};
    std::thread worker;

    void push_task(TS::Task* t) {
        if (t) {
            std::lock_guard<std::mutex> lock(mtx_stm);
            queue_stm.push_back(t);
        } else {
            LOG_WARN("task", t->task_name, "is null!");
        }
    }

    bool start_worker() {
        if (!running) {
            {
                std::lock_guard<std::mutex> lock(mtx_stm);
                worker = std::thread([this]() { worker_function(); });
                running = true;
            }
            LOG_INFO("cuda stream worker", worker.get_id(), "started running");
        }
        return running;
    }

    void stop_worker() {
        {
            std::lock_guard<std::mutex> lock(mtx_stm);
            running = false;
        }
        cv_stm.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

private:
    void worker_function() {
        while (true) {
            TS::Task* t = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx_stm);
                cv_stm.wait(lock, [this]() { return !queue_stm.empty() || !running; });

                if (!running || queue_stm.empty()) {
                    break;
                }

                t = queue_stm.front();
                queue_stm.pop_front();
            }
            process_task(t);
        }
    }

    void process_task(TS::Task* task) {
        if (task) {
            LOG_INFO("processing task(name:", task->task_name, ") in cuda stream worker:", std::this_thread::get_id());
            bool res = task->func();
            if (res) {
                LOG_INFO(task->task_name, "done!");
            } else {
                LOG_ERROR(task->task_name, "failed! please check error");
            }
        }
    }
};

class CUDAStream final : public IStream {
public:
    static std::shared_ptr<CUDAStream> default_cuda_stream() {
        static auto stream = std::make_shared<CUDAStream>();
        bool start_ok = stream->state_->start_worker();
        if (!start_ok) {
            LOG_ERROR("cuda stream worker start error!");
        }
        return stream;
    }

    void submit(TS::Task* task) const override {
        state_->push_task(task);
    }

    void wait_event(IEvent* ev) override {
        if (ev) {
            ev->synchronize();
        }
    }

    IEvent* record_event() override {
        return new CUDAEvent{};
    }

    void synchronize() override {
        // 在 CUDA 中，通常同步可以通过事件和流来完成
        for (auto& ev : events) {
            wait_event(ev.get());
        }
    }

    DI device() const override {
        return DI::cuda();
    }

private:
    CUDAStream() : state_(std::make_unique<CUDAStreamState>()) {}

    std::unique_ptr<CUDAStreamState> state_;
    std::vector<std::shared_ptr<IEvent>> events;  // 存储所有事件的列表
};


}
#endif