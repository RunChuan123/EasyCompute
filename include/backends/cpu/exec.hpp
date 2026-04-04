#pragma once

#include <mutex>
#include <condition_variable>
#include <memory>
#include <deque>
#include <atomic>

#include "runtime/task_executor.hpp"
#include "util/logger.hpp"

namespace EC::Dev::CPU
{

struct CPUEvent final:public IEvent{
        CPUEvent() {
        // 初始化内部同步机制
        std::lock_guard<std::mutex> lock(mtx_);
        completed_ = false;
    }

    // 标记事件为完成
    void set_complete() {
        std::lock_guard<std::mutex> lock(mtx_);
        completed_ = true;
        cv_.notify_all();
    }

    // 查询事件是否完成
    bool query() const override {
        std::lock_guard<std::mutex> lock(mtx_);
        return completed_;
    }

    // 阻塞直到事件完成
    void synchronize() override {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return completed_; });
    }

    DI device() const{
        return DI::cpu();
    }

private:
    bool completed_{false};
    mutable std::mutex mtx_;
    std::condition_variable cv_;
};

struct CPUStream final : public IStream{
public:
    static std::shared_ptr<CPUStream> default_cpu_stream(){
        static auto stream = std::make_shared<CPUStream>();
        bool start_ok = stream->state_->start_worker();
        if(!start_ok){
            LOG_ERROR("cpu stream worker start error!");
        }
        return stream;
    }
    void submit(TS::Task* task)const override{
        state_->push_task(task);
    }
    void wait_event(IEvent* ev)override{
        if(ev){
            ev->synchronize();
        }
    }
    IEvent* record_event()override{
        return new CPUEvent{};
    }
    void synchronize()override{

    }
    DI device() const override{
        return DI::cpu();
    }

private:
    CPUStream():state_(std::make_unique<CPUStreamState>()){}
    std::unique_ptr<CPUStreamState> state_;
};

struct CPUStreamState{
public:
    std::mutex mtx_stm;
    std::condition_variable cv_stm;
    std::deque<TS::Task*> queue_stm;
    bool running{false};
    // TODO： 先写一个 worker，未来可能扩展多线程提高并发
    std::thread worker;
    void push_task(TS::Task* t){
        if(t){
            std::lock_guard<std::mutex> lock(mtx_stm);
            queue_stm.push_back(t);
        }else{
            LOG_WARN("task",t->task_name,"is null!");
        }
    }

    bool start_worker(){
        if(!running){
            {
                std::lock_guard<std::mutex> lock(mtx_stm);
                worker = std::thread([this](){worker_function();});
                running=true;
            }
            LOG_INFO("cpu stream worke r",worker.get_id()," start runing");
        }
        return running;
    }
    void stop_worker(){
        {
            std::lock_guard<std::mutex> lock(mtx_stm);
            running=false;
        }
        cv_stm.notify_all();
        if(worker.joinable()){
            worker.join();
        }
    }
private:
    
    // 不需要用锁
    void worker_function(){
        while(true){
            TS::Task* t = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx_stm);
                cv_stm.wait(lock,[this](){return !queue_stm.empty() || !running;});
                // 如果stop set或者空（代表停了）
                if(!running || queue_stm.empty()){
                    break;
                }
                // 只有这一个线程在操作 queue_stm
                t = queue_stm.front();
                queue_stm.pop_front();
            }
            process_task(t);
        }
    }
    void process_task(TS::Task* task){
        if(task){
            LOG_INFO("processing task(name:",task->task_name,") in cpu stream worker:",std::this_thread::get_id());
            bool res = task->func();
            if(res){
                LOG_INFO(task->task_name," done!");
            }else{
                LOG_ERROR(task->task_name," failed! please check error");
            }
        }
    }

};



    
} // namespace EC::Dev::CPU
