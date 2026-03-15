#pragma once


#include <functional>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>

#include "manager.hpp"
#include "abstract.hpp"
#include "tensor/device.hpp"
#include "util/logger.hpp"
#include "util/err.hpp"
#include "util/check_cuda.cuh"

namespace EC{
namespace Func{

    
enum class TaskType{
    Memcpy,
    Compute,
    Free,
    Barrier,
    Custom,
};

// struct MemcpyTaskDesc {
//     void* dst;
//     const void* src;
//     size_t bytes;
//     Device dst_device;
//     Device src_device;
// };

// struct KernelTaskDesc {
//     std::string op_name;
//     std::vector<void*> args;
// };

enum class Priority{
    High=0,
    Normal,
    Low
};

// 任务单元
struct AsyncTask{
    std::string task_name;
    TaskType func_type;
    Priority priority;
    int order; // 相同优先级，order 小的先执行；

    DI device;
    Dev::StreamHandle stream;
    // 等待这些任务执行完毕后才可执行
    std::vector<AsyncTask*> dependencies;

    std::function<bool()> func;

    Dev::EventHandle start_event;
    Dev::EventHandle end_event;

    AsyncTask(){
        // cudaEventCreate(&start_event);
        // cudaEventCreate(&end_event);
    }
    ~AsyncTask(){
        // if (start_event) cudaEventDestroy(start_event);
        // if (end_event)cudaEventDestroy(end_event);
    }
    AsyncTask(const AsyncTask&)=delete;
    AsyncTask& operator=(const AsyncTask&) = delete;

    AsyncTask(AsyncTask&& o)noexcept{
        *this = std::move(o);
    }
    AsyncTask& operator=(AsyncTask&& o) noexcept{
        task_name = std::move(o.task_name);
        func_type = o.func_type;
        priority = o.priority;
        order = o.order;
        device = o.device;
        stream = o.stream;
        dependencies = std::move(o.dependencies);
        func = std::move(o.func);
        start_event = o.start_event;
        end_event = o.end_event;
        o.start_event.reset();
        o.end_event.reset();
        return *this;
    }
};

struct TaskComparator{
    // 越小优先级越高
    bool operator()(const AsyncTask* a,const AsyncTask* b){
        if(a->priority != b->priority){return static_cast<int>(a->priority) > static_cast<int>(b->priority);}
        return a->order > b->order;
    }
};

struct AsyncTaskExecutor{
private:
    static std::unique_ptr<AsyncTaskExecutor> instance_;
    static std::mutex mtx_;
    // 任务队列
    std::priority_queue<AsyncTask*,std::vector<AsyncTask*>,TaskComparator> task_queue_;
    // 已提交的任务
    std::unordered_map<std::string,std::unique_ptr<AsyncTask>> tasks_;
    // 流依赖 某stream 依赖的流s
    std::unordered_map<Dev::StreamHandle,std::vector<Dev::StreamHandle>> stream_deps_;
    AsyncTaskExecutor() = default;
    bool checkDependencied(const AsyncTask* task);
    void setStreamDependency(Dev::StreamHandle cur_stream,Dev::StreamHandle dep_stream);
    Dev::EventHandle getStreamDoneEvent(Dev::StreamHandle stream);

public:
    static AsyncTaskExecutor& get_instance(){
        std::lock_guard<std::mutex> lock(mtx_);
        if(!instance_){
            instance_.reset(new AsyncTaskExecutor{});
        }
        return *instance_;
    }
    AsyncTask* registerTask(const std::string& task_name,
        TaskType func_type,
        Priority priority,
        int order,
        DI dev,
        Dev::StreamHandle stream,
        std::function<bool()> func,
        const std::vector<AsyncTask*>& dependencies = {});
    const std::unordered_map<Dev::StreamHandle, std::vector<Dev::StreamHandle>>& getStreamDependencies() const {return stream_deps_;}
    void executeReadyTasks();
    void waitAllTasks();
    void waitTask(const std::string& task_name);
    bool waitTask(const std::string& task_name, int timeout_ms);
    bool isTaskDone(const std::string& name);
    void clearAllTasks();
    // inline bool isTaskCompleted(const std::string& task_name) {
    //     std::lock_guard<std::mutex> lock(mtx_);
    //     if (tasks_.find(task_name) == tasks_.end()) return false;
    //     auto& task = tasks_[task_name];
    //     return cudaEventQuery(task->end_event) == cudaSuccess;
    // }
    ~AsyncTaskExecutor(){
        clearAllTasks();
    }
};

std::unique_ptr<AsyncTaskExecutor> AsyncTaskExecutor::instance_ = nullptr;
std::mutex AsyncTaskExecutor::mtx_;
}
} 
