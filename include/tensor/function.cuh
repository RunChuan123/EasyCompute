#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>

#include "device/device.hpp"
#include "util/logger.hpp"
#include "util/err.hpp"
#include "util/check_cuda.cuh"

namespace EC{
namespace Func{

    
enum class FuncType{
    Memcpy_H2D,
    Memcpy_D2H,
    Memcpy_D2D,
    Compute_Unary, // softmax...
    Compute_Binary, // add... 
    Compute_Ternary, // attention...
    Compute, // symbol
    MemFree,
    Custom, 
};

enum class Priority{
    High=0,
    Normal,
    Low
};

// 任务单元
struct AsyncTask{
    std::string task_name;
    FuncType func_type;
    Priority priority;
    int order; // 相同优先级，order 小的先执行；

    Device device;
    cudaStream_t stream = nullptr;
    // 等待这些任务执行完毕后才可执行
    std::vector<AsyncTask*> dependencies;
    std::function<bool()> func;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;

    AsyncTask(){
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
    }
    ~AsyncTask(){
        if (start_event) CUDA_CHECK(cudaEventDestroy(start_event));
        if (end_event) CUDA_CHECK(cudaEventDestroy(end_event));
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
        o.start_event = nullptr;
        o.end_event = nullptr;
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

struct FunctionManager{
private:
    static std::unique_ptr<FunctionManager> instance_;
    static std::mutex mtx_;
    // 任务队列
    std::priority_queue<AsyncTask*,std::vector<AsyncTask*>,TaskComparator> task_queue_;
    // 已提交的任务
    std::unordered_map<std::string,std::unique_ptr<AsyncTask>> tasks_;
    // 流依赖 某stream 依赖的流s
    std::unordered_map<cudaStream_t,std::vector<cudaStream_t>> stream_deps_;
    FunctionManager() = default;
    bool checkDependencied(const AsyncTask* task);
    void setStreamDependency(cudaStream_t cur_stream,cudaStream_t dep_stream);
    cudaEvent_t getStreamDoneEvent(cudaStream_t stream);

public:
    static FunctionManager& get_instance(){
        std::lock_guard<std::mutex> lock(mtx_);
        if(!instance_){
            instance_.reset(new FunctionManager{});
        }
        return *instance_;
    }
    AsyncTask* registerTask(const std::string& task_name,
        FuncType func_type,
        Priority priority,
        int order,
        Device dev,
        cudaStream_t stream,
        std::function<bool()> func,
        const std::vector<AsyncTask*>& dependencies = {});
    const std::unordered_map<cudaStream_t, std::vector<cudaStream_t>>& getStreamDependencies() const {return stream_deps_;}
    void executeReadyTasks();
    void waitAllTasks();
    void waitTask(const std::string& task_name);
    bool waitTask(const std::string& task_name, int timeout_ms);
    bool isTaskDone(const std::string& name);
    void clearAllTasks();
    ~FunctionManager(){
        clearAllTasks();
    }
};

std::unique_ptr<FunctionManager> FunctionManager::instance_ = nullptr;
std::mutex FunctionManager::mtx_;
}
} 
