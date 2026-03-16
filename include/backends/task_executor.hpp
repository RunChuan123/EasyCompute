#pragma once


#include <functional>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <string>
#include <condition_variable>

#include "device_manager.hpp"
#include "abstract.hpp"
#include "tensor/device.hpp"
#include "util/logger.hpp"
#include "util/err.hpp"
#include "util/check_cuda.cuh"

namespace EC{
namespace Task{

    
enum class TaskType{
    Memcpy,
    Compute,
    Free,
    Barrier,
    Custom,
};


enum class Priority{
    High=0,
    Normal,
    Low
};

enum class TaskStatus{
    Created,
    Ready,
    Submitted,
    Finished,
    Failed,
    Cancelled,
};

// 任务单元
struct AsyncTask{
    std::string task_name;
    TaskType task_type{TaskType::Custom};
    Priority priority{Priority::Normal};
    int order; // 相同优先级，order 小的先执行；

    DI device;
    Dev::StreamHandle stream;
    // 任务依赖前驱和后继
    std::vector<AsyncTask*> dependencies; // 当前任务依赖的任务
    std::vector<AsyncTask*> dependents; // 依赖当前任务的任务
    std::function<bool()> func;
    // 剩余多少依赖未完成
    std::atomic<int> pending_dependencies{0};
    Dev::EventHandle start_event;
    Dev::EventHandle end_event;
    std::atomic<TaskStatus> status{TaskStatus::Created};

    AsyncTask()=default;
    ~AsyncTask()=default;
    AsyncTask(const AsyncTask&)=delete;
    AsyncTask& operator=(const AsyncTask&) = delete;

    AsyncTask(AsyncTask&& o)noexcept{
        *this = std::move(o);
    }
    AsyncTask& operator=(AsyncTask&& o) noexcept{
        task_name = std::move(o.task_name);
        task_type = o.task_type;
        priority = o.priority;
        order = o.order;
        device = o.device;
        stream = o.stream;
        dependencies = std::move(o.dependencies);
        dependents = std::move(o.dependents);
        pending_dependencies.store(o.pending_dependencies.load(std::memory_order_relaxed),std::memory_order_relaxed);
        func = std::move(o.func);
        start_event = o.start_event;
        end_event = o.end_event;
        o.start_event.reset();
        o.end_event.reset();
        o.status.store(TaskStatus::Cancelled, std::memory_order_relaxed);
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
    using ReadyQueue = std::priority_queue<AsyncTask*,std::vector<AsyncTask*>,TaskComparator>;
    ReadyQueue ready_queue_;
    inline static std::unique_ptr<AsyncTaskExecutor> instance_ = nullptr;
    inline static std::mutex instance_mtx_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    // 已提交的任务
    std::unordered_map<std::string,std::unique_ptr<AsyncTask>> tasks_;
    std::atomic<std::uint64_t> total_tasks_{0};
    std::atomic<std::uint64_t> finished_tasks_{0};
    std::atomic<bool> stop_{false};
    AsyncTaskExecutor() = default;
private:
    void validateRegisterInputsUnlocked(const std::string& task_name,const std::vector<AsyncTask*>& dependencies);
    bool wouldIntroduceCycleUnlocked(const std::vector<AsyncTask*>& dependencies,const AsyncTask* new_task) const;
    bool isReachableUnlocked(const AsyncTask* src, const AsyncTask* dst) const;
    void enqueueReadyUnlocked(AsyncTask* task);
    void markTaskFinishedUnlocked(AsyncTask* task);
    void markTaskFailedUnlocked(AsyncTask* task);
    void notifyDependentsUnlocked(AsyncTask* task);
    void ensureTaskEventsCreated(AsyncTask* task);
    bool executeTaskOutsideLock(AsyncTask* task);
    AsyncTask* popReadyTaskUnlocked();
public:
    ~AsyncTaskExecutor();
    AsyncTaskExecutor(const AsyncTaskExecutor&) = delete;
    AsyncTaskExecutor& operator=(const AsyncTaskExecutor&) = delete;
    static AsyncTaskExecutor& get_instance() {
        std::lock_guard<std::mutex> lock(instance_mtx_);
        if (!instance_) {
            instance_ = std::unique_ptr<AsyncTaskExecutor>(new AsyncTaskExecutor());
        }
        return *instance_;
    }

    AsyncTask* registerTask(const std::string& task_name,
                            TaskType task_type,
                            Priority priority,
                            int order,
                            DI dev,
                            Dev::StreamHandle stream,
                            std::function<bool()> func,
                            const std::vector<AsyncTask*>& dependencies = {});

    void executeReadyTasks();
    void executeUntilIdle();

    void waitTask(const std::string& task_name);
    bool waitTask(const std::string& task_name, int timeout_ms);

    void waitAllTasks();

    bool isTaskDone(const std::string& task_name);
    bool isTaskCompleted(const std::string& task_name);
    bool isTaskFailed(const std::string& task_name);

    void clearAllTasks();

    std::size_t numTasks() const;
    std::size_t numReadyTasks() const;
};

}
} 
