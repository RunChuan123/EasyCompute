#pragma once



#include <functional>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <memory>
#include <string>
#include <condition_variable>

#include "tensor/device.hpp"
#include "backends/abstract.hpp"


namespace EC::TS
{
    
enum class TaskType{
    Default,
    Alloc,
    Memcpy,
    Compute,
    Free,
    Barrier,
    HostCallback,
    Custom,
};


enum class Priority{
    High=0,
    Normal,
    Low
};

enum class TaskStatus{
    Created,
    WaitingDeps,
    Running,
    Ready,
    Submitted,
    Finished,
    Failed,
    Cancelled,
};

// 任务单元
struct Task{
    uint64_t task_id{0};
    std::string task_name;
    TaskType task_type{TaskType::Custom};
    
    Priority priority{Priority::Normal};
    int order{0}; // 相同优先级，order 小的先执行；

    // 任务依赖前驱和后继
    std::vector<std::shared_ptr<Task>> dependencies; // 当前任务依赖的任务
    std::vector<std::shared_ptr<Task>> dependents; // 依赖当前任务的任务
    std::function<bool()> func;
    // 剩余多少依赖未完成
    std::atomic<int> pending_dependencies{0};
    
    DI exec_device{DI::cpu()};
    Dev::StreamType stream_type{Dev::StreamType::Default};
    // std::shared_ptr<Dev::IEvent> start_event;
    std::shared_ptr<Dev::IEvent> end_event;
    std::atomic<TaskStatus> status{TaskStatus::Created};
    std::string error_msg;

    Task()=default;
    ~Task()=default;
    Task(const Task&)=delete;
    Task& operator=(const Task&) = delete;

    Task(Task&& o)noexcept{
        *this = std::move(o);
    }
    Task& operator=(Task&& o) noexcept{
        task_name = std::move(o.task_name);
        task_type = o.task_type;
        priority = o.priority;
        order = o.order;
        exec_device = o.exec_device;
        stream_type = o.stream_type;
        dependencies = std::move(o.dependencies);
        dependents = std::move(o.dependents);
        pending_dependencies.store(o.pending_dependencies.load(std::memory_order_relaxed),std::memory_order_relaxed);
        func = std::move(o.func);
        // start_event = o.start_event;
        end_event = o.end_event;
        // o.start_event.reset();
        o.end_event.reset();
        o.status.store(TaskStatus::Cancelled, std::memory_order_relaxed);
        return *this;
    }
};

inline std::shared_ptr<Task> make_task(
    uint64_t id,std::string name,DI device,TaskType type,
    Dev::StreamType stream_type,Priority prio,std::function<bool()> func){
        std::shared_ptr<Task> t = std::make_shared<Task>();
        t->task_id = id;
        t->task_name = name;
        t->exec_device = device;
        t->task_type = type;
        t->stream_type = stream_type;
        t->priority = prio;
        t->func = std::move(func);
        return t;
}

struct TaskComparator{
    // 越小优先级越高
    bool operator()(const Task* a,const Task* b){
        if(a->priority != b->priority){return static_cast<int>(a->priority) > static_cast<int>(b->priority);}
        return a->order > b->order;
    }
};

} // namespace EC::TS