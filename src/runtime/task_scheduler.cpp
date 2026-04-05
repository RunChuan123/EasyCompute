
#include "backends/control.hpp"
#include "runtime/task_scheduler.hpp"
#include "backends/device_manager.hpp"

namespace EC::TS {

inline std::shared_ptr<Task> TaskScheduler::register_task(
    const std::string& task_name,
    TaskType task_type,
    Priority priority,
    int order,
    DI exec_device,
    Dev::StreamType stream_type,
    std::function<bool()> func,
    const std::vector<std::shared_ptr<Task>>& dependencies
) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto task = std::make_shared<Task>();
    task->task_name = task_name;
    task->task_type = task_type;
    task->priority = priority;
    task->order = order;
    task->exec_device = exec_device;
    task->stream_type = stream_type;
    task->func = std::move(func);

    task->dependencies = dependencies;
    task->pending_dependencies.store(static_cast<int>(dependencies.size()), std::memory_order_release);

    for (std::shared_ptr<Task> dep : dependencies) {
        if (!dep) continue;
        dep->dependents.push_back(task);
    }

    ensure_task_events(task);

    if (task->pending_dependencies.load(std::memory_order_acquire) == 0) {
        task->status.store(TaskStatus::Ready, std::memory_order_release);
        enqueue_ready_unlocked(task);
    } else {
        task->status.store(TaskStatus::WaitingDeps, std::memory_order_release);
    }

    std::shared_ptr<Task> raw = task;
    tasks_[task_name] = task;
    total_tasks_.fetch_add(1, std::memory_order_relaxed);

    cv_.notify_all();
    return raw;
}

inline void TaskScheduler::ensure_task_events(std::shared_ptr<Task> task) {
    if (!task) return;
    if (task->end_event) return;

    auto stream = Dev::Control::get_instance().default_stream(task->exec_device);
    if (!stream) {
        LOG_ERROR("cannot create event for task ", task->task_name);
        return;
    }

    switch (task->exec_device.type()) {
        case DeviceType::CPU:
            task->end_event = stream->recordCPUEvent();
            break;
        case DeviceType::CUDA:
            task->end_event = stream->recordCUDAEvent();
            break;
        default:
            LOG_ERROR("unsupported device when creating event for task ", task->task_name);
            break;
    }
}

inline void TaskScheduler::dispatch_ready_tasks() {
    while (true) {
        std::shared_ptr<Task> task = nullptr;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (ready_queue_.empty()) break;
            task = pop_ready_unlocked();
            if (!task) break;
            task->status.store(TaskStatus::Submitted, std::memory_order_release);
        }

        bool ok = Dev::Control::get_instance().dispatch(task);
        if (!ok) {
            on_task_finished(task, false);
        }
    }
}

inline std::shared_ptr<Task> TaskScheduler::pop_ready_unlocked() {
    if (ready_queue_.empty()) return nullptr;
    std::shared_ptr<Task> task = ready_queue_.top();
    ready_queue_.pop();
    return task;
}

inline void TaskScheduler::enqueue_ready_unlocked(std::shared_ptr<Task> task) {
    if (!task) return;
    ready_queue_.push(task);
}

inline void TaskScheduler::on_task_finished(std::shared_ptr<Task> task, bool success) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!task) return;

    task->status.store(
        success ? TaskStatus::Finished : TaskStatus::Failed,
        std::memory_order_release
    );

    finished_tasks_.fetch_add(1, std::memory_order_relaxed);

    notify_dependents_unlocked(task);
    cv_.notify_all();
}

inline void TaskScheduler::notify_dependents_unlocked(std::shared_ptr<Task> task) {
    for (std::shared_ptr<Task> dep : task->dependents) {
        if (!dep) continue;

        int old = dep->pending_dependencies.fetch_sub(1, std::memory_order_acq_rel);
        if (old == 1) {
            dep->status.store(TaskStatus::Ready, std::memory_order_release);
            enqueue_ready_unlocked(dep);
        }
    }
}

inline void TaskScheduler::wait_task(const std::string& task_name) {
    Task* task = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = tasks_.find(task_name);
        if (it == tasks_.end()) return;
        task = it->second.get();
    }

    if (task && task->end_event) {
        task->end_event->synchronize();
    }
}

inline void TaskScheduler::wait_all() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() {
        return finished_tasks_.load(std::memory_order_acquire) ==
               total_tasks_.load(std::memory_order_acquire);
    });
}

}
// TaskExecutor::~TaskExecutor() {
//     try {
//         clearAllTasks();
//     } catch (...) {
//     }
// }

// void TaskExecutor::validateRegisterInputsUnlocked(
//     const std::string& task_name,
//     const std::vector<std::shared_ptr<Task>>& dependencies) {

//     if (task_name.empty()) {
//         throw std::runtime_error("registerTask failed: empty task_name");
//     }

//     if (tasks_.find(task_name) != tasks_.end()) {
//         throw std::runtime_error("registerTask failed: duplicated task_name: " + task_name);
//     }

//     for (auto* dep : dependencies) {
//         if (dep == nullptr) {
//             throw std::runtime_error("registerTask failed: nullptr dependency in " + task_name);
//         }
//     }
// }

// /**
//  * 从 src 到 dst
//  */
// bool TaskExecutor::isReachableUnlocked(const std::shared_ptr<Task> src, const std::shared_ptr<Task> dst) const {
//     if (src == nullptr || dst == nullptr) return false;
//     if (src == dst) return true;

//     std::vector<const std::shared_ptr<Task>> stack;
//     stack.push_back(src);

//     std::unordered_map<const std::shared_ptr<Task>, bool> visited;
//     visited[src] = true;

//     while (!stack.empty()) {
//         auto* cur = stack.back();
//         stack.pop_back();

//         if (cur == dst) return true;

//         for (auto* next : cur->dependents) {
//             if (next && !visited[next]) {
//                 visited[next] = true;
//                 stack.push_back(next);
//             }
//         }
//     }
//     return false;
// }

// bool TaskExecutor::wouldIntroduceCycleUnlocked(
//     const std::vector<std::shared_ptr<Task>>& dependencies,
//     const std::shared_ptr<Task> new_task) const {

//     // 对于“新任务只从已有任务指向自己”的注册过程，
//     // 理论上不会形成环。这里保留接口是为了以后扩展动态加边。
//     (void)dependencies;
//     (void)new_task;
//     return false;
// }

// void TaskExecutor::enqueueReadyUnlocked(std::shared_ptr<Task> task) {
//     if (task == nullptr) return;

//     auto st = task->status.load(std::memory_order_acquire);
//     if (st == TaskStatus::Created || st == TaskStatus::Ready) {
//         task->status.store(TaskStatus::Ready, std::memory_order_release);
//         ready_queue_.push(task);
//         cv_.notify_all();
//     }
// }

// void TaskExecutor::markTaskFinishedUnlocked(std::shared_ptr<Task> task) {
//     if (task == nullptr) return;
//     task->status.store(TaskStatus::Finished, std::memory_order_release);
//     ++finished_tasks_;
//     cv_.notify_all();
// }

// void TaskExecutor::markTaskFailedUnlocked(std::shared_ptr<Task> task) {
//     if (task == nullptr) return;
//     task->status.store(TaskStatus::Failed, std::memory_order_release);
//     ++finished_tasks_;
//     cv_.notify_all();
// }

// void TaskExecutor::notifyDependentsUnlocked(std::shared_ptr<Task> task) {
//     if (task == nullptr) return;

//     for (auto* dependent : task->dependents) {
//         if (dependent == nullptr) continue;

//         int old = dependent->pending_dependencies.fetch_sub(1, std::memory_order_acq_rel);
//         if (old <= 0) {
//             throw std::runtime_error("notifyDependentsUnlocked: pending_dependencies underflow: " +
//                                      dependent->task_name);
//         }

//         if (old == 1) {
//             enqueueReadyUnlocked(dependent);
//         }
//     }
// }

// void TaskExecutor::ensureTaskEventsCreated(std::shared_ptr<Task> task) {
//     if (task == nullptr) return;

//     auto& dm = DM::get_instance();

//     if (!task->start_event.valid()) {
//         task->start_event = dm.createEvent(task->device, false);
//     }
//     if (!task->end_event.valid()) {
//         task->end_event = dm.createEvent(task->device, false);
//     }
// }

// bool TaskExecutor::executeTaskOutsideLock(std::shared_ptr<Task> task) {
//     if (task == nullptr) {
//         throw std::runtime_error("executeTaskOutsideLock: nullptr task");
//     }

//     auto& dm = Dev::DeviceManager::get_instance();

//     // 先让当前 task 所在 stream 等待所有前驱的完成事件
//     for (auto* dep : task->dependencies) {
//         if (dep == nullptr) {
//             throw std::runtime_error("executeTaskOutsideLock: nullptr dependency in task: " +
//                                      task->task_name);
//         }
//         if (dep->end_event.valid()) {
//             dm.waitEvent(task->stream, dep->end_event);
//         }
//     }

//     dm.recordEvent(task->start_event, task->stream);

//     bool ok = true;
//     if (task->func) {
//         ok = task->func();
//     }

//     dm.recordEvent(task->end_event, task->stream);
//     return ok;
// }

// std::shared_ptr<Task> TaskExecutor::popReadyTaskUnlocked() {
//     while (!ready_queue_.empty()) {
//         std::shared_ptr<Task> task = ready_queue_.top();
//         ready_queue_.pop();

//         if (task == nullptr) continue;

//         auto st = task->status.load(std::memory_order_acquire);
//         if (st == TaskStatus::Ready) {
//             task->status.store(TaskStatus::Submitted, std::memory_order_release);
//             return task;
//         }
//     }
//     return nullptr;
// }

// std::shared_ptr<Task> TaskExecutor::registerTask(const std::string& task_name,
//                                            TaskType task_type,
//                                            Priority priority,
//                                            int order,
//                                            DI dev,
//                                            Dev::IStream stream,
//                                            std::function<bool()> func,
//                                            const std::vector<std::shared_ptr<Task>>& dependencies) {
//     std::lock_guard<std::mutex> lock(mtx_);

//     validateRegisterInputsUnlocked(task_name, dependencies);

//     auto task = std::make_unique<Task>();
//     task->task_name = task_name;
//     task->task_type = task_type;
//     task->priority = priority;
//     task->order = order;
//     task->device = dev;
//     task->stream = stream;
//     task->func = std::move(func);
//     task->dependencies = dependencies;
//     task->pending_dependencies.store(static_cast<int>(dependencies.size()),
//                                      std::memory_order_release);
//     task->status.store(TaskStatus::Created, std::memory_order_release);

//     std::shared_ptr<Task> raw = task.get();

//     if (wouldIntroduceCycleUnlocked(dependencies, raw)) {
//         throw std::runtime_error("registerTask failed: cycle detected for task: " + task_name);
//     }

//     for (auto* dep : dependencies) {
//         dep->dependents.push_back(raw);
//     }

//     tasks_.emplace(task_name, std::move(task));
//     ++total_tasks_;

//     if (raw->pending_dependencies.load(std::memory_order_acquire) == 0) {
//         enqueueReadyUnlocked(raw);
//     }

//     return raw;
// }

// void TaskExecutor::executeReadyTasks() {
//     while (true) {
//         std::shared_ptr<Task> task = nullptr;

//         {
//             std::lock_guard<std::mutex> lock(mtx_);
//             task = popReadyTaskUnlocked();
//             if (task == nullptr) {
//                 return;
//             }
//             ensureTaskEventsCreated(task);
//         }

//         bool ok = false;
//         try {
//             ok = executeTaskOutsideLock(task);
//         } catch (...) {
//             std::lock_guard<std::mutex> lock(mtx_);
//             markTaskFailedUnlocked(task);
//             throw;
//         }

//         {
//             std::lock_guard<std::mutex> lock(mtx_);
//             if (ok) {
//                 markTaskFinishedUnlocked(task);
//                 notifyDependentsUnlocked(task);
//             } else {
//                 markTaskFailedUnlocked(task);
//             }
//         }
//     }
// }

// void TaskExecutor::executeUntilIdle() {
//     executeReadyTasks();
// }

// void TaskExecutor::waitTask(const std::string& task_name) {
//     std::shared_ptr<Task> task = nullptr;

//     {
//         std::lock_guard<std::mutex> lock(mtx_);
//         auto it = tasks_.find(task_name);
//         if (it == tasks_.end()) {
//             throw std::runtime_error("waitTask failed: task not found: " + task_name);
//         }
//         task = it->second.get();
//     }

//     while (true) {
//         executeReadyTasks();

//         {
//             std::lock_guard<std::mutex> lock(mtx_);
//             auto st = task->status.load(std::memory_order_acquire);
//             if (st == TaskStatus::Finished) {
//                 break;
//             }
//             if (st == TaskStatus::Failed || st == TaskStatus::Cancelled) {
//                 throw std::runtime_error("waitTask failed: task execution failed: " + task_name);
//             }
//         }

//         std::unique_lock<std::mutex> lk(mtx_);
//         cv_.wait_for(lk, std::chrono::milliseconds(1));
//     }

//     if (task->end_event.valid()) {
//         Dev::DeviceManager::get_instance().synchronize(task->end_event);
//     }
// }

// bool TaskExecutor::waitTask(const std::string& task_name, int timeout_ms) {
//     std::shared_ptr<Task> task = nullptr;
//     auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

//     {
//         std::lock_guard<std::mutex> lock(mtx_);
//         auto it = tasks_.find(task_name);
//         if (it == tasks_.end()) {
//             throw std::runtime_error("waitTask(timeout) failed: task not found: " + task_name);
//         }
//         task = it->second.get();
//     }

//     while (std::chrono::steady_clock::now() < deadline) {
//         executeReadyTasks();

//         {
//             std::lock_guard<std::mutex> lock(mtx_);
//             auto st = task->status.load(std::memory_order_acquire);
//             if (st == TaskStatus::Finished) {
//                 if (task->end_event.valid()) {
//                     Dev::DeviceManager::get_instance().synchronize(task->end_event);
//                 }
//                 return true;
//             }
//             if (st == TaskStatus::Failed || st == TaskStatus::Cancelled) {
//                 return false;
//             }
//         }

//         std::unique_lock<std::mutex> lk(mtx_);
//         cv_.wait_for(lk, std::chrono::milliseconds(1));
//     }

//     return false;
// }

// void TaskExecutor::waitAllTasks() {
//     while (true) {
//         executeReadyTasks();

//         {
//             std::lock_guard<std::mutex> lock(mtx_);
//             if (finished_tasks_.load(std::memory_order_acquire) ==
//                 total_tasks_.load(std::memory_order_acquire)) {
//                 break;
//             }
//         }

//         std::unique_lock<std::mutex> lk(mtx_);
//         cv_.wait_for(lk, std::chrono::milliseconds(1));
//     }

//     // 最后确保所有设备事件都完成
//     std::lock_guard<std::mutex> lock(mtx_);
//     for (auto& kv : tasks_) {
//         std::shared_ptr<Task> task = kv.second.get();
//         if (task && task->end_event.valid()) {
//             Dev::DeviceManager::get_instance().synchronize(task->end_event);
//         }
//     }
// }

// bool TaskExecutor::isTaskDone(const std::string& task_name) {
//     std::lock_guard<std::mutex> lock(mtx_);
//     auto it = tasks_.find(task_name);
//     if (it == tasks_.end()) return false;

//     auto st = it->second->status.load(std::memory_order_acquire);
//     return st == TaskStatus::Finished ||
//            st == TaskStatus::Failed ||
//            st == TaskStatus::Cancelled;
// }

// bool TaskExecutor::isTaskCompleted(const std::string& task_name) {
//     std::lock_guard<std::mutex> lock(mtx_);
//     auto it = tasks_.find(task_name);
//     if (it == tasks_.end()) return false;

//     std::shared_ptr<Task> task = it->second.get();
//     if (!task->end_event.valid()) {
//         return false;
//     }
//     return Dev::DeviceManager::get_instance().queryEvent(task->end_event);
// }

// bool TaskExecutor::isTaskFailed(const std::string& task_name) {
//     std::lock_guard<std::mutex> lock(mtx_);
//     auto it = tasks_.find(task_name);
//     if (it == tasks_.end()) return false;

//     return it->second->status.load(std::memory_order_acquire) == TaskStatus::Failed;
// }

// void TaskExecutor::clearAllTasks() {

//     std::lock_guard<std::mutex> lock(mtx_);


//     ReadyQueue empty_queue;
//     ready_queue_.swap(empty_queue);

//     auto& dm = DM::get_instance();
//     for (auto& kv : tasks_) {
//         std::shared_ptr<Task> task = kv.second.get();
//         if (!task) continue;

//         if (task->start_event.valid()) {
//             dm.destroyEvent(task->start_event);
//             task->start_event.reset();
//         }
//         if (task->end_event.valid()) {
//             dm.destroyEvent(task->end_event);
//             task->end_event.reset();
//         }
//         task->status.store(TaskStatus::Cancelled, std::memory_order_release);
//     }

//     tasks_.clear();
//     total_tasks_.store(0, std::memory_order_release);
//     finished_tasks_.store(0, std::memory_order_release);
//     cv_.notify_all();
// }

// std::size_t TaskExecutor::numTasks() const {
//     std::lock_guard<std::mutex> lock(mtx_);
//     return tasks_.size();
// }

// std::size_t TaskExecutor::numReadyTasks() const {
//     std::lock_guard<std::mutex> lock(mtx_);
//     return ready_queue_.size();
// }

// } // namespace Func
// } // namespace EC