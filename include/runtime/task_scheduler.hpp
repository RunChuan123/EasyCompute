#pragma once


#include <functional>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <memory>
#include <string>
#include <condition_variable>

#include "backends/device_manager.hpp"
#include "runtime/task.hpp"
#include "abstract.hpp"
#include "backends/abstract.hpp"
#include "tensor/device.hpp"
#include "util/logger.hpp"
#include "util/err.hpp"
// #include "util/check_cuda.cuh"

namespace EC{
namespace TS{


struct Scheduler{

};

// struct TaskScheduler{
// private:
//     using ReadyQueue = std::priority_queue<std::shared_ptr<Task>,std::vector<std::shared_ptr<Task>>,TaskComparator>;
//     ReadyQueue ready_queue_;
//     inline static std::unique_ptr<TaskScheduler> instance_ = nullptr;
//     inline static std::mutex instance_mtx_;
//     mutable std::mutex mtx_;
//     std::condition_variable cv_;
//     // 已提交的任务
//     std::unordered_map<std::string,std::unique_ptr<Task>> tasks_;
//     std::atomic<std::uint64_t> total_tasks_{0};
//     std::atomic<std::uint64_t> finished_tasks_{0};
//     std::atomic<bool> stop_{false};
//     TaskScheduler() = default;
// private:
//     void validateRegisterInputsUnlocked(const std::string& task_name,const std::vector<std::shared_ptr<Task>>& dependencies);
//     bool wouldIntroduceCycleUnlocked(const std::vector<std::shared_ptr<Task>>& dependencies,const std::shared_ptr<Task> new_task) const;
//     bool isReachableUnlocked(const std::shared_ptr<Task> src, const std::shared_ptr<Task> dst) const;
//     void enqueueReadyUnlocked(std::shared_ptr<Task> task);
//     void markTaskFinishedUnlocked(std::shared_ptr<Task> task);
//     void markTaskFailedUnlocked(std::shared_ptr<Task> task);
//     void notifyDependentsUnlocked(std::shared_ptr<Task> task);
//     void ensureTaskEventsCreated(std::shared_ptr<Task> task);
//     bool executeTaskOutsideLock(std::shared_ptr<Task> task);
//     std::shared_ptr<Task> popReadyTaskUnlocked();
// public:
//     ~TaskScheduler();
//     TaskScheduler(const TaskScheduler&) = delete;
//     TaskScheduler& operator=(const TaskScheduler&) = delete;
//     static TaskScheduler& get_instance() {
//         std::lock_guard<std::mutex> lock(instance_mtx_);
//         if (!instance_) {
//             instance_ = std::unique_ptr<TaskScheduler>(new TaskScheduler());
//         }
//         return *instance_;
//     }

//     std::shared_ptr<Task> registerTask(const std::string& task_name,
//                             TaskType task_type,
//                             Priority priority,
//                             int order,
//                             DI dev,
//                             Dev::IStream stream,
//                             std::function<bool()> func,
//                             const std::vector<std::shared_ptr<Task>>& dependencies = {});

//     void executeReadyTasks();
//     void executeUntilIdle();

//     void waitTask(const std::string& task_name);
//     bool waitTask(const std::string& task_name, int timeout_ms);

//     void waitAllTasks();

//     bool isTaskDone(const std::string& task_name);
//     bool isTaskCompleted(const std::string& task_name);
//     bool isTaskFailed(const std::string& task_name);

//     void clearAllTasks();

//     std::size_t numTasks() const;
//     std::size_t numReadyTasks() const;
// };

class TaskScheduler final {
public:
    using ReadyQueue = std::priority_queue<std::shared_ptr<Task>, std::vector<std::shared_ptr<Task>>, TaskComparator>;

    static TaskScheduler& get_instance() {
        static TaskScheduler inst;
        return inst;
    }

    TaskScheduler(const TaskScheduler&) = delete;
    TaskScheduler& operator=(const TaskScheduler&) = delete;

    std::shared_ptr<Task> register_task(
        const std::string& task_name,
        TaskType task_type,
        Priority priority,
        int order,
        DI exec_device,
        Dev::StreamType stream_type,
        std::function<bool()> func,
        const std::vector<std::shared_ptr<Task>>& dependencies = {}
    );

    void dispatch_ready_tasks();
    void dispatch_until_idle();

    void on_task_finished(std::shared_ptr<Task> task, bool success);

    void wait_task(const std::string& task_name);
    void wait_all();

    void clear();

private:
    TaskScheduler() = default;
    ~TaskScheduler() = default;

    void enqueue_ready_unlocked(std::shared_ptr<Task> task);
    std::shared_ptr<Task> pop_ready_unlocked();
    void notify_dependents_unlocked(std::shared_ptr<Task> task);
    void ensure_task_events(std::shared_ptr<Task> task);

private:
    mutable std::mutex mtx_;
    std::condition_variable cv_;

    ReadyQueue ready_queue_;
    std::unordered_map<std::string, std::shared_ptr<Task>> tasks_;

    std::atomic<std::size_t> total_tasks_{0};
    std::atomic<std::size_t> finished_tasks_{0};
};
}
} 
