#include "tensor/function.cuh"

namespace EC{
namespace Func{





bool FunctionManager::checkDependencied(const AsyncTask* task){
    for(const auto& dep : task->dependencies){
        cudaError_t _err = cudaEventQuery(dep->end_event);
        if(_err != cudaSuccess && _err != cudaErrorNotReady){
            CUDA_CHECK(_err);return false;
        }
        if(_err == cudaErrorNotReady)return false;
    }
    return true;
}
void FunctionManager::setStreamDependency(cudaStream_t cur_stream,cudaStream_t dep_stream){
    if(cur_stream == nullptr || dep_stream == nullptr)return;
    stream_deps_[cur_stream].push_back(dep_stream);
    CUDA_CHECK(cudaStreamWaitEvent(cur_stream,getStreamDoneEvent(dep_stream)));
}
cudaEvent_t FunctionManager::getStreamDoneEvent(cudaStream_t stream){
    // 每个流对应的 end_event
    static std::unordered_map<cudaStream_t, cudaEvent_t> stream_end_event_;
    if(stream_end_event_.find(stream) != stream_end_event_.end()){
        cudaEvent_t evt;
        cudaEventCreate(&evt);
        stream_end_event_[stream] = evt;
    }
    CUDA_CHECK(cudaEventRecord(stream_end_event_[stream],stream));
    return stream_end_event_[stream];
}

void FunctionManager::waitTask(const std::string& task_name){
    std::lock_guard<std::mutex> lock(mtx_);
    if (tasks_.find(task_name) == tasks_.end()) {
        throw std::runtime_error("waitTask error: Task '" + task_name + "' not found");
    }
    auto& task = tasks_[task_name];
    CUDA_CHECK(cudaEventSynchronize(task->end_event));
    LOG_WARN("Task ",task_name," has completed!");
}

bool FunctionManager::waitTask(const std::string& task_name, int timeout_ms) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (tasks_.find(task_name) == tasks_.end()) {
        throw std::runtime_error("waitTask error: Task '" + task_name + "' not found");
    }
    auto& task = tasks_[task_name];
    
    const auto start_time = std::chrono::steady_clock::now();
    while (true) {
        cudaError_t err = cudaEventQuery(task->end_event);
        if (err == cudaSuccess) {
            LOG_WARN("Task ",task_name," has completed!", "timeout: ",timeout_ms,"ms)");
            return true; 
        }
        if (err != cudaErrorNotReady) {
            CUDA_CHECK(err);
            return false;
        }
        // 检查是否超时
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time
        ).count();
        if (elapsed >= timeout_ms) {
            LOG_WARN("waitTask timeout: Task ",task_name,"not completed in",timeout_ms,"ms)");
            return false;
        }
        // 避免空循环占用CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
AsyncTask* FunctionManager::registerTask(const std::string& task_name,
    FuncType func_type,
    Priority priority,
    int order,
    Device dev,
    cudaStream_t stream,
    std::function<bool()> func,
    const std::vector<AsyncTask*>& dependencies ){
        std::lock_guard<std::mutex> lock(mtx_);
        
        // 创建新任务
        auto task = std::make_unique<AsyncTask>();
        task->task_name = task_name;
        task->func_type = func_type;
        task->priority = priority;
        task->order = order;
        task->device = dev;
        task->stream = stream;
        task->dependencies = dependencies;
        task->func = std::move(func);

        AsyncTask* task_ptr = task.get();
        tasks_[task_name] = std::move(task);
        
        // 添加到优先级队列
        task_queue_.push(task_ptr);
        
        // 如果任务有依赖，同步流
        for (const auto& dep : dependencies) {
            setStreamDependency(task_ptr->stream, dep->stream);
        }
        return task_ptr;
}
/**
 * 只执行就绪的任务
 */
void FunctionManager::executeReadyTasks(){
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<AsyncTask*> temp_queue;
    while(!task_queue_.empty()){
        AsyncTask* task = task_queue_.top();
        task_queue_.pop();
        if(!checkDependencied(task)){
            temp_queue.push_back(task);
            continue;
        }
        if(task->stream != nullptr) cudaEventRecord(task->start_event,task->stream);
        if (task->device.type() == DeviceType::CUDA) cudaSetDevice(task->device.id());
        // 执行回调函数
        bool success = task->func();
        if(!success){
            LOG_WARN("Task",task->task_name," execute failed!");
            continue;
        }
        if(task->stream != nullptr) cudaEventRecord(task->end_event,task->stream);
        LOG_INFO("Task",task->task_name," executed successfully!");
    }
    // 剩下的任务继续留在队列
    for(auto& task : temp_queue){
            task_queue_.push(task);
        }
}

void FunctionManager::waitAllTasks(){
    std::lock_guard<std::mutex> lock(mtx_);
    for(auto& [name,task]:tasks_){
        CUDA_CHECK(cudaEventSynchronize(task->end_event));
    }
}
bool FunctionManager::isTaskDone(const std::string& name){
    std::lock_guard<std::mutex> lock(mtx_);
    if (tasks_.find(name) == tasks_.end()) return false;
    auto& task = tasks_[name];
    return cudaEventQuery(task->end_event) == cudaSuccess;
}
void FunctionManager::clearAllTasks(){
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.clear();
    while (!task_queue_.empty()) {
        task_queue_.pop();
    }
    stream_deps_.clear();
}

}
}