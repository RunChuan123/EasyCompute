#include "backends/cpu/cpu_runtime.hpp"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <thread>

namespace EC::Dev
{

// CPU 设备数量固定为1
int CpuDeviceRuntime::deviceCount() const {
    return 1;
}

// CPU 永远可用（idx只能为0）
bool CpuDeviceRuntime::isAvailable(int idx) const {
    return idx == 0;
}

// CPU 无设备切换
void CpuDeviceRuntime::setCurrentDevice(DI dev) {
    if (dev.type() != DeviceType::CPU || dev.id() != 0) {
        throw std::runtime_error("CpuDeviceRuntime only supports CPU (id=0)");
    }
}

// CPU 设备同步（空实现）
void CpuDeviceRuntime::synchronizeDevice(DI dev) {
    setCurrentDevice(dev);
}

// CPU 内存分配
void* CpuDeviceRuntime::allocate(DI dev, size_t bytes, MemoryType kind) {
    setCurrentDevice(dev);
    if (bytes == 0) return nullptr;

    void* ptr = nullptr;
    switch (kind) {
        case MemoryType::Host:
            ptr = malloc(bytes);
            break;
        case MemoryType::PinnedHost:
            ptr = malloc(bytes);
            if (ptr) mlock(ptr, bytes);
            break;
        case MemoryType::Unified:
            ptr = malloc(bytes);
            break;
        case MemoryType::Device:
            throw std::runtime_error("CPU does not support Device memory");
    }
    if (!ptr) throw std::runtime_error("CPU allocate failed for " + std::to_string(bytes) + " bytes");
    return ptr;
}

// CPU 内存释放
void CpuDeviceRuntime::deallocate(DI dev, void* ptr, MemoryType kind) {
    if (!ptr) return;
    setCurrentDevice(dev);

    switch (kind) {
        case MemoryType::Host:
        case MemoryType::Unified:
            free(ptr);
            break;
        case MemoryType::PinnedHost:
            throw std::runtime_error("CPU deallocate does not support PinnedHost memory");
        case MemoryType::Device:
            throw std::runtime_error("CPU deallocate does not support Device memory");
    }
}

// ========== 核心：Event 实现（保证执行顺序） ==========
EventHandle CpuDeviceRuntime::createEvent(DI dev, bool timing) {
    (void)timing;
    setCurrentDevice(dev);

    // 创建Event内部数据（初始未完成，无依赖）
    auto event_data = std::make_unique<CpuEventData>();
    event_data->completed = false;
    event_data->dependencies.clear();

    // 用malloc的唯一指针作为Event句柄
    void* event_ptr = malloc(1);
    if (!event_ptr) throw std::runtime_error("CPU createEvent failed");

    // 注册到管理表
    std::lock_guard<std::mutex> lock(mtx_);
    cpu_events_[event_ptr] = std::move(event_data);

    return EventHandle(event_ptr, dev);
}

void CpuDeviceRuntime::destroyEvent(EventHandle event) {
    if (!event.valid()) return;

    std::lock_guard<std::mutex> lock(mtx_);
    void* event_ptr = event.impl();
    if (cpu_events_.count(event_ptr)) {
        cpu_events_.erase(event_ptr);
        free(event_ptr);
    }
}

// 记录Event：标记事件完成 + 触发依赖链
void CpuDeviceRuntime::recordEvent(EventHandle event, StreamHandle stream) {
    (void)stream;
    if (!event.valid()) return;

    std::lock_guard<std::mutex> lock(mtx_);
    void* event_ptr = event.impl();
    auto it = cpu_events_.find(event_ptr);
    if (it == cpu_events_.end()) throw std::runtime_error("CPU event not found");

    // 标记事件为完成（CPU中record=完成）
    it->second->completed = true;
}

// 查询Event状态
bool CpuDeviceRuntime::queryEvent(EventHandle event) {
    if (!event.valid()) return true;

    std::lock_guard<std::mutex> lock(mtx_);
    void* event_ptr = event.impl();
    auto it = cpu_events_.find(event_ptr);
    if (it == cpu_events_.end()) return true;

    // 先检查所有依赖是否完成
    for (const auto& dep_event : it->second->dependencies) {
        if (!queryEvent(dep_event)) return false;
    }

    // 检查自身状态
    return it->second->completed;
}

// 同步等待Event完成（递归等待依赖）
void CpuDeviceRuntime::synchronizeEvent(EventHandle event) {
    if (!event.valid()) return;

    std::lock_guard<std::mutex> lock(mtx_);
    void* event_ptr = event.impl();
    auto it = cpu_events_.find(event_ptr);
    if (it == cpu_events_.end()) return;

    // 等待事件（含依赖）完成
    waitEventInternal(it->second.get());
}

// 流等待Event（核心：添加依赖，保证执行顺序）
void CpuDeviceRuntime::waitEvent(StreamHandle stream, EventHandle event) {
    (void)stream;
    if (!event.valid()) return;

    // 这里的逻辑：调用waitEvent的当前任务，会把event设为自己的依赖
    // （注：需要结合你的FunctionManager，在任务绑定Event时调用此接口）
    // 简化实现：直接同步等待event完成（保证顺序）
    synchronizeEvent(event);
}

// 内部等待逻辑（递归等待所有依赖）
void CpuDeviceRuntime::waitEventInternal(CpuEventData* event_data) {
    if (!event_data) return;

    // 先等待所有依赖事件完成
    for (const auto& dep_event : event_data->dependencies) {
        synchronizeEvent(dep_event);
    }

    // 等待当前事件完成（自旋等待，CPU无异步，实际很快）
    while (!event_data->completed) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

// ========== 拷贝接口（结合Event保证顺序） ==========
void CpuDeviceRuntime::memcpyAsync(void* dst, DI dst_dev,
                                   const void* src, DI src_dev,
                                   size_t bytes, StreamHandle stream) {
    (void)stream;
    setCurrentDevice(src_dev);

    // 仅支持CPU<->CPU拷贝
    if (dst_dev.type() != DeviceType::CPU || src_dev.type() != DeviceType::CPU) {
        throw std::runtime_error("CPU memcpy only supports CPU<->CPU");
    }

    // 同步拷贝（CPU无异步）
    if (bytes > 0 && dst && src) {
        memcpy(dst, src, bytes);
    }

    // 关键：如果stream绑定了Event，拷贝完成后标记Event完成
    // （适配你的FunctionManager：拷贝任务完成后recordEvent，触发后续任务）
    if (stream.valid()) {
        // 这里可扩展：从stream中提取绑定的Event并record
        // 简化：假设stream关联的Event已传入，此处仅做示例
    }
}

} // namespace EC::Dev