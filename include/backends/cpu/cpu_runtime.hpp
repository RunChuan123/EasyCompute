#pragma once

#include <stdexcept>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <atomic>
#include "../device_runtime.hpp"
#include "../device_manager.hpp"

namespace EC::Dev
{

// CPU Event 内部结构（记录状态+依赖）
struct CpuEventData {
    std::atomic<bool> completed = false;       // 事件完成状态
    std::vector<EventHandle> dependencies;     // 事件依赖（需先完成的事件）
    std::mutex mtx;                            // 线程安全锁
};

// CPU 运行时实现（强化Event，去掉虚拟Stream）
class CpuDeviceRuntime final : public IDeviceRuntime {
public:
    DI device() const override { return DI::cpu(); }

    int deviceCount() const override;
    bool isAvailable(int idx) const override;
    void setCurrentDevice(DI dev) override;
    void synchronizeDevice(DI dev) override;

    void* allocate(DI dev, size_t bytes, MemoryType kind) override;
    void deallocate(DI dev, void* ptr, MemoryType kind) override;

    // Stream 相关接口做空实现（CPU无流）
    void* allocateAsync(DI dev, size_t bytes, MemoryType kind, StreamHandle stream) override {
        (void)stream;
        return allocate(dev, bytes, kind);
    }
    void deallocateAsync(DI dev, void* ptr, MemoryType kind, StreamHandle stream) override {
        (void)stream;
        deallocate(dev, ptr, kind);
    }
    StreamHandle createStream(DI dev, int priority = 0) override {
        (void)dev; (void)priority;
        return StreamHandle(nullptr, DI::cpu()); // 返回空流句柄
    }
    void destroyStream(StreamHandle stream) override { (void)stream; }
    void synchronizeStream(StreamHandle stream) override { (void)stream; }

    // 核心：强化的Event接口（保证执行顺序）
    EventHandle createEvent(DI dev, bool timing = false) override;
    void destroyEvent(EventHandle event) override;
    void recordEvent(EventHandle event, StreamHandle stream) override;
    bool queryEvent(EventHandle event) override;
    void synchronizeEvent(EventHandle event) override;
    void waitEvent(StreamHandle stream, EventHandle event) override;

    // 拷贝接口：通过Event保证顺序（异步接口=同步+Event标记完成）
    void memcpyAsync(void* dst, DI dst_dev,
                     const void* src, DI src_dev,
                     size_t bytes, StreamHandle stream) override;

private:
    // 管理CPU Event的内部数据
    std::mutex mtx_;
    std::unordered_map<void*, std::unique_ptr<CpuEventData>> cpu_events_;

    // 等待事件完成（递归等待依赖）
    void waitEventInternal(CpuEventData* event_data);
};

// 注册CPU后端
void registerCpuBackend();

} // namespace EC::Dev