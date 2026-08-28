#pragma once

#include <cstddef>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "easycompute/core/device.hpp"

namespace ec {

class Runtime;

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void* allocate(std::size_t nbytes, Device device) = 0;
  virtual void deallocate(void* pointer, std::size_t nbytes, Device device) noexcept = 0;
  virtual const char* name() const noexcept = 0;
};

using DeviceAvailable = bool (*)(Device device);
using DeviceSynchronize = void (*)(Device device);
using DeviceCopy = void (*)(void* destination, Device destination_device,
                            const void* source, Device source_device, std::size_t nbytes);

struct DeviceRuntime {
  Allocator* allocator{nullptr};
  DeviceAvailable available{nullptr};
  DeviceSynchronize synchronize{nullptr};
  bool host_accessible{false};
  std::string_view name;
};

struct CopyRoute {
  DeviceType source;
  DeviceType destination;
  friend bool operator==(const CopyRoute& lhs, const CopyRoute& rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

struct CopyRouteHash { std::size_t operator()(const CopyRoute& route) const noexcept; };

class DeviceRegistry {
public:
  void register_runtime(DeviceType type, DeviceRuntime runtime);
  void register_copy(DeviceType source, DeviceType destination, DeviceCopy copy);
  DeviceRuntime find_runtime(DeviceType type) const;
  DeviceCopy find_copy(DeviceType source, DeviceType destination) const;

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::uint16_t, DeviceRuntime> runtimes_;
  std::unordered_map<CopyRoute, DeviceCopy, CopyRouteHash> copies_;
};

DeviceRegistry& global_device_registry();
DeviceRegistry& device_registry(Runtime& runtime);
void ensure_builtin_device_backends_registered();
void register_cpu_memory_backend(DeviceRegistry& registry);
#if EC_HAS_CUDA
void register_cuda_memory_backend(DeviceRegistry& registry);
#endif

Allocator& allocator_for(Device device);
bool device_available(Device device);
bool is_host_accessible(Device device);
bool cuda_available();
void copy_bytes(void* destination, Device destination_device,
                const void* source, Device source_device, std::size_t nbytes);
void device_synchronize(Device device);

}  // namespace ec
