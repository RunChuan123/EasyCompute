#include "easycompute/memory/allocator.hpp"

#include <cstring>
#include <new>
#include <sstream>

#include "easycompute/core/error.hpp"

namespace ec {
namespace {

class CPUAllocator final : public Allocator {
public:
  void* allocate(std::size_t nbytes, Device device) override {
    if (device.type() != device_types::cpu) throw DeviceError("CPUAllocator received " + device.str());
    if (nbytes == 0) return nullptr;
    return ::operator new(nbytes, std::align_val_t{64});
  }
  void deallocate(void* pointer, std::size_t, Device) noexcept override {
    if (pointer != nullptr) ::operator delete(pointer, std::align_val_t{64});
  }
  const char* name() const noexcept override { return "cpu"; }
};

CPUAllocator& cpu_allocator() { static CPUAllocator allocator; return allocator; }
bool cpu_available(Device) { return true; }
void cpu_synchronize(Device) {}
void cpu_copy(void* destination, Device, const void* source, Device, std::size_t nbytes) {
  std::memcpy(destination, source, nbytes);
}

std::string missing_backend(DeviceType type) {
  return "no device runtime registered for " + std::string(type.name()) +
         " (id=" + std::to_string(type.id()) + ")";
}

}  // namespace

std::size_t CopyRouteHash::operator()(const CopyRoute& route) const noexcept {
  return (static_cast<std::size_t>(route.source.id()) << 16U) ^ route.destination.id();
}

void DeviceRegistry::register_runtime(DeviceType type, DeviceRuntime runtime) {
  if (runtime.allocator == nullptr || runtime.available == nullptr || runtime.synchronize == nullptr)
    throw DeviceError("device runtime registration is incomplete");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto [iterator, inserted] = runtimes_.emplace(type.id(), runtime);
  if (!inserted && iterator->second.allocator != runtime.allocator)
    throw DeviceError("a different runtime is already registered for this device type");
}

void DeviceRegistry::register_copy(DeviceType source, DeviceType destination, DeviceCopy copy) {
  if (copy == nullptr) throw DeviceError("cannot register a null device copy route");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto [iterator, inserted] = copies_.emplace(CopyRoute{source, destination}, copy);
  if (!inserted && iterator->second != copy)
    throw DeviceError("a different device copy route is already registered");
}

DeviceRuntime DeviceRegistry::find_runtime(DeviceType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iterator = runtimes_.find(type.id());
  if (iterator == runtimes_.end()) throw DeviceError(missing_backend(type));
  return iterator->second;
}

DeviceCopy DeviceRegistry::find_copy(DeviceType source, DeviceType destination) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iterator = copies_.find(CopyRoute{source, destination});
  if (iterator == copies_.end()) {
    std::ostringstream message;
    message << "no copy route registered from " << source.name() << " to " << destination.name();
    throw DeviceError(message.str());
  }
  return iterator->second;
}

DeviceRegistry& global_device_registry() { static DeviceRegistry registry; return registry; }

void register_cpu_memory_backend(DeviceRegistry& registry) {
  registry.register_runtime(device_types::cpu,
      DeviceRuntime{&cpu_allocator(), &cpu_available, &cpu_synchronize, true, "cpu"});
  registry.register_copy(device_types::cpu, device_types::cpu, &cpu_copy);
}

void ensure_builtin_device_backends_registered() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto& registry = global_device_registry();
    register_cpu_memory_backend(registry);
#if EC_HAS_CUDA
    register_cuda_memory_backend(registry);
#endif
  });
}

Allocator& allocator_for(Device device) {
  ensure_builtin_device_backends_registered();
  return *global_device_registry().find_runtime(device.type()).allocator;
}

bool device_available(Device device) {
  ensure_builtin_device_backends_registered();
  const auto runtime = global_device_registry().find_runtime(device.type());
  return runtime.available(device);
}

bool is_host_accessible(Device device) {
  ensure_builtin_device_backends_registered();
  return global_device_registry().find_runtime(device.type()).host_accessible;
}

bool cuda_available() {
#if EC_HAS_CUDA
  return device_available(Device::cuda());
#else
  return false;
#endif
}

void copy_bytes(void* destination, Device destination_device,
                const void* source, Device source_device, std::size_t nbytes) {
  if (nbytes == 0) return;
  if (destination == nullptr || source == nullptr) throw DeviceError("copy_bytes received a null pointer");
  ensure_builtin_device_backends_registered();
  const auto copy = global_device_registry().find_copy(source_device.type(), destination_device.type());
  copy(destination, destination_device, source, source_device, nbytes);
}

void device_synchronize(Device device) {
  ensure_builtin_device_backends_registered();
  const auto runtime = global_device_registry().find_runtime(device.type());
  runtime.synchronize(device);
}

}  // namespace ec
