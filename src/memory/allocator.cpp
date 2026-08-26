#include "easycompute/memory/allocator.hpp"

#include <cstring>
#include <new>

#include "easycompute/core/error.hpp"

namespace ec {

#if EC_HAS_CUDA
void* cuda_allocate(std::size_t nbytes, Device device);
void cuda_deallocate(void* pointer, Device device) noexcept;
void cuda_copy_bytes(void* destination, Device destination_device,
                     const void* source, Device source_device, std::size_t nbytes);
bool cuda_runtime_available();
void cuda_synchronize(Device device);
#endif

namespace {

class CPUAllocator final : public Allocator {
public:
  void* allocate(std::size_t nbytes, Device device) override {
    if (!device.is_cpu()) throw DeviceError("CPUAllocator received " + device.str());
    if (nbytes == 0) return nullptr;
    return ::operator new(nbytes, std::align_val_t{64});
  }

  void deallocate(void* pointer, std::size_t, Device) noexcept override {
    if (pointer != nullptr) ::operator delete(pointer, std::align_val_t{64});
  }

  const char* name() const noexcept override { return "cpu"; }
};

CPUAllocator& cpu_allocator() { static CPUAllocator allocator; return allocator; }

#if EC_HAS_CUDA
class CUDAAllocator final : public Allocator {
public:
  void* allocate(std::size_t nbytes, Device device) override { return cuda_allocate(nbytes, device); }
  void deallocate(void* pointer, std::size_t, Device device) noexcept override { cuda_deallocate(pointer, device); }
  const char* name() const noexcept override { return "cuda"; }
};

CUDAAllocator& cuda_allocator() { static CUDAAllocator allocator; return allocator; }
#endif

}  // namespace

Allocator& allocator_for(Device device) {
  if (device.is_cpu()) return cpu_allocator();
#if EC_HAS_CUDA
  if (device.is_cuda()) return cuda_allocator();
#endif
  throw DeviceError("backend is unavailable for " + device.str());
}

bool cuda_available() {
#if EC_HAS_CUDA
  return cuda_runtime_available();
#else
  return false;
#endif
}

void copy_bytes(void* destination, Device destination_device,
                const void* source, Device source_device, std::size_t nbytes) {
  if (nbytes == 0) return;
  if (destination == nullptr || source == nullptr) throw DeviceError("copy_bytes received a null pointer");
  if (destination_device.is_cpu() && source_device.is_cpu()) {
    std::memcpy(destination, source, nbytes);
    return;
  }
#if EC_HAS_CUDA
  cuda_copy_bytes(destination, destination_device, source, source_device, nbytes);
#else
  throw DeviceError("CUDA copy requested from a CPU-only build");
#endif
}

void device_synchronize(Device device) {
  if (device.is_cpu()) return;
#if EC_HAS_CUDA
  cuda_synchronize(device);
#else
  throw DeviceError("CUDA synchronize requested from a CPU-only build");
#endif
}

}  // namespace ec
