#include "easycompute/memory/allocator.hpp"

#include <cuda_runtime.h>

#include <string>

#include "easycompute/core/error.hpp"

namespace ec {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw DeviceError(std::string(operation) + ": " + cudaGetErrorString(status));
}

class CUDAAllocator final : public Allocator {
public:
  void* allocate(std::size_t nbytes, Device device) override {
    if (device.type() != DeviceType::CUDA) throw DeviceError("CUDAAllocator received " + device.str());
    if (nbytes == 0) return nullptr;
    check_cuda(cudaSetDevice(device.index()), "cudaSetDevice");
    void* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, nbytes), "cudaMalloc");
    return pointer;
  }
  void deallocate(void* pointer, std::size_t, Device device) noexcept override {
    if (pointer == nullptr) return;
    cudaSetDevice(device.index());
    cudaFree(pointer);
  }
  const char* name() const noexcept override { return "cuda"; }
};

CUDAAllocator& cuda_allocator() { static CUDAAllocator allocator; return allocator; }

bool cuda_runtime_available(Device device) {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) { cudaGetLastError(); return false; }
  return device.index() >= 0 && device.index() < count;
}

void cuda_synchronize(Device device) {
  check_cuda(cudaSetDevice(device.index()), "cudaSetDevice");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void host_to_cuda(void* destination, Device destination_device,
                  const void* source, Device, std::size_t nbytes) {
  check_cuda(cudaSetDevice(destination_device.index()), "cudaSetDevice");
  check_cuda(cudaMemcpy(destination, source, nbytes, cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice");
}

void cuda_to_host(void* destination, Device,
                  const void* source, Device source_device, std::size_t nbytes) {
  check_cuda(cudaSetDevice(source_device.index()), "cudaSetDevice");
  check_cuda(cudaMemcpy(destination, source, nbytes, cudaMemcpyDeviceToHost), "cudaMemcpyDeviceToHost");
}

void cuda_to_cuda(void* destination, Device destination_device,
                  const void* source, Device, std::size_t nbytes) {
  check_cuda(cudaSetDevice(destination_device.index()), "cudaSetDevice");
  check_cuda(cudaMemcpy(destination, source, nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpyDeviceToDevice");
}

}  // namespace

void register_cuda_memory_backend(DeviceRegistry& registry) {
  registry.register_runtime(DeviceType::CUDA,
      DeviceRuntime{&cuda_allocator(), &cuda_runtime_available, &cuda_synchronize, false, "cuda"});
  registry.register_copy(DeviceType::CPU, DeviceType::CUDA, &host_to_cuda);
  registry.register_copy(DeviceType::CUDA, DeviceType::CPU, &cuda_to_host);
  registry.register_copy(DeviceType::CUDA, DeviceType::CUDA, &cuda_to_cuda);
}

}  // namespace ec
