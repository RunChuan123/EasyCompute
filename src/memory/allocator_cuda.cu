#include "easycompute/memory/allocator.hpp"

#include <cuda_runtime.h>

#include <string>

#include "easycompute/core/error.hpp"

namespace ec {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw DeviceError(std::string(operation) + ": " + cudaGetErrorString(status));
}

cudaMemcpyKind copy_kind(Device destination, Device source) {
  if (source.is_cpu() && destination.is_cuda()) return cudaMemcpyHostToDevice;
  if (source.is_cuda() && destination.is_cpu()) return cudaMemcpyDeviceToHost;
  if (source.is_cuda() && destination.is_cuda()) return cudaMemcpyDeviceToDevice;
  return cudaMemcpyHostToHost;
}

}  // namespace

void* cuda_allocate(std::size_t nbytes, Device device) {
  if (!device.is_cuda()) throw DeviceError("CUDA allocator received " + device.str());
  if (nbytes == 0) return nullptr;
  check_cuda(cudaSetDevice(device.index()), "cudaSetDevice");
  void* pointer = nullptr;
  check_cuda(cudaMalloc(&pointer, nbytes), "cudaMalloc");
  return pointer;
}

void cuda_deallocate(void* pointer, Device device) noexcept {
  if (pointer == nullptr) return;
  cudaSetDevice(device.index());
  cudaFree(pointer);
}

void cuda_copy_bytes(void* destination, Device destination_device,
                     const void* source, Device source_device, std::size_t nbytes) {
  const Device active = destination_device.is_cuda() ? destination_device : source_device;
  check_cuda(cudaSetDevice(active.index()), "cudaSetDevice");
  check_cuda(cudaMemcpy(destination, source, nbytes, copy_kind(destination_device, source_device)), "cudaMemcpy");
}

bool cuda_runtime_available() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) { cudaGetLastError(); return false; }
  return count > 0;
}

void cuda_synchronize(Device device) {
  check_cuda(cudaSetDevice(device.index()), "cudaSetDevice");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}  // namespace ec
