#include "easycompute/core/dtype.hpp"
#include "easycompute/core/device.hpp"
#include "easycompute/core/error.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ec {
namespace {

template <typename T, bool Multiply>
__global__ void binary_kernel(T* output, const T* lhs, const T* rhs, std::int64_t count) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = Multiply ? lhs[index] * rhs[index] : lhs[index] + rhs[index];
}

template <bool Multiply>
__global__ void binary_half_kernel(__half* output, const __half* lhs, const __half* rhs, std::int64_t count) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = Multiply ? __hmul(lhs[index], rhs[index]) : __hadd(lhs[index], rhs[index]);
}

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw DeviceError(std::string(operation) + ": " + cudaGetErrorString(status));
}

}  // namespace

void launch_binary_cuda(void* output, const void* lhs, const void* rhs,
                        std::int64_t count, DType dtype, int operation, Device device) {
  check(cudaSetDevice(device.index()), "cudaSetDevice");
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  if (dtype == DType::Float32) {
    if (operation == 1) binary_kernel<float, true><<<blocks, threads>>>(static_cast<float*>(output), static_cast<const float*>(lhs), static_cast<const float*>(rhs), count);
    else binary_kernel<float, false><<<blocks, threads>>>(static_cast<float*>(output), static_cast<const float*>(lhs), static_cast<const float*>(rhs), count);
  } else {
    if (operation == 1) binary_half_kernel<true><<<blocks, threads>>>(static_cast<__half*>(output), static_cast<const __half*>(lhs), static_cast<const __half*>(rhs), count);
    else binary_half_kernel<false><<<blocks, threads>>>(static_cast<__half*>(output), static_cast<const __half*>(lhs), static_cast<const __half*>(rhs), count);
  }
  check(cudaGetLastError(), "binary kernel launch");
  check(cudaDeviceSynchronize(), "binary kernel synchronize");
}

}  // namespace ec
