#include "easycompute/compute/registry.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>

#include "easycompute/core/error.hpp"

namespace ec::compute {
namespace {

struct CUDAAddFloat {
  __device__ float operator()(float lhs, float rhs) const { return lhs + rhs; }
};

struct CUDAMultiplyFloat {
  __device__ float operator()(float lhs, float rhs) const { return lhs * rhs; }
};

struct CUDAAddHalf {
  __device__ __half operator()(__half lhs, __half rhs) const { return __hadd(lhs, rhs); }
};

struct CUDAMultiplyHalf {
  __device__ __half operator()(__half lhs, __half rhs) const { return __hmul(lhs, rhs); }
};

template <typename Scalar, typename Operation>
__global__ void strided_binary_kernel(Scalar* output, const Scalar* lhs, const Scalar* rhs,
                                      BinaryIterationPlan plan, Operation operation) {
  const auto linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= plan.numel) return;
  const auto output_offset = offset_for_linear(plan, plan.output, linear);
  const auto lhs_offset = offset_for_linear(plan, plan.lhs, linear);
  const auto rhs_offset = offset_for_linear(plan, plan.rhs, linear);
  output[output_offset] = operation(lhs[lhs_offset], rhs[rhs_offset]);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw DeviceError(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename Scalar, typename Operation>
void launch(const BinaryKernelCall& call, Operation operation) {
  check_cuda(cudaSetDevice(call.device.index()), "cudaSetDevice");
  constexpr int threads = 256;
  const int blocks = static_cast<int>((call.plan.numel + threads - 1) / threads);
  strided_binary_kernel<<<blocks, threads>>>(
      static_cast<Scalar*>(call.output), static_cast<const Scalar*>(call.lhs),
      static_cast<const Scalar*>(call.rhs), call.plan, operation);
  check_cuda(cudaGetLastError(), "strided binary kernel launch");
  check_cuda(cudaDeviceSynchronize(), "strided binary kernel synchronize");
}

void add_float32(const BinaryKernelCall& call) { launch<float>(call, CUDAAddFloat{}); }
void multiply_float32(const BinaryKernelCall& call) { launch<float>(call, CUDAMultiplyFloat{}); }
void add_float16(const BinaryKernelCall& call) { launch<__half>(call, CUDAAddHalf{}); }
void multiply_float16(const BinaryKernelCall& call) { launch<__half>(call, CUDAMultiplyHalf{}); }

}  // namespace

void register_cuda_kernels(KernelRegistry& registry) {
  registry.register_binary<CUDABackendTag, AddTag, float>(&add_float32);
  registry.register_binary<CUDABackendTag, AddTag, Float16>(&add_float16);
  registry.register_binary<CUDABackendTag, MultiplyTag, float>(&multiply_float32);
  registry.register_binary<CUDABackendTag, MultiplyTag, Float16>(&multiply_float16);
}

}  // namespace ec::compute
