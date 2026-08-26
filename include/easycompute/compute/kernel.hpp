#pragma once

#include <cstdint>

#include "easycompute/core/device.hpp"

#if defined(__CUDACC__)
#define EC_HOST_DEVICE __host__ __device__
#else
#define EC_HOST_DEVICE
#endif

namespace ec::compute {

inline constexpr int kMaxTensorRank = 16;

struct OperandPlan {
  std::int64_t storage_offset{0};
  std::int64_t strides[kMaxTensorRank]{};
};

struct BinaryIterationPlan {
  int rank{0};
  std::int64_t numel{0};
  std::int64_t shape[kMaxTensorRank]{};
  OperandPlan output;
  OperandPlan lhs;
  OperandPlan rhs;
};

EC_HOST_DEVICE inline std::int64_t offset_for_linear(
    const BinaryIterationPlan& plan, const OperandPlan& operand, std::int64_t linear) {
  std::int64_t offset = operand.storage_offset;
  for (int dim = plan.rank - 1; dim >= 0; --dim) {
    const std::int64_t coordinate = linear % plan.shape[dim];
    linear /= plan.shape[dim];
    offset += coordinate * operand.strides[dim];
  }
  return offset;
}

struct BinaryKernelCall {
  void* output{nullptr};
  const void* lhs{nullptr};
  const void* rhs{nullptr};
  BinaryIterationPlan plan;
  Device device;
};

using BinaryKernel = void (*)(const BinaryKernelCall& call);
using BinaryKernelPredicate = bool (*)(const BinaryKernelCall& call);

struct BinaryKernelRegistration {
  BinaryKernel kernel{nullptr};
  BinaryKernelPredicate supports{nullptr};
  int priority{0};
};

}  // namespace ec::compute

#undef EC_HOST_DEVICE
