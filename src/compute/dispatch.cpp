#include "easycompute/tensor/tensor.hpp"

#include <algorithm>

#include "easycompute/compute/registry.hpp"
#include "easycompute/core/error.hpp"
#include "easycompute/tensor/access.hpp"

namespace ec {
namespace {

void validate_binary(const Tensor& lhs, const Tensor& rhs) {
  if (!lhs.defined() || !rhs.defined()) throw TensorError("binary operation received an undefined tensor");
  if (lhs.shape() != rhs.shape()) throw TensorError("binary operation shape mismatch");
  if (lhs.dtype() != rhs.dtype()) throw TensorError("binary operation dtype mismatch");
  if (lhs.device() != rhs.device()) throw TensorError("binary operation device mismatch");
}

compute::OperandPlan make_operand(const Tensor& tensor) {
  compute::OperandPlan operand;
  operand.storage_offset = tensor.storage_offset();
  const auto strides = tensor.stride().flatten();
  std::copy(strides.begin(), strides.end(), operand.strides);
  return operand;
}

compute::BinaryIterationPlan make_plan(const Tensor& output, const Tensor& lhs, const Tensor& rhs) {
  const auto shape = output.shape().flatten();
  if (shape.size() > static_cast<std::size_t>(compute::kMaxTensorRank)) {
    throw TensorError("tensor rank exceeds the current iteration-plan limit");
  }
  compute::BinaryIterationPlan plan;
  plan.rank = static_cast<int>(shape.size());
  plan.numel = output.numel();
  std::copy(shape.begin(), shape.end(), plan.shape);
  plan.output = make_operand(output);
  plan.lhs = make_operand(lhs);
  plan.rhs = make_operand(rhs);
  return plan;
}

template <typename OpTag>
Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) {
  validate_binary(lhs, rhs);
  Tensor output = Tensor::zeros(lhs.shape(), lhs.dtype(), lhs.device());
  const auto& output_impl = detail::TensorAccess::get(output);
  const auto& lhs_impl = detail::TensorAccess::get(lhs);
  const auto& rhs_impl = detail::TensorAccess::get(rhs);

  compute::BinaryKernelCall call;
  call.output = output_impl.storage->data();
  call.lhs = lhs_impl.storage->data();
  call.rhs = rhs_impl.storage->data();
  call.plan = make_plan(output, lhs, rhs);
  call.device = lhs.device();

  compute::ensure_builtin_kernels_registered();
  const auto kernel = compute::global_kernel_registry().find_binary(
      compute::KernelKey{lhs.device().type(), lhs.dtype(), OpTag::id}, call);
  kernel(call);
  return output;
}

}  // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary<compute::AddTag>(lhs, rhs);
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary<compute::MultiplyTag>(lhs, rhs);
}

}  // namespace ec
