#include "easycompute/compute/registry.hpp"

#include <type_traits>

namespace ec::compute {
namespace {

struct CPUAdd {
  static float apply(float lhs, float rhs) { return lhs + rhs; }
};

struct CPUMultiply {
  static float apply(float lhs, float rhs) { return lhs * rhs; }
};

template <typename Scalar>
float load_scalar(const Scalar* data, std::int64_t offset) {
  if constexpr (std::is_same_v<Scalar, Float16>) return static_cast<float>(data[offset]);
  else return data[offset];
}

template <typename Scalar>
void store_scalar(Scalar* data, std::int64_t offset, float value) {
  if constexpr (std::is_same_v<Scalar, Float16>) data[offset] = Float16(value);
  else data[offset] = value;
}

template <typename Scalar, typename Operation>
void binary_loop(const BinaryKernelCall& call) {
  auto* output = static_cast<Scalar*>(call.output);
  const auto* lhs = static_cast<const Scalar*>(call.lhs);
  const auto* rhs = static_cast<const Scalar*>(call.rhs);
  for (std::int64_t linear = 0; linear < call.plan.numel; ++linear) {
    const auto output_offset = offset_for_linear(call.plan, call.plan.output, linear);
    const auto lhs_offset = offset_for_linear(call.plan, call.plan.lhs, linear);
    const auto rhs_offset = offset_for_linear(call.plan, call.plan.rhs, linear);
    store_scalar(output, output_offset,
                 Operation::apply(load_scalar(lhs, lhs_offset), load_scalar(rhs, rhs_offset)));
  }
}

template <typename Scalar>
void add_kernel(const BinaryKernelCall& call) { binary_loop<Scalar, CPUAdd>(call); }

template <typename Scalar>
void multiply_kernel(const BinaryKernelCall& call) { binary_loop<Scalar, CPUMultiply>(call); }

}  // namespace

void register_cpu_kernels(KernelRegistry& registry) {
  registry.register_binary<CPUBackendTag, AddTag, float>(&add_kernel<float>);
  registry.register_binary<CPUBackendTag, AddTag, Float16>(&add_kernel<Float16>);
  registry.register_binary<CPUBackendTag, MultiplyTag, float>(&multiply_kernel<float>);
  registry.register_binary<CPUBackendTag, MultiplyTag, Float16>(&multiply_kernel<Float16>);
}

}  // namespace ec::compute

