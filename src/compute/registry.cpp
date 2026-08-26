#include "easycompute/compute/registry.hpp"

#include <algorithm>
#include <sstream>

#include "easycompute/core/error.hpp"

namespace ec::compute {

std::string_view op_name(OpId op) {
  switch (op) {
    case OpId::Add: return AddTag::name;
    case OpId::Multiply: return MultiplyTag::name;
  }
  return "unknown";
}

std::size_t KernelKeyHash::operator()(const KernelKey& key) const noexcept {
  const auto device = static_cast<std::size_t>(key.device.id());
  const auto dtype = static_cast<std::size_t>(key.dtype);
  const auto op = static_cast<std::size_t>(key.op);
  return (device << 24U) ^ (dtype << 16U) ^ op;
}

void KernelRegistry::register_binary(KernelKey key, BinaryKernelRegistration registration) {
  if (registration.kernel == nullptr) throw TensorError("cannot register a null binary kernel");
  std::lock_guard<std::mutex> lock(mutex_);
  auto& candidates = binary_kernels_[key];
  const auto duplicate = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
    return candidate.kernel == registration.kernel;
  });
  if (duplicate != candidates.end()) return;
  candidates.push_back(registration);
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.priority > rhs.priority;
  });
}

BinaryKernel KernelRegistry::find_binary(KernelKey key, const BinaryKernelCall& call) const {
  std::vector<BinaryKernelRegistration> candidates;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = binary_kernels_.find(key);
    if (iterator != binary_kernels_.end()) candidates = iterator->second;
  }
  for (const auto& candidate : candidates) {
    if (candidate.supports == nullptr || candidate.supports(call)) return candidate.kernel;
  }
  std::ostringstream message;
  message << "no compatible kernel registered for op=" << op_name(key.op)
          << ", dtype=" << dtype_name(key.dtype)
          << ", device=" << key.device.name();
  throw TensorError(message.str());
}

KernelRegistry& global_kernel_registry() {
  static KernelRegistry registry;
  return registry;
}

void ensure_builtin_kernels_registered() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto& registry = global_kernel_registry();
    register_cpu_kernels(registry);
#if EC_HAS_CUDA
    register_cuda_kernels(registry);
#endif
  });
}

}  // namespace ec::compute
