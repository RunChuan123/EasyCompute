#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "easycompute/compute/kernel.hpp"
#include "easycompute/compute/op.hpp"
#include "easycompute/core/dtype.hpp"

namespace ec::compute {

struct KernelKey {
  DeviceType device;
  DType dtype;
  OpId op;
  friend bool operator==(const KernelKey& lhs, const KernelKey& rhs) {
    return lhs.device == rhs.device && lhs.dtype == rhs.dtype && lhs.op == rhs.op;
  }
};

struct KernelKeyHash {
  std::size_t operator()(const KernelKey& key) const noexcept;
};

class KernelRegistry {
public:
  void register_binary(KernelKey key, BinaryKernelRegistration registration);
  BinaryKernel find_binary(KernelKey key, const BinaryKernelCall& call) const;

  template <typename BackendTag, typename OpTag, typename Scalar>
  void register_binary(BinaryKernel kernel, int priority = 0,
                       BinaryKernelPredicate supports = nullptr) {
    register_binary(KernelKey{BackendTag::device_type, dtype_of_v<Scalar>, OpTag::id},
                    BinaryKernelRegistration{kernel, supports, priority});
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<KernelKey, std::vector<BinaryKernelRegistration>, KernelKeyHash> binary_kernels_;
};

KernelRegistry& global_kernel_registry();
void ensure_builtin_kernels_registered();
void register_cpu_kernels(KernelRegistry& registry);
#if EC_HAS_CUDA
void register_cuda_kernels(KernelRegistry& registry);
#endif

}  // namespace ec::compute
