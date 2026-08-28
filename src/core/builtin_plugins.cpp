#include "easycompute/core/runtime.hpp"

#include <stdexcept>

#include "easycompute/compute/registry.hpp"
#include "easycompute/memory/allocator.hpp"

namespace ec {
namespace {

const ExtensionKey& memory_backend_capability() {
  static const ExtensionKey key{"easycompute.runtime", "memory_backend", 1};
  return key;
}

const ExtensionKey& kernel_backend_capability() {
  static const ExtensionKey key{"easycompute.runtime", "kernel_backend", 1};
  return key;
}

const ExtensionKey& cpu_plugin_key() {
  static const ExtensionKey key{"easycompute.backend", "cpu", 1};
  return key;
}

int cpu_memory_api_token;
int cpu_kernel_api_token;
#if EC_HAS_CUDA
const ExtensionKey& cuda_plugin_key() {
  static const ExtensionKey key{"easycompute.backend", "cuda", 1};
  return key;
}

int cuda_memory_api_token;
int cuda_kernel_api_token;
#endif

Status add_provider(RegistrationTransaction& transaction, const ExtensionKey& capability,
                    const ExtensionKey& provider, const void* api) {
  return transaction.add(CapabilityProvider{capability, provider, {}, {}, 1, 0, 0, api, {}});
}

Status register_cpu_plugin(Runtime& runtime, RegistrationTransaction& transaction) {
  auto status = add_provider(transaction, memory_backend_capability(), cpu_plugin_key(), &cpu_memory_api_token);
  if (!status.ok()) return status;
  status = add_provider(transaction, kernel_backend_capability(), cpu_plugin_key(), &cpu_kernel_api_token);
  if (!status.ok()) return status;
  transaction.defer([&runtime] {
    register_cpu_memory_backend(runtime.devices());
    compute::register_cpu_kernels(runtime.kernels());
    return Status::success();
  });
  return Status::success();
}

#if EC_HAS_CUDA
Status register_cuda_plugin(Runtime& runtime, RegistrationTransaction& transaction) {
  auto status = add_provider(transaction, memory_backend_capability(), cuda_plugin_key(), &cuda_memory_api_token);
  if (!status.ok()) return status;
  status = add_provider(transaction, kernel_backend_capability(), cuda_plugin_key(), &cuda_kernel_api_token);
  if (!status.ok()) return status;
  transaction.defer([&runtime] {
    register_cuda_memory_backend(runtime.devices());
    compute::register_cuda_kernels(runtime.kernels());
    return Status::success();
  });
  return Status::success();
}
#endif

const StaticPluginDescriptor& cpu_plugin() {
  static const StaticPluginDescriptor descriptor{
      cpu_plugin_key(), Version{1, 0, 0}, VersionRange{{1, 0, 0}, {2, 0, 0}}, {}, &register_cpu_plugin};
  return descriptor;
}

#if EC_HAS_CUDA
const StaticPluginDescriptor& cuda_plugin() {
  static const StaticPluginDescriptor descriptor{
      cuda_plugin_key(), Version{1, 0, 0}, VersionRange{{1, 0, 0}, {2, 0, 0}},
      {{cpu_plugin_key(), VersionRange{{1, 0, 0}, {2, 0, 0}}}}, &register_cuda_plugin};
  return descriptor;
}
#endif

void require_loaded(const Status& status) {
  if (!status.ok()) throw std::runtime_error("builtin plugin registration failed: " + status.message());
}

}  // namespace

void ensure_builtin_plugins_registered(Runtime& runtime) {
  require_loaded(runtime.plugins().load_static(cpu_plugin()));
#if EC_HAS_CUDA
  require_loaded(runtime.plugins().load_static(cuda_plugin()));
#endif
}

}  // namespace ec
