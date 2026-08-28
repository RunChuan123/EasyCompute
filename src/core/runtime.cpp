#include "easycompute/core/runtime.hpp"

#include "easycompute/compute/registry.hpp"
#include "easycompute/memory/allocator.hpp"

namespace ec {

struct Runtime::Impl {
  explicit Impl(Runtime& owner) : capabilities(ids), plugins(owner) {}
  InternTable ids;
  CapabilityRegistry capabilities;
  PluginManager plugins;
  DeviceRegistry devices;
  compute::KernelRegistry kernels;
};

Runtime::Runtime() : impl_(std::make_unique<Impl>(*this)) {}
Runtime::~Runtime() = default;

InternTable& Runtime::ids() { return impl_->ids; }
const InternTable& Runtime::ids() const { return impl_->ids; }
CapabilityRegistry& Runtime::capabilities() { return impl_->capabilities; }
const CapabilityRegistry& Runtime::capabilities() const { return impl_->capabilities; }
PluginManager& Runtime::plugins() { return impl_->plugins; }
const PluginManager& Runtime::plugins() const { return impl_->plugins; }
DeviceRegistry& Runtime::devices() { return impl_->devices; }
compute::KernelRegistry& Runtime::kernels() { return impl_->kernels; }

Runtime& default_runtime() {
  static Runtime runtime;
  return runtime;
}

}  // namespace ec
