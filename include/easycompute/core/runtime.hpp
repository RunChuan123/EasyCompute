#pragma once

#include <memory>

#include "easycompute/core/capability.hpp"
#include "easycompute/core/id.hpp"
#include "easycompute/core/plugin.hpp"

namespace ec {

class DeviceRegistry;
namespace compute { class KernelRegistry; }

class Runtime {
public:
  Runtime();
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  InternTable& ids();
  const InternTable& ids() const;
  CapabilityRegistry& capabilities();
  const CapabilityRegistry& capabilities() const;
  PluginManager& plugins();
  const PluginManager& plugins() const;
  DeviceRegistry& devices();
  compute::KernelRegistry& kernels();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

Runtime& default_runtime();
void ensure_builtin_plugins_registered(Runtime& runtime);
inline void ensure_builtin_plugins_registered() { ensure_builtin_plugins_registered(default_runtime()); }

}  // namespace ec
