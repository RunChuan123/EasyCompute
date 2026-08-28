#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "easycompute/core/capability.hpp"

namespace ec {

class Runtime;

struct Version {
  std::uint32_t major{0};
  std::uint32_t minor{0};
  std::uint32_t patch{0};
  friend bool operator==(const Version&, const Version&) = default;
};

struct VersionRange {
  Version minimum;
  Version maximum_exclusive;
};

using StaticPluginRegister = Status (*)(Runtime&, RegistrationTransaction&);

struct PluginDependency {
  ExtensionKey key;
  VersionRange required_version;
};

struct StaticPluginDescriptor {
  ExtensionKey key;
  Version version;
  VersionRange required_core_abi;
  std::vector<PluginDependency> dependencies;
  StaticPluginRegister register_capabilities{nullptr};
};

struct PluginState {
  ExtensionKey key;
  Version version;
};

using PluginLease = std::shared_ptr<const PluginState>;

class PluginManager {
public:
  explicit PluginManager(Runtime& runtime);
  ~PluginManager();

  PluginManager(const PluginManager&) = delete;
  PluginManager& operator=(const PluginManager&) = delete;

  Status load_static(const StaticPluginDescriptor& descriptor);
  bool is_loaded(const ExtensionKey& key) const;
  std::vector<ExtensionKey> loaded_plugins() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ec
