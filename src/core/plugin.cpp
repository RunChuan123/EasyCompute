#include "easycompute/core/plugin.hpp"

#include <algorithm>
#include <exception>
#include <mutex>
#include <unordered_map>

#include "easycompute/core/runtime.hpp"

namespace ec {
namespace {

constexpr Version kCoreAbiVersion{1, 0, 0};

bool version_in_range(Version value, const VersionRange& range) {
  const auto encode = [](Version version) {
    return (static_cast<std::uint64_t>(version.major) << 42U) |
           (static_cast<std::uint64_t>(version.minor) << 21U) | version.patch;
  };
  return encode(value) >= encode(range.minimum) && encode(value) < encode(range.maximum_exclusive);
}

}  // namespace

struct PluginManager::Impl {
  explicit Impl(Runtime& owner) : runtime(&owner) {}
  Runtime* runtime;
  mutable std::mutex mutex;
  std::unordered_map<ExtensionKey, std::shared_ptr<PluginState>, ExtensionKeyHash> loaded;
};

PluginManager::PluginManager(Runtime& runtime) : impl_(std::make_unique<Impl>(runtime)) {}
PluginManager::~PluginManager() = default;

Status PluginManager::load_static(const StaticPluginDescriptor& descriptor) {
  if (descriptor.key.empty()) return {StatusCode::InvalidArgument, "plugin key must not be empty"};
  if (descriptor.register_capabilities == nullptr)
    return {StatusCode::InvalidArgument, "plugin register callback must not be null"};
  if (!version_in_range(kCoreAbiVersion, descriptor.required_core_abi))
    return {StatusCode::VersionMismatch, "plugin " + descriptor.key.str() + " does not support core ABI 1.0"};

  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->loaded.contains(descriptor.key)) return Status::success();
    for (const auto& dependency : descriptor.dependencies) {
      const auto iterator = impl_->loaded.find(dependency.key);
      if (iterator == impl_->loaded.end())
        return {StatusCode::FailedPrecondition,
                "plugin " + descriptor.key.str() + " requires " + dependency.key.str()};
      if (!version_in_range(iterator->second->version, dependency.required_version))
        return {StatusCode::VersionMismatch,
                "plugin " + descriptor.key.str() + " requires an incompatible version of " +
                    dependency.key.str()};
    }
  }

  auto state = std::make_shared<PluginState>(PluginState{descriptor.key, descriptor.version});
  auto transaction = impl_->runtime->capabilities().begin(state);
  Status registered;
  try {
    registered = descriptor.register_capabilities(*impl_->runtime, transaction);
  } catch (const std::exception& error) {
    return {StatusCode::Internal, std::string("plugin registration threw an exception: ") + error.what()};
  } catch (...) {
    return {StatusCode::Internal, "plugin registration threw an unknown exception"};
  }
  if (!registered.ok()) return registered;
  const auto committed = transaction.commit();
  if (!committed.ok()) return committed;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto [unused, inserted] = impl_->loaded.emplace(descriptor.key, std::move(state));
  (void)unused;
  if (!inserted) return {StatusCode::AlreadyExists, "plugin loaded concurrently: " + descriptor.key.str()};
  return Status::success();
}

bool PluginManager::is_loaded(const ExtensionKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->loaded.contains(key);
}

std::vector<ExtensionKey> PluginManager::loaded_plugins() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::vector<ExtensionKey> result;
  result.reserve(impl_->loaded.size());
  for (const auto& [key, unused] : impl_->loaded) {
    (void)unused;
    result.push_back(key);
  }
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace ec
