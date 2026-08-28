#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "easycompute/core/id.hpp"
#include "easycompute/core/status.hpp"

namespace ec {

struct CapabilityProvider {
  ExtensionKey capability;
  ExtensionKey provider;
  RuntimeId capability_id;
  RuntimeId provider_id;
  std::uint32_t abi_major{1};
  std::uint32_t abi_minor{0};
  int priority{0};
  const void* api{nullptr};
  std::shared_ptr<const void> owner;
};

struct CapabilitySnapshot {
  std::uint64_t epoch{0};
  std::unordered_map<RuntimeId, std::vector<CapabilityProvider>, RuntimeIdHash> providers;
};

class CapabilityRegistry;

class RegistrationTransaction {
public:
  RegistrationTransaction(RegistrationTransaction&&) noexcept = default;
  RegistrationTransaction& operator=(RegistrationTransaction&&) noexcept = default;
  RegistrationTransaction(const RegistrationTransaction&) = delete;
  RegistrationTransaction& operator=(const RegistrationTransaction&) = delete;

  Status add(CapabilityProvider provider);
  void defer(std::function<Status()> action);
  Status commit();
  bool committed() const { return committed_; }

private:
  friend class CapabilityRegistry;
  RegistrationTransaction(CapabilityRegistry& registry, std::uint64_t base_epoch,
                          std::shared_ptr<const void> owner);

  CapabilityRegistry* registry_{nullptr};
  std::uint64_t base_epoch_{0};
  std::shared_ptr<const void> owner_;
  std::vector<CapabilityProvider> pending_;
  std::vector<std::function<Status()>> deferred_;
  bool committed_{false};
};

class CapabilityRegistry {
public:
  explicit CapabilityRegistry(InternTable& ids);

  RegistrationTransaction begin(std::shared_ptr<const void> owner = {});
  std::shared_ptr<const CapabilitySnapshot> snapshot() const;
  std::vector<CapabilityProvider> query(const ExtensionKey& capability) const;
  std::uint64_t epoch() const;

private:
  friend class RegistrationTransaction;
  Status commit(RegistrationTransaction& transaction);

  mutable std::mutex commit_mutex_;
  InternTable* ids_;
  std::shared_ptr<const CapabilitySnapshot> snapshot_;
};

}  // namespace ec
