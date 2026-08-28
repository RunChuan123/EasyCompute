#include "easycompute/core/capability.hpp"

#include <algorithm>
#include <atomic>
#include <exception>

namespace ec {

RegistrationTransaction::RegistrationTransaction(CapabilityRegistry& registry,
                                                 std::uint64_t base_epoch,
                                                 std::shared_ptr<const void> owner)
    : registry_(&registry), base_epoch_(base_epoch), owner_(std::move(owner)) {}

Status RegistrationTransaction::add(CapabilityProvider provider) {
  if (committed_) return {StatusCode::FailedPrecondition, "registration transaction is already committed"};
  if (provider.capability.empty() || provider.provider.empty())
    return {StatusCode::InvalidArgument, "capability and provider keys must not be empty"};
  if (provider.api == nullptr) return {StatusCode::InvalidArgument, "capability provider API must not be null"};
  if (!provider.owner) provider.owner = owner_;
  pending_.push_back(std::move(provider));
  return Status::success();
}

void RegistrationTransaction::defer(std::function<Status()> action) {
  if (committed_) throw std::logic_error("registration transaction is already committed");
  deferred_.push_back(std::move(action));
}

Status RegistrationTransaction::commit() {
  if (committed_) return {StatusCode::FailedPrecondition, "registration transaction is already committed"};
  if (registry_ == nullptr) return {StatusCode::FailedPrecondition, "registration transaction has no registry"};
  return registry_->commit(*this);
}

CapabilityRegistry::CapabilityRegistry(InternTable& ids) : ids_(&ids) {
  std::atomic_store_explicit(&snapshot_, std::make_shared<const CapabilitySnapshot>(),
                             std::memory_order_release);
}

RegistrationTransaction CapabilityRegistry::begin(std::shared_ptr<const void> owner) {
  return RegistrationTransaction(*this, epoch(), std::move(owner));
}

std::shared_ptr<const CapabilitySnapshot> CapabilityRegistry::snapshot() const {
  return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

std::vector<CapabilityProvider> CapabilityRegistry::query(const ExtensionKey& capability) const {
  const auto id = ids_->find(capability);
  if (!id) return {};
  const auto current = snapshot();
  const auto iterator = current->providers.find(*id);
  if (iterator == current->providers.end()) return {};
  return iterator->second;
}

std::uint64_t CapabilityRegistry::epoch() const { return snapshot()->epoch; }

Status CapabilityRegistry::commit(RegistrationTransaction& transaction) {
  std::lock_guard<std::mutex> lock(commit_mutex_);
  const auto current = snapshot();
  if (current->epoch != transaction.base_epoch_)
    return {StatusCode::FailedPrecondition, "capability registry changed during registration"};

  auto next = std::make_shared<CapabilitySnapshot>(*current);
  for (auto provider : transaction.pending_) {
    provider.capability_id = ids_->intern(provider.capability);
    provider.provider_id = ids_->intern(provider.provider);
    auto& entries = next->providers[provider.capability_id];
    const auto duplicate = std::find_if(entries.begin(), entries.end(), [&](const auto& existing) {
      return existing.provider == provider.provider;
    });
    if (duplicate != entries.end()) {
      return {StatusCode::AlreadyExists,
              "provider " + provider.provider.str() + " already supplies " + provider.capability.str()};
    }
    entries.push_back(provider);
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
      return lhs.provider < rhs.provider;
    });
  }

  try {
    for (auto& action : transaction.deferred_) {
      const auto status = action();
      if (!status.ok()) return status;
    }
  } catch (const std::exception& error) {
    return {StatusCode::Internal, std::string("deferred plugin activation failed: ") + error.what()};
  } catch (...) {
    return {StatusCode::Internal, "deferred plugin activation failed with an unknown exception"};
  }

  next->epoch = current->epoch + 1;
  std::atomic_store_explicit(&snapshot_, std::const_pointer_cast<const CapabilitySnapshot>(next),
                             std::memory_order_release);
  transaction.committed_ = true;
  return Status::success();
}

}  // namespace ec
