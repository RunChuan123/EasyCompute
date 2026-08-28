#include "easycompute/core/id.hpp"

#include <stdexcept>

namespace ec {

ExtensionKey::ExtensionKey(std::string namespace_name, std::string local_name,
                           std::uint32_t semantic_major)
    : namespace_name_(std::move(namespace_name)), local_name_(std::move(local_name)),
      semantic_major_(semantic_major) {
  if (namespace_name_.empty() || local_name_.empty())
    throw std::invalid_argument("extension key names must not be empty");
  if (semantic_major_ == 0) throw std::invalid_argument("extension semantic major must be non-zero");
}

std::string ExtensionKey::str() const {
  return namespace_name_ + "/" + local_name_ + "@" + std::to_string(semantic_major_);
}

std::size_t ExtensionKeyHash::operator()(const ExtensionKey& key) const noexcept {
  const auto first = std::hash<std::string>{}(key.namespace_name());
  const auto second = std::hash<std::string>{}(key.local_name());
  return first ^ (second << 1U) ^ (static_cast<std::size_t>(key.semantic_major()) << 24U);
}

RuntimeId InternTable::intern(const ExtensionKey& key) {
  if (key.empty()) throw std::invalid_argument("cannot intern an empty extension key");
  std::lock_guard<std::mutex> lock(mutex_);
  if (const auto iterator = ids_.find(key); iterator != ids_.end()) return iterator->second;
  const auto id = RuntimeId{static_cast<std::uint32_t>(keys_.size() + 1)};
  keys_.push_back(key);
  ids_.emplace(key, id);
  return id;
}

std::optional<RuntimeId> InternTable::find(const ExtensionKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iterator = ids_.find(key);
  if (iterator == ids_.end()) return std::nullopt;
  return iterator->second;
}

ExtensionKey InternTable::lookup(RuntimeId id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!id.valid() || id.value() > keys_.size()) throw std::out_of_range("unknown runtime id");
  return keys_[id.value() - 1];
}

std::size_t InternTable::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return keys_.size();
}

}  // namespace ec
