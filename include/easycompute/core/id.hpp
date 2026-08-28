#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ec {

class ExtensionKey {
public:
  ExtensionKey() = default;
  ExtensionKey(std::string namespace_name, std::string local_name, std::uint32_t semantic_major = 1);

  const std::string& namespace_name() const { return namespace_name_; }
  const std::string& local_name() const { return local_name_; }
  std::uint32_t semantic_major() const { return semantic_major_; }
  std::string str() const;
  bool empty() const { return namespace_name_.empty() || local_name_.empty(); }

  friend bool operator==(const ExtensionKey&, const ExtensionKey&) = default;
  friend bool operator<(const ExtensionKey& lhs, const ExtensionKey& rhs) {
    if (lhs.namespace_name_ != rhs.namespace_name_) return lhs.namespace_name_ < rhs.namespace_name_;
    if (lhs.local_name_ != rhs.local_name_) return lhs.local_name_ < rhs.local_name_;
    return lhs.semantic_major_ < rhs.semantic_major_;
  }

private:
  std::string namespace_name_;
  std::string local_name_;
  std::uint32_t semantic_major_{1};
};

struct ExtensionKeyHash { std::size_t operator()(const ExtensionKey& key) const noexcept; };

class RuntimeId {
public:
  constexpr RuntimeId() = default;
  explicit constexpr RuntimeId(std::uint32_t value) : value_(value) {}
  constexpr std::uint32_t value() const { return value_; }
  constexpr bool valid() const { return value_ != 0; }
  friend constexpr bool operator==(RuntimeId, RuntimeId) = default;

private:
  std::uint32_t value_{0};
};

struct RuntimeIdHash {
  std::size_t operator()(RuntimeId id) const noexcept { return id.value(); }
};

class InternTable {
public:
  RuntimeId intern(const ExtensionKey& key);
  std::optional<RuntimeId> find(const ExtensionKey& key) const;
  ExtensionKey lookup(RuntimeId id) const;
  std::size_t size() const;

private:
  mutable std::mutex mutex_;
  std::unordered_map<ExtensionKey, RuntimeId, ExtensionKeyHash> ids_;
  std::vector<ExtensionKey> keys_;
};

}  // namespace ec
