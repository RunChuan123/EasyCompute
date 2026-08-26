#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "easycompute/core/error.hpp"

namespace ec::layout {

class IntTuple {
public:
  using Children = std::vector<IntTuple>;

  IntTuple();
  IntTuple(std::int64_t value);
  IntTuple(std::initializer_list<IntTuple> children);
  explicit IntTuple(Children children);

  bool is_leaf() const;
  std::int64_t value() const;
  const Children& children() const;
  std::size_t depth() const;
  std::size_t leaf_count() const;
  std::int64_t product() const;
  std::vector<std::int64_t> flatten() const;
  std::string str() const;

  static IntTuple from_flat_like(const IntTuple& profile, std::span<const std::int64_t> values);
  friend bool same_profile(const IntTuple& lhs, const IntTuple& rhs);
  friend bool operator==(const IntTuple&, const IntTuple&) = default;

private:
  std::variant<std::int64_t, Children> data_;
};

using Shape = IntTuple;
using Stride = IntTuple;
using Coord = IntTuple;

bool same_profile(const IntTuple& lhs, const IntTuple& rhs);

}  // namespace ec::layout

