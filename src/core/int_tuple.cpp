#include "easycompute/layout/int_tuple.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace ec::layout {
namespace {

void flatten_into(const IntTuple& tuple, std::vector<std::int64_t>& out) {
  if (tuple.is_leaf()) { out.push_back(tuple.value()); return; }
  for (const auto& child : tuple.children()) flatten_into(child, out);
}

IntTuple rebuild(const IntTuple& profile, std::span<const std::int64_t> values, std::size_t& cursor) {
  if (profile.is_leaf()) {
    if (cursor >= values.size()) throw LayoutError("not enough values for IntTuple profile");
    return IntTuple(values[cursor++]);
  }
  IntTuple::Children children;
  children.reserve(profile.children().size());
  for (const auto& child : profile.children()) children.push_back(rebuild(child, values, cursor));
  return IntTuple(std::move(children));
}

}  // namespace

IntTuple::IntTuple() : data_(Children{}) {}
IntTuple::IntTuple(std::int64_t value) : data_(value) {}
IntTuple::IntTuple(std::initializer_list<IntTuple> children) : data_(Children(children)) {}
IntTuple::IntTuple(Children children) : data_(std::move(children)) {}

bool IntTuple::is_leaf() const { return std::holds_alternative<std::int64_t>(data_); }

std::int64_t IntTuple::value() const {
  if (!is_leaf()) throw LayoutError("IntTuple node is not a leaf");
  return std::get<std::int64_t>(data_);
}

const IntTuple::Children& IntTuple::children() const {
  if (is_leaf()) throw LayoutError("IntTuple leaf has no children");
  return std::get<Children>(data_);
}

std::size_t IntTuple::depth() const {
  if (is_leaf()) return 0;
  std::size_t child_depth = 0;
  for (const auto& child : children()) child_depth = std::max(child_depth, child.depth());
  return children().empty() ? 1 : child_depth + 1;
}

std::size_t IntTuple::leaf_count() const {
  if (is_leaf()) return 1;
  std::size_t count = 0;
  for (const auto& child : children()) count += child.leaf_count();
  return count;
}

std::int64_t IntTuple::product() const {
  if (is_leaf()) return value();
  std::int64_t result = 1;
  for (const auto& child : children()) result *= child.product();
  return result;
}

std::vector<std::int64_t> IntTuple::flatten() const {
  std::vector<std::int64_t> result;
  result.reserve(leaf_count());
  flatten_into(*this, result);
  return result;
}

std::string IntTuple::str() const {
  if (is_leaf()) return std::to_string(value());
  std::ostringstream out;
  out << '(';
  for (std::size_t i = 0; i < children().size(); ++i) {
    if (i != 0) out << ',';
    out << children()[i].str();
  }
  out << ')';
  return out.str();
}

IntTuple IntTuple::from_flat_like(const IntTuple& profile, std::span<const std::int64_t> values) {
  if (values.size() != profile.leaf_count()) throw LayoutError("value count does not match IntTuple profile");
  std::size_t cursor = 0;
  return rebuild(profile, values, cursor);
}

bool same_profile(const IntTuple& lhs, const IntTuple& rhs) {
  if (lhs.is_leaf() != rhs.is_leaf()) return false;
  if (lhs.is_leaf()) return true;
  if (lhs.children().size() != rhs.children().size()) return false;
  for (std::size_t i = 0; i < lhs.children().size(); ++i) {
    if (!same_profile(lhs.children()[i], rhs.children()[i])) return false;
  }
  return true;
}

}  // namespace ec::layout

