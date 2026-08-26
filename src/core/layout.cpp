#include "easycompute/layout/layout.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace ec::layout {
namespace {

void validate(const Shape& shape, const Stride& stride) {
  if (!same_profile(shape, stride)) throw LayoutError("shape and stride profiles are not congruent");
  for (auto extent : shape.flatten()) if (extent <= 0) throw LayoutError("layout extents must be positive");
  for (auto value : stride.flatten()) if (value < 0) throw LayoutError("negative strides are not supported yet");
}

std::vector<std::int64_t> left_strides(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::size_t i = 1; i < shape.size(); ++i) strides[i] = strides[i - 1] * shape[i - 1];
  return strides;
}

std::vector<std::int64_t> right_strides(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::size_t i = shape.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * shape[i - 1];
  return strides;
}

std::size_t checked_mode(std::int64_t mode, std::size_t rank) {
  if (mode < 0) mode += static_cast<std::int64_t>(rank);
  if (mode < 0 || static_cast<std::size_t>(mode) >= rank) throw LayoutError("layout mode is out of range");
  return static_cast<std::size_t>(mode);
}

}  // namespace

Layout::Layout(Shape shape, Stride stride) : shape_(std::move(shape)), stride_(std::move(stride)) {
  validate(shape_, stride_);
}

Layout Layout::left(Shape shape) {
  const auto flat = shape.flatten();
  return Layout(shape, Stride::from_flat_like(shape, left_strides(flat)));
}

Layout Layout::right(Shape shape) {
  const auto flat = shape.flatten();
  return Layout(shape, Stride::from_flat_like(shape, right_strides(flat)));
}

std::int64_t Layout::cosize() const {
  const auto extents = shape_.flatten();
  const auto strides = stride_.flatten();
  std::int64_t maximum = 0;
  for (std::size_t i = 0; i < extents.size(); ++i) maximum += (extents[i] - 1) * strides[i];
  return maximum + 1;
}

std::int64_t Layout::operator()(const Coord& coord) const {
  if (!same_profile(shape_, coord)) throw LayoutError("coordinate profile does not match layout shape");
  const auto extents = shape_.flatten();
  const auto strides = stride_.flatten();
  const auto coords = coord.flatten();
  std::int64_t result = 0;
  for (std::size_t i = 0; i < coords.size(); ++i) {
    if (coords[i] < 0 || coords[i] >= extents[i]) throw LayoutError("coordinate is out of range");
    result += coords[i] * strides[i];
  }
  return result;
}

std::int64_t Layout::operator()(std::int64_t linear_coord) const {
  if (linear_coord < 0 || linear_coord >= size()) throw LayoutError("linear coordinate is out of range");
  const auto extents = shape_.flatten();
  std::vector<std::int64_t> coords(extents.size(), 0);
  for (std::size_t i = 0; i < extents.size(); ++i) {
    coords[i] = linear_coord % extents[i];
    linear_coord /= extents[i];
  }
  return (*this)(Coord::from_flat_like(shape_, coords));
}

bool Layout::is_contiguous_left() const { return *this == left(shape_); }
bool Layout::is_contiguous_right() const { return *this == right(shape_); }

bool Layout::is_injective() const {
  if (size() > 1'000'000) return false;
  std::unordered_set<std::int64_t> seen;
  seen.reserve(static_cast<std::size_t>(size()));
  for (std::int64_t i = 0; i < size(); ++i) if (!seen.insert((*this)(i)).second) return false;
  return true;
}

Layout Layout::coalesce() const {
  const auto extents = shape_.flatten();
  const auto strides = stride_.flatten();
  std::vector<std::int64_t> out_shape;
  std::vector<std::int64_t> out_stride;
  for (std::size_t i = 0; i < extents.size(); ++i) {
    if (extents[i] == 1) continue;
    if (!out_shape.empty() && strides[i] == out_stride.back() * out_shape.back()) {
      out_shape.back() *= extents[i];
    } else {
      out_shape.push_back(extents[i]);
      out_stride.push_back(strides[i]);
    }
  }
  if (out_shape.empty()) { out_shape.push_back(1); out_stride.push_back(0); }
  IntTuple::Children shape_nodes;
  IntTuple::Children stride_nodes;
  for (auto value : out_shape) shape_nodes.emplace_back(value);
  for (auto value : out_stride) stride_nodes.emplace_back(value);
  return Layout(Shape(std::move(shape_nodes)), Stride(std::move(stride_nodes)));
}

Layout Layout::permute(std::span<const std::int64_t> modes) const {
  if (shape_.is_leaf() || stride_.is_leaf()) throw LayoutError("permute requires a tuple at the root");
  const auto rank = shape_.children().size();
  if (modes.size() != rank) throw LayoutError("permutation rank mismatch");
  std::vector<bool> used(rank, false);
  IntTuple::Children new_shape;
  IntTuple::Children new_stride;
  new_shape.reserve(rank); new_stride.reserve(rank);
  for (auto mode : modes) {
    const auto index = checked_mode(mode, rank);
    if (used[index]) throw LayoutError("permutation contains a repeated mode");
    used[index] = true;
    new_shape.push_back(shape_.children()[index]);
    new_stride.push_back(stride_.children()[index]);
  }
  return Layout(Shape(std::move(new_shape)), Stride(std::move(new_stride)));
}

Layout Layout::transpose(std::int64_t mode_a, std::int64_t mode_b) const {
  if (shape_.is_leaf()) throw LayoutError("transpose requires a tuple at the root");
  const auto rank = shape_.children().size();
  const auto a = checked_mode(mode_a, rank);
  const auto b = checked_mode(mode_b, rank);
  std::vector<std::int64_t> modes(rank);
  std::iota(modes.begin(), modes.end(), 0);
  std::swap(modes[a], modes[b]);
  return permute(modes);
}

std::string Layout::str() const { return shape_.str() + ':' + stride_.str(); }

LayoutFunction::LayoutFunction(Layout layout)
    : domain_(layout.shape()), map_([layout](std::int64_t x) { return layout(x); }),
      expression_(layout.str()) {}

LayoutFunction::LayoutFunction(Shape domain, std::function<std::int64_t(std::int64_t)> map,
                               std::string expression)
    : domain_(std::move(domain)), map_(std::move(map)), expression_(std::move(expression)) {
  if (!map_) throw LayoutError("LayoutFunction requires a mapping");
}

std::int64_t LayoutFunction::operator()(std::int64_t linear_coord) const {
  if (linear_coord < 0 || linear_coord >= size()) throw LayoutError("LayoutFunction coordinate is out of range");
  return map_(linear_coord);
}

LayoutFunction composition(LayoutFunction outer, LayoutFunction inner) {
  const auto domain = inner.shape();
  const auto expression = '(' + outer.expression() + " o " + inner.expression() + ')';
  return LayoutFunction(domain,
      [outer = std::move(outer), inner = std::move(inner)](std::int64_t x) { return outer(inner(x)); },
      expression);
}

}  // namespace ec::layout
