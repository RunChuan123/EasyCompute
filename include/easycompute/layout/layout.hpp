#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "easycompute/layout/int_tuple.hpp"

namespace ec::layout {

class Layout {
public:
  Layout(Shape shape, Stride stride);

  static Layout left(Shape shape);
  static Layout right(Shape shape);

  const Shape& shape() const { return shape_; }
  const Stride& stride() const { return stride_; }
  std::int64_t size() const { return shape_.product(); }
  std::int64_t cosize() const;

  std::int64_t operator()(const Coord& coord) const;
  std::int64_t operator()(std::int64_t linear_coord) const;

  bool is_contiguous_left() const;
  bool is_contiguous_right() const;
  bool is_injective() const;

  Layout coalesce() const;
  Layout permute(std::span<const std::int64_t> modes) const;
  Layout transpose(std::int64_t mode_a, std::int64_t mode_b) const;
  std::string str() const;

  friend bool operator==(const Layout&, const Layout&) = default;

private:
  Shape shape_;
  Stride stride_;
};

// General layouts are functions. A composed layout intentionally has no stride:
// non-affine composition cannot always be represented by one shape/stride pair.
class LayoutFunction {
public:
  explicit LayoutFunction(Layout layout);
  LayoutFunction(Shape domain, std::function<std::int64_t(std::int64_t)> map, std::string expression);

  const Shape& shape() const { return domain_; }
  std::int64_t size() const { return domain_.product(); }
  std::int64_t operator()(std::int64_t linear_coord) const;
  const std::string& expression() const { return expression_; }

private:
  Shape domain_;
  std::function<std::int64_t(std::int64_t)> map_;
  std::string expression_;
};

LayoutFunction composition(LayoutFunction outer, LayoutFunction inner);

}  // namespace ec::layout

