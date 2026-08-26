#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "easycompute/core/dtype.hpp"
#include "easycompute/layout/layout.hpp"
#include "easycompute/memory/storage.hpp"

namespace ec {

namespace detail { class TensorAccess; }

struct TensorImpl {
  std::shared_ptr<Storage> storage;
  layout::Layout layout;
  std::int64_t storage_offset{0};
  DType dtype{DType::Float32};
};

class Tensor {
public:
  Tensor() = default;

  static Tensor full(layout::Shape shape, float value, DType dtype = DType::Float32,
                     Device device = Device::cpu());
  static Tensor zeros(layout::Shape shape, DType dtype = DType::Float32,
                      Device device = Device::cpu());
  static Tensor arange(layout::Shape shape, DType dtype = DType::Float32,
                       Device device = Device::cpu());
  static Tensor from_floats(layout::Shape shape, std::span<const float> values,
                            DType dtype = DType::Float32, Device device = Device::cpu());

  bool defined() const { return static_cast<bool>(impl_); }
  const layout::Shape& shape() const;
  const layout::Stride& stride() const;
  const layout::Layout& get_layout() const;
  std::int64_t numel() const;
  DType dtype() const;
  Device device() const;
  std::int64_t storage_offset() const;
  bool is_contiguous() const;

  Tensor transpose(std::int64_t mode_a, std::int64_t mode_b) const;
  Tensor permute(std::span<const std::int64_t> modes) const;
  Tensor reshape(layout::Shape new_shape) const;
  Tensor contiguous() const;
  Tensor clone() const;
  Tensor to(Device destination) const;
  Tensor to(DType destination) const;

  float at(const layout::Coord& coordinate) const;
  std::vector<float> to_vector() const;
  std::string repr(std::size_t max_elements = 16) const;

private:
  friend class detail::TensorAccess;
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  const TensorImpl& impl() const;
  std::shared_ptr<TensorImpl> impl_;
};

Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor multiply(const Tensor& lhs, const Tensor& rhs);
inline Tensor operator+(const Tensor& lhs, const Tensor& rhs) { return add(lhs, rhs); }
inline Tensor operator*(const Tensor& lhs, const Tensor& rhs) { return multiply(lhs, rhs); }
std::ostream& operator<<(std::ostream& out, const Tensor& tensor);

}  // namespace ec
