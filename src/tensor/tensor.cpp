#include "easycompute/tensor/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "easycompute/core/error.hpp"

namespace ec {

#if EC_HAS_CUDA
void launch_binary_cuda(void* output, const void* lhs, const void* rhs,
                        std::int64_t count, DType dtype, int operation, Device device);
#endif

namespace {

void require_defined(const Tensor& tensor) {
  if (!tensor.defined()) throw TensorError("operation received an undefined tensor");
}

layout::Coord row_major_coordinate(const layout::Shape& shape, std::int64_t linear) {
  const auto extents = shape.flatten();
  std::vector<std::int64_t> coordinate(extents.size(), 0);
  for (std::size_t i = extents.size(); i > 0; --i) {
    coordinate[i - 1] = linear % extents[i - 1];
    linear /= extents[i - 1];
  }
  return layout::Coord::from_flat_like(shape, coordinate);
}

float read_value(const void* base, DType dtype, std::int64_t offset) {
  switch (dtype) {
    case DType::Float16:
      return static_cast<float>(static_cast<const Float16*>(base)[offset]);
    case DType::Float32:
      return static_cast<const float*>(base)[offset];
  }
  throw DTypeError("read_value received an unknown dtype");
}

void write_value(void* base, DType dtype, std::int64_t offset, float value) {
  switch (dtype) {
    case DType::Float16:
      static_cast<Float16*>(base)[offset] = Float16(value);
      return;
    case DType::Float32:
      static_cast<float*>(base)[offset] = value;
      return;
  }
  throw DTypeError("write_value received an unknown dtype");
}

std::shared_ptr<TensorImpl> make_impl(layout::Layout tensor_layout, DType dtype, Device device) {
  const auto elements = tensor_layout.cosize();
  const auto bytes = static_cast<std::size_t>(elements) * item_size(dtype);
  return std::make_shared<TensorImpl>(TensorImpl{Storage::create(bytes, device), std::move(tensor_layout), 0, dtype});
}

void check_binary(const Tensor& lhs, const Tensor& rhs) {
  require_defined(lhs); require_defined(rhs);
  if (lhs.shape() != rhs.shape()) throw TensorError("binary operation shape mismatch");
  if (lhs.dtype() != rhs.dtype()) throw TensorError("binary operation dtype mismatch");
  if (lhs.device() != rhs.device()) throw TensorError("binary operation device mismatch");
}

}  // namespace

const TensorImpl& Tensor::impl() const { require_defined(*this); return *impl_; }
const layout::Shape& Tensor::shape() const { return impl().layout.shape(); }
const layout::Stride& Tensor::stride() const { return impl().layout.stride(); }
const layout::Layout& Tensor::get_layout() const { return impl().layout; }
std::int64_t Tensor::numel() const { return impl().layout.size(); }
DType Tensor::dtype() const { return impl().dtype; }
Device Tensor::device() const { return impl().storage->device(); }
std::int64_t Tensor::storage_offset() const { return impl().storage_offset; }
bool Tensor::is_contiguous() const { return impl().layout.is_contiguous_right(); }

Tensor Tensor::from_floats(layout::Shape shape, std::span<const float> values, DType dtype, Device device) {
  layout::Layout tensor_layout = layout::Layout::right(std::move(shape));
  if (static_cast<std::int64_t>(values.size()) != tensor_layout.size()) throw TensorError("from_floats value count mismatch");
  auto result = Tensor(make_impl(std::move(tensor_layout), dtype, device));
  std::vector<std::byte> host(static_cast<std::size_t>(result.numel()) * item_size(dtype));
  for (std::int64_t i = 0; i < result.numel(); ++i) write_value(host.data(), dtype, i, values[static_cast<std::size_t>(i)]);
  copy_bytes(result.impl_->storage->data(), device, host.data(), Device::cpu(), host.size());
  return result;
}

Tensor Tensor::full(layout::Shape shape, float value, DType dtype, Device device) {
  const auto count = shape.product();
  std::vector<float> values(static_cast<std::size_t>(count), value);
  return from_floats(std::move(shape), values, dtype, device);
}

Tensor Tensor::zeros(layout::Shape shape, DType dtype, Device device) {
  return full(std::move(shape), 0.0F, dtype, device);
}

Tensor Tensor::arange(layout::Shape shape, DType dtype, Device device) {
  const auto count = shape.product();
  std::vector<float> values(static_cast<std::size_t>(count));
  std::iota(values.begin(), values.end(), 0.0F);
  return from_floats(std::move(shape), values, dtype, device);
}

Tensor Tensor::transpose(std::int64_t mode_a, std::int64_t mode_b) const {
  return Tensor(std::make_shared<TensorImpl>(TensorImpl{impl().storage, impl().layout.transpose(mode_a, mode_b),
                                                        impl().storage_offset, impl().dtype}));
}

Tensor Tensor::permute(std::span<const std::int64_t> modes) const {
  return Tensor(std::make_shared<TensorImpl>(TensorImpl{impl().storage, impl().layout.permute(modes),
                                                        impl().storage_offset, impl().dtype}));
}

Tensor Tensor::reshape(layout::Shape new_shape) const {
  if (new_shape.product() != numel()) throw TensorError("reshape changes the number of elements");
  if (!is_contiguous()) throw TensorError("reshape requires a contiguous tensor; call contiguous() first");
  return Tensor(std::make_shared<TensorImpl>(TensorImpl{impl().storage, layout::Layout::right(std::move(new_shape)),
                                                        impl().storage_offset, impl().dtype}));
}

Tensor Tensor::contiguous() const {
  if (is_contiguous() && storage_offset() == 0) return *this;
  return from_floats(shape(), to_vector(), dtype(), device());
}

Tensor Tensor::clone() const {
  auto cloned = std::make_shared<TensorImpl>(TensorImpl{Storage::create(impl().storage->nbytes(), device()),
                                                        impl().layout, impl().storage_offset, impl().dtype});
  copy_bytes(cloned->storage->data(), device(), impl().storage->data(), device(), impl().storage->nbytes());
  return Tensor(std::move(cloned));
}

Tensor Tensor::to(Device destination) const {
  if (destination == device()) return *this;
  auto moved = std::make_shared<TensorImpl>(TensorImpl{Storage::create(impl().storage->nbytes(), destination),
                                                       impl().layout, impl().storage_offset, impl().dtype});
  copy_bytes(moved->storage->data(), destination, impl().storage->data(), device(), impl().storage->nbytes());
  return Tensor(std::move(moved));
}

Tensor Tensor::to(DType destination) const {
  if (destination == dtype()) return *this;
  return from_floats(shape(), to_vector(), destination, device());
}

float Tensor::at(const layout::Coord& coordinate) const {
  if (!device().is_cpu()) return to(Device::cpu()).at(coordinate);
  const auto offset = storage_offset() + get_layout()(coordinate);
  return read_value(impl().storage->data(), dtype(), offset);
}

std::vector<float> Tensor::to_vector() const {
  if (!device().is_cpu()) return to(Device::cpu()).to_vector();
  std::vector<float> values(static_cast<std::size_t>(numel()));
  for (std::int64_t i = 0; i < numel(); ++i) {
    const auto coordinate = row_major_coordinate(shape(), i);
    values[static_cast<std::size_t>(i)] = read_value(impl().storage->data(), dtype(),
                                                     storage_offset() + get_layout()(coordinate));
  }
  return values;
}

std::string Tensor::repr(std::size_t max_elements) const {
  if (!defined()) return "Tensor(undefined)";
  std::ostringstream out;
  out << "Tensor(shape=" << shape().str() << ", stride=" << stride().str()
      << ", dtype=" << dtype_name(dtype()) << ", device=" << device().str()
      << ", contiguous=" << std::boolalpha << is_contiguous() << ", values=[";
  const auto values = to_vector();
  const auto count = std::min(max_elements, values.size());
  out << std::setprecision(6);
  for (std::size_t i = 0; i < count; ++i) { if (i != 0) out << ", "; out << values[i]; }
  if (count < values.size()) out << ", ...";
  out << "])";
  return out.str();
}

Tensor Tensor::binary_cpu(const Tensor& lhs, const Tensor& rhs, bool is_multiply) {
  auto output = Tensor::zeros(lhs.shape(), lhs.dtype(), lhs.device());
  const auto* lhs_data = lhs.impl().storage->data();
  const auto* rhs_data = rhs.impl().storage->data();
  auto* output_data = output.impl_->storage->data();
  for (std::int64_t i = 0; i < lhs.numel(); ++i) {
    const auto coordinate = row_major_coordinate(lhs.shape(), i);
    const auto lhs_offset = lhs.storage_offset() + lhs.get_layout()(coordinate);
    const auto rhs_offset = rhs.storage_offset() + rhs.get_layout()(coordinate);
    const float a = read_value(lhs_data, lhs.dtype(), lhs_offset);
    const float b = read_value(rhs_data, rhs.dtype(), rhs_offset);
    write_value(output_data, output.dtype(), i, is_multiply ? a * b : a + b);
  }
  return output;
}

Tensor Tensor::binary_cuda(const Tensor& lhs, const Tensor& rhs, bool is_multiply) {
#if EC_HAS_CUDA
  const Tensor a = lhs.contiguous();
  const Tensor b = rhs.contiguous();
  Tensor output = Tensor::zeros(lhs.shape(), lhs.dtype(), lhs.device());
  launch_binary_cuda(output.impl_->storage->data(), a.impl().storage->data(), b.impl().storage->data(),
                     lhs.numel(), lhs.dtype(), is_multiply ? 1 : 0, lhs.device());
  return output;
#else
  (void)lhs; (void)rhs; (void)is_multiply;
  throw DeviceError("CUDA binary operation requested from a CPU-only build");
#endif
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  check_binary(lhs, rhs);
  return lhs.device().is_cpu() ? Tensor::binary_cpu(lhs, rhs, false) : Tensor::binary_cuda(lhs, rhs, false);
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
  check_binary(lhs, rhs);
  return lhs.device().is_cpu() ? Tensor::binary_cpu(lhs, rhs, true) : Tensor::binary_cuda(lhs, rhs, true);
}

std::ostream& operator<<(std::ostream& out, const Tensor& tensor) { return out << tensor.repr(); }

}  // namespace ec
