#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "easycompute/core/error.hpp"

namespace ec {

struct Float16 {
  std::uint16_t bits{0};
  Float16() = default;
  explicit Float16(float value);
  explicit operator float() const;
  friend bool operator==(Float16 lhs, Float16 rhs) { return lhs.bits == rhs.bits; }
};

enum class DType : std::uint8_t { Float16 = 0, Float32 = 1 };

std::size_t item_size(DType dtype);
std::string_view dtype_name(DType dtype);

template <typename T> struct dtype_of;
template <> struct dtype_of<Float16> : std::integral_constant<DType, DType::Float16> {};
template <> struct dtype_of<float> : std::integral_constant<DType, DType::Float32> {};
template <typename T> inline constexpr DType dtype_of_v = dtype_of<std::remove_cv_t<T>>::value;

}  // namespace ec

