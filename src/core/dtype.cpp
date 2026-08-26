#include "easycompute/core/dtype.hpp"

#include <cstring>

namespace ec {
namespace {

std::uint32_t float_bits(float value) { std::uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); return bits; }
float bits_float(std::uint32_t bits) { float value = 0.0F; std::memcpy(&value, &bits, sizeof(value)); return value; }

std::uint16_t float_to_half_bits(float value) {
  const std::uint32_t f = float_bits(value);
  const std::uint32_t sign = (f >> 16U) & 0x8000U;
  const std::uint32_t exponent = (f >> 23U) & 0xffU;
  std::uint32_t mantissa = f & 0x7fffffU;
  if (exponent == 0xffU) return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000U;
    const int shift = 14 - half_exponent;
    std::uint32_t rounded = mantissa >> static_cast<unsigned>(shift);
    const std::uint32_t remainder = mantissa & ((1U << static_cast<unsigned>(shift)) - 1U);
    const std::uint32_t halfway = 1U << static_cast<unsigned>(shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U) != 0U)) {
    ++rounded;
    if (rounded == 0x400U) {
      rounded = 0;
      if (half_exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
      return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10U) | rounded);
}

float half_bits_to_float(std::uint16_t value) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x3ffU;
  if (exponent == 0) {
    if (mantissa == 0) return bits_float(sign);
    int shift = 0;
    while ((mantissa & 0x400U) == 0U) { mantissa <<= 1U; ++shift; }
    mantissa &= 0x3ffU;
    exponent = static_cast<std::uint32_t>(127 - 15 - shift + 1);
  } else if (exponent == 0x1fU) {
    exponent = 0xffU;
  } else {
    exponent += 127U - 15U;
  }
  return bits_float(sign | (exponent << 23U) | (mantissa << 13U));
}

}  // namespace

Float16::Float16(float value) : bits(float_to_half_bits(value)) {}
Float16::operator float() const { return half_bits_to_float(bits); }

std::size_t item_size(DType dtype) {
  switch (dtype) { case DType::Float16: return sizeof(Float16); case DType::Float32: return sizeof(float); }
  throw DTypeError("unknown dtype");
}

std::string_view dtype_name(DType dtype) {
  switch (dtype) { case DType::Float16: return "float16"; case DType::Float32: return "float32"; }
  throw DTypeError("unknown dtype");
}

}  // namespace ec
