#pragma once

#include <cstdint>
#include <string_view>

#include "easycompute/core/device.hpp"

namespace ec::compute {

// Stable IDs are part of the future graph serialization contract.
enum class OpId : std::uint16_t {
  Add = 1,
  Multiply = 2,
};

struct AddTag {
  static constexpr OpId id = OpId::Add;
  static constexpr std::string_view name = "add";
};

struct MultiplyTag {
  static constexpr OpId id = OpId::Multiply;
  static constexpr std::string_view name = "multiply";
};

struct CPUBackendTag { static constexpr DeviceType device_type = DeviceType::CPU; };
struct CUDABackendTag { static constexpr DeviceType device_type = DeviceType::CUDA; };

std::string_view op_name(OpId op);

}  // namespace ec::compute

