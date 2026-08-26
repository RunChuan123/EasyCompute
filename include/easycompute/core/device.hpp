#pragma once

#include <cstdint>
#include <string>

namespace ec {

enum class DeviceType : std::uint8_t { CPU = 0, CUDA = 1 };

class Device {
public:
  constexpr Device(DeviceType type = DeviceType::CPU, int index = 0) : type_(type), index_(index) {}
  static constexpr Device cpu() { return Device{DeviceType::CPU, 0}; }
  static constexpr Device cuda(int index = 0) { return Device{DeviceType::CUDA, index}; }
  constexpr DeviceType type() const { return type_; }
  constexpr int index() const { return index_; }
  constexpr bool is_cpu() const { return type_ == DeviceType::CPU; }
  constexpr bool is_cuda() const { return type_ == DeviceType::CUDA; }
  std::string str() const { return std::string(is_cpu() ? "cpu:" : "cuda:") + std::to_string(index_); }
  friend constexpr bool operator==(Device, Device) = default;

private:
  DeviceType type_;
  int index_;
};

}  // namespace ec

