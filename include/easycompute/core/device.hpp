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
  std::string str() const {
    switch (type_) {
      case DeviceType::CPU: return "cpu:" + std::to_string(index_);
      case DeviceType::CUDA: return "cuda:" + std::to_string(index_);
    }
    return "device(" + std::to_string(static_cast<int>(type_)) + "):" + std::to_string(index_);
  }
  constexpr bool operator==(const Device& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }
  constexpr bool operator!=(const Device& other) const { return !(*this == other); }

private:
  DeviceType type_;
  int index_;
};

}  // namespace ec
