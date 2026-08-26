#pragma once

#include <cstdint>
#include <string>

namespace ec {

class DeviceType {
public:
  constexpr DeviceType(std::uint16_t id, const char* name) : id_(id), name_(name) {}
  constexpr std::uint16_t id() const { return id_; }
  constexpr const char* name() const { return name_; }
  constexpr bool operator==(const DeviceType& other) const { return id_ == other.id_; }
  constexpr bool operator!=(const DeviceType& other) const { return !(*this == other); }

private:
  std::uint16_t id_;
  const char* name_;
};

namespace device_types {
inline constexpr DeviceType cpu{0, "cpu"};
inline constexpr DeviceType cuda{1, "cuda"};
}  // namespace device_types

class Device {
public:
  constexpr Device(DeviceType type = device_types::cpu, int index = 0) : type_(type), index_(index) {}
  static constexpr Device cpu() { return Device{device_types::cpu, 0}; }
  static constexpr Device cuda(int index = 0) { return Device{device_types::cuda, index}; }
  constexpr DeviceType type() const { return type_; }
  constexpr int index() const { return index_; }
  std::string str() const { return std::string(type_.name()) + ":" + std::to_string(index_); }
  constexpr bool operator==(const Device& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }
  constexpr bool operator!=(const Device& other) const { return !(*this == other); }

private:
  DeviceType type_;
  int index_;
};

}  // namespace ec
