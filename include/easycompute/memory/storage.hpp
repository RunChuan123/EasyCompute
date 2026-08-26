#pragma once

#include <cstddef>
#include <memory>

#include "easycompute/memory/allocator.hpp"

namespace ec {

class Storage final {
public:
  Storage(std::size_t nbytes, Device device, Allocator& allocator);
  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  static std::shared_ptr<Storage> create(std::size_t nbytes, Device device);

  void* data() { return data_; }
  const void* data() const { return data_; }
  std::size_t nbytes() const { return nbytes_; }
  Device device() const { return device_; }
  const char* allocator_name() const noexcept { return allocator_->name(); }

private:
  void* data_{nullptr};
  std::size_t nbytes_{0};
  Device device_;
  Allocator* allocator_{nullptr};
};

}  // namespace ec

