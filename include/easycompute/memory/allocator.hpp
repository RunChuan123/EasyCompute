#pragma once

#include <cstddef>

#include "easycompute/core/device.hpp"

namespace ec {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void* allocate(std::size_t nbytes, Device device) = 0;
  virtual void deallocate(void* pointer, std::size_t nbytes, Device device) noexcept = 0;
  virtual const char* name() const noexcept = 0;
};

Allocator& allocator_for(Device device);
bool cuda_available();
void copy_bytes(void* destination, Device destination_device,
                const void* source, Device source_device, std::size_t nbytes);
void device_synchronize(Device device);

}  // namespace ec

