#include "easycompute/memory/storage.hpp"

namespace ec {

Storage::Storage(std::size_t nbytes, Device device, Allocator& allocator)
    : nbytes_(nbytes), device_(device), allocator_(&allocator) {
  data_ = allocator_->allocate(nbytes_, device_);
}

Storage::~Storage() {
  allocator_->deallocate(data_, nbytes_, device_);
  data_ = nullptr;
}

std::shared_ptr<Storage> Storage::create(std::size_t nbytes, Device device) {
  return std::make_shared<Storage>(nbytes, device, allocator_for(device));
}

}  // namespace ec

