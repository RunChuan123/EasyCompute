#include "backends/abstract.hpp"
#include "backends/device_manager.hpp"

namespace EC::Dev
{


void* DevAllocator::allocate(size_t bytes,DI device){
    switch (device.type())
    {
    case DeviceType::CPU:
        DM
        break;
    
    default:
        break;
    }
}
void DevAllocator::deallocate(void* ptr,size_t bytes,DI device,void* ctx){

}

    
} // namespace EC
