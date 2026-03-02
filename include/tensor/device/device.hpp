#pragma once

#include <cstdint>

namespace EC
{
enum class Device:uint8_t{
    CPU=0,
    NV_GPU,

    NumDevice
};
} // namespace Ec
