#pragma once

#include <cstdint>

namespace EC::Gr
{

enum class ValueKind : uint8_t{
    Input,
    Param,
    Const,
    Temp
};
    
} // namespace Gr

