/**
 * 访存事务
 */

#pragma once

#include "tensor.hpp"
#include "device.hpp"
#include "backends/abstract.hpp"


namespace EC::AT
{
    
enum class AccessDev{
    Host,
    Device
};

enum class AccessMode{
    Read,
    Write,
    ReadWrite
};


struct ViewDesc{
    AccessDev dev;
    AccessMode mode;
    DI device;
    Dev::StreamHandle stream{};
    bool async = false;
};

struct TensorView{
    
};

struct TensorRView : TensorView{

};

struct TensorRWView : TensorView{

};


} // namespace EC::AT
