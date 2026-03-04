#pragma once

#include <vector>
#include <memory>

#include "tensor/api.hpp"
// #include "tensor/tensor.hpp"
// #include "tensor/tensor_op.hpp"
namespace EC::Gr{

using ValueId = int32_t;
using NodeId = int32_t;

enum class ValueKind : uint8_t{
    Input,
    Param,
    Const,
    Temp
};

struct TensorMeta{
    Shape shape;
    DType dtype;
    Device device;
};

struct Value{
    ValueId id;
    TensorMeta meta;
    ValueKind kind = ValueKind::Temp;
    bool requires_grad = false;

    NodeId producer = -1;
    std::vector<NodeId> users;
    std::string name;
};

struct Node{
    NodeId id;
    TOp op;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    std::vector<ValueId> attrs;
    
    std::string name;   
    AT::KernelContext ctx;

};






}