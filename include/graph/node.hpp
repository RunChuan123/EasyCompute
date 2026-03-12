#pragma once

#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>

#include "tensor/api.hpp"
#include "kind.hpp"
// #include "tensor/tensor.hpp"
// #include "tensor/tensor_op.hpp"
namespace EC::Gr{

    

using ValueId = int32_t;
using NodeId = int32_t;



struct TensorMeta{
    Shape shape;
    DType dtype;
    Device device;
};

struct Value{
    ValueId id;
    TensorMeta meta;
    ValueKind kind = ValueKind::Temp;
    std::string name;
    bool requires_grad = false;

    std::optional<NodeId> producer;
    std::vector<NodeId> users;
};

struct Node{
    NodeId id;
    TOp op;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    std::vector<ValueId> attrs;
    
    std::string name;   
    std::string scope;

    std::unordered_map<std::string,std::string> meta;
    // AT::KernelContext& ctx;

};

}