#pragma once

#include <cstdint>

#include "node.hpp"
// #include "tensor/api.hpp"

namespace EC::Gr
{


struct Graph{
    std::vector<Node> nodes;
    std::vector<Value> values;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;

    ValueId new_value(const TensorMeta& meta, ValueKind kind,bool req_grad,std::string name = {});

    NodeId new_node(TOp op,std::vector<ValueId> in,std::vector<ValueId> out,std::string name = {});
};

    
} // namespace EC
