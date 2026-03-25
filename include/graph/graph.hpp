#pragma once

#include <cstdint>
#include <unordered_map>

#include "node.hpp"
#include "tensor/api.hpp"

namespace EC::Gr
{

struct IODesc{
    ValueId v_id;
    std::string name;
};

// interface
struct GraphSignature{
    std::vector<IODesc> inputs;
    std::vector<IODesc> outputs;
};

struct Graph{
    std::vector<Node> nodes;
    std::vector<Value> values;

    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;

    std::unordered_map<ValueId,AT::Tensor> const_table;

    ValueId next_value_id = 0;
    NodeId  next_node_id = 0;

    ValueId new_value(const AT::TensorMeta& meta, ValueKind kind,std::string name );

    NodeId new_node(TOp op,std::vector<ValueId> in,std::vector<ValueId> out,const std::vector<IValue>& attrs={},std::string name ="",std::string scope = "");

    Value& value(ValueId id);
    const Value& value(ValueId id)const;

    Node& node(NodeId id);
    const Node& node(NodeId id)const;

    // interface
    void mark_output(ValueId id);
    void rename_value(ValueId id,std::string name);

    // consts
    void set_const(ValueId id,const AT::Tensor& t);
    bool is_const(ValueId id)const;
    const AT::Tensor& get_const(ValueId id)const;

    void validate() const;
};

struct GraphModule{
    std::string name;
    Graph graph;
    GraphSignature signature;

    std::unordered_map<std::string,std::string> meta;
};

    
} 
