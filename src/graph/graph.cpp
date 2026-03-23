#include "graph/graph.hpp"

namespace EC::Gr
{

ValueId Graph::new_value(const AT::TensorMeta& meta, ValueKind kind,std::string name){
    ValueId id = next_value_id++;
    values.push_back(Value{id,meta,kind,std::move(name),std::nullopt,{}});
    if(kind == ValueKind::Input || kind == ValueKind::Param) inputs.push_back(id);
    return id;
}

NodeId Graph::new_node(TOp op,std::vector<ValueId> in,std::vector<ValueId> out,std::vector<ValueId>& attrs,std::string name ,std::string scope){
    NodeId id = next_node_id++;
    nodes.push_back(Node{id,op,std::move(in),std::move(out),attrs,std::move(name), std::move(scope),{}});
    for(auto vid : nodes.back().inputs) values[vid].users.push_back(id);
    for(auto vid : nodes.back().outputs) values[vid].producer = id;
    return id;
}

    
} // namespace EC
