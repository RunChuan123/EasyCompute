#include "graph/graph.hpp"

namespace EC::Gr
{

ValueId Graph::new_value(const TensorMeta& meta, ValueKind kind,bool req_grad,std::string name){
    ValueId id = values.size();
    values.push_back(Value{id,meta,kind,req_grad,-1,{},std::move(name)});
    if(kind == ValueKind::Input || kind == ValueKind::Param) inputs.push_back(id);
    return id;
}

NodeId Graph::new_node(TOp op,std::vector<ValueId> in,std::vector<ValueId> out,std::string name){
    NodeId id = nodes.size();
    nodes.push_back(Node{id,op,std::move(in),std::move(out),{},std::move(name),{}});
    for(auto vid : nodes.back().inputs) values[vid].users.push_back(id);
    for(auto vid : nodes.back().outputs) values[vid].producer = id;
    return id;
}

    
} // namespace EC

