#include "graph/graph.hpp"

namespace EC::Gr
{

ValueId Graph::new_value(const AT::TensorMeta& meta, ValueKind kind,std::string name){
    ValueId id = next_value_id++;
    values.push_back(Value{id,meta,kind,std::move(name),std::nullopt,{}});
    if(kind == ValueKind::Input || kind == ValueKind::Param) inputs.push_back(id);
    return id;
}

NodeId Graph::new_node(TOp op,std::vector<ValueId> in,std::vector<ValueId> out,const std::vector<IValue>& attrs,std::string name ,std::string scope){
    NodeId id = next_node_id++;
    nodes.push_back(Node{id,op,std::move(in),std::move(out),attrs,std::move(name), std::move(scope),{}});
    for(auto vid : nodes.back().inputs) values[vid].users.push_back(id);
    for(auto vid : nodes.back().outputs) values[vid].producer = id;
    return id;
}

Value& Graph::value(ValueId id){
    if(id < 0 || static_cast<size_t>(id) > values.size()){
        throw std::out_of_range("Graph value invalid id");
    }
    return values[id];
}
const Value& Graph::value(ValueId id)const{
    if(id < 0 || static_cast<size_t>(id) > values.size()){
        throw std::out_of_range("Graph value invalid id");
    }
    return values[id];
}
Node& Graph::node(NodeId id){
    if(id < 0 || static_cast<size_t>(id) > nodes.size()){
        throw std::out_of_range("Graph node invalid id");
    }
    return nodes[id];
}
const Node& Graph::node(NodeId id)const{
    if(id < 0 || static_cast<size_t>(id) > nodes.size()){
        throw std::out_of_range("Graph node invalid id");
    }
    return nodes[id];
}

void Graph::mark_output(ValueId id){
    value(id);
    outputs.push_back(id);
}

void Graph::rename_value(ValueId id, std::string name) {
    value(id).name = std::move(name);
}

void Graph::set_const(ValueId id, const AT::Tensor& t) {
    auto& v = value(id);
    if (v.kind != ValueKind::Const) {
        throw GraphException("Graph::set_const: value is not Const");
    }
    const_table[id] = t;
}

bool Graph::is_const(ValueId id)const{
    return const_table.find(id) != const_table.end();
}

const AT::Tensor& Graph::get_const(ValueId id)const{
    auto it = const_table.find(id);
    if (it == const_table.end()) {
        throw GraphException("Graph::get_const: const value not found");
    }
    return it->second;
}

void Graph::validate() const {
    auto check_value_id = [this](ValueId id, const char* where) {
        if (id < 0 || static_cast<size_t>(id) >= values.size()) {
            throw GraphException(std::string("Graph::validate: invalid ValueId in ") + where);
        }
    };

    auto check_node_id = [this](NodeId id, const char* where) {
        if (id < 0 || static_cast<size_t>(id) >= nodes.size()) {
            throw GraphException(std::string("Graph::validate: invalid NodeId in ") + where);
        }
    };

    for (auto vid : inputs) {
        check_value_id(vid, "inputs");
    }

    for (auto vid : outputs) {
        check_value_id(vid, "outputs");
    }

    for (const auto& n : nodes) {
        for (auto vid : n.inputs) {
            check_value_id(vid, "node.inputs");
        }
        for (auto vid : n.outputs) {
            check_value_id(vid, "node.outputs");
        }
    }

    for (const auto& kv : const_table) {
        auto vid = kv.first;
        check_value_id(vid, "const_table");
        if (values[vid].kind != ValueKind::Const) {
            throw GraphException("Graph::validate: const_table key is not a Const value");
        }
    }

    for (const auto& v : values) {
        if (v.producer.has_value()) {
            check_node_id(*v.producer, "value.producer");
        }
        for (auto nid : v.users) {
            check_node_id(nid, "value.users");
        }
    }
}
} // namespace EC
