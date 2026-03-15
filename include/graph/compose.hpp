#pragma once

#include <unordered_map>
#include <vector>

#include "node.hpp"
#include "graph.hpp"

namespace EC::Gr
{
    
struct CloneMap{
    std::unordered_map<ValueId,ValueId> value_map;
    std::unordered_map<NodeId,NodeId> node_map;
};

struct InlineResult{
    CloneMap mapping;
    std::vector<ValueId> outputs;
};

InlineResult inline_graph(Graph& dst,const Graph& src,const std::vector<ValueId>& bound_inputs,const std::string& scope_prefix="");

InlineResult instantiate_module(Graph& dst,const GraphModule& mod,const std::vector<ValueId>& bound_inputs,const std::string& instance_name);

} // namespace EC
