#pragma once

#include <vector>
#include <unordered_map>

#include "tensor/api.hpp"
#include "graph.hpp"

namespace EC::Gr
{   
struct RunResult{
    std::vector<AT::Tensor> outputs;
};

std::unordered_map<ValueId, AT::Tensor> make_feeds(
    const GraphModule& mod,
    const std::unordered_map<std::string, AT::Tensor>& named_inputs
);

struct Executor{
    RunResult run(const Graph& g,const std::unordered_map<ValueId,AT::Tensor>& feeds)const;
};


} // namespace EC
