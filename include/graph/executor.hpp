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

struct Executor{
    RunResult run(const Graph& g,const std::unordered_map<ValueId,AT::Tensor>& feeds)const;
};


} // namespace EC
