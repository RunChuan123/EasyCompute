#include <stdexcept>
#include "graph/executor.hpp"

#include "kernel/tensor_op.hpp"  // lookup, TOp

namespace EC::Gr {


RunResult Executor::run(const Graph& g, const std::unordered_map<ValueId, AT::Tensor>& feeds){
    std::unordered_map<ValueId, AT::Tensor> table;
    for(const auto& [vid,t] : g.const_table){
        table.emplace(vid,t);
    }

    // 绑定输入/参数/常量
    for (auto vid : g.inputs) {
        auto it = feeds.find(vid);
        if (it == feeds.end()) throw ExecuteException("Missing feed for graph input");
        table.emplace(vid, it->second);
    }

    // 执行节点（假设 nodes 已拓扑序）
    for (const auto& n : g.nodes) {
        // 准备输入 tensors
        std::vector<IValue> ins;
        ins.reserve(n.inputs.size());
        for (auto vid : n.inputs) {
            auto it = table.find(vid);
            if (it == table.end()) throw ExecuteException("Missing input tensor for node");
            ins.push_back(it->second);
        }

        AT::TensorMeta tmp = g.value(n.inputs[0]).meta;
        auto kernel = AT::GlobalKernelTable().lookup(n.op,tmp.dtype,tmp.device.type());
        if(!kernel) throw std::runtime_error("kernel not found");

        AT::KernelContext kctx;
        kctx.inputs = ins;
        kctx.attrs = n.attrs;
        kernel(kctx);
        if (kctx.outputs.size() != n.outputs.size()){
            throw std::runtime_error("Executor::run: kernel outputs size mismatch");
        }

        for(size_t i = 0;i<n.outputs.size();i++){
            table[n.outputs[i]] = std::move(kctx.output<AT::Tensor>(i));
        }
    }

    RunResult rr;
    rr.reserve(g.outputs.size());
    for (auto vid : g.outputs) {
        auto it = table.find(vid);
        if(it == table.end())
            throw std::runtime_error("Executor::run: missing graph output");
        rr.push_back(it->second);
    }
    return rr;
}

} // namespace EC::Graph