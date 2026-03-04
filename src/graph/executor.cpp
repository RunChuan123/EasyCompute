#include <stdexcept>
#include "graph/executor.hpp"

#include "tensor/tensor_op.hpp"  // lookup, TOp

namespace EC::Gr {


RunResult Executor::run(const Graph& g, const std::unordered_map<ValueId, AT::Tensor>& feeds) const {
    std::unordered_map<ValueId, AT::Tensor> table;

    // 绑定输入/参数/常量
    for (auto vid : g.inputs) {
        auto it = feeds.find(vid);
        if (it == feeds.end()) throw std::runtime_error("Missing feed for graph input");
        table.emplace(vid, it->second);
    }

    // 执行节点（假设 nodes 已拓扑序）
    for (const auto& n : g.nodes) {
        // 准备输入 tensors
        std::vector<IValue> in;
        in.reserve(n.inputs.size());
        for (auto vin : n.inputs) {
            auto it = table.find(vin);
            if (it == table.end()) throw std::runtime_error("Missing input tensor for node");
            in.emplace_back(it->second);
        }

        // device/dtype：起步先从第一个输入取
        const  AT::Tensor& t0 = ivalue_cast< AT::Tensor>(in[0]);
        AT::KernelContext k{ t0.device(), t0.dtype(), std::move(in) };

        // attrs（如果你 Node 存 IValue，就直接塞进 k.attrs）
        // k.attrs = n.attrs;

        // lookup + call
        auto fn = AT::lookup(n.op, t0.dtype(), t0.device());
        fn(k);

        // 写回 outputs
        if (n.outputs.size() == 1) {
            table[n.outputs[0]] = k.output< AT::Tensor>(0);
        } else {
            for (size_t i = 0; i < n.outputs.size(); ++i) {
                table[n.outputs[i]] = k.output< AT::Tensor>(i);
            }
        }
    }

    RunResult rr;
    rr.outputs.reserve(g.outputs.size());
    for (auto vid : g.outputs) rr.outputs.push_back(table.at(vid));
    return rr;
}

} // namespace EC::Graph