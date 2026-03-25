// #include "graph/compose.hpp"


// namespace EC::Gr
// {

//     // 这里是 tensor 还是 tensorMeta？
// InlineResult inline_graph(
//     Graph& dst,
//     const Graph& src,
//     const std::vector<ValueId>& bound_inputs,
//     const std::string& scope_prefix
// ) {
//     if (src.inputs.size() != bound_inputs.size()) {
//         throw std::runtime_error("inline_graph: input size mismatch");
//     }

//     CloneMap cmap;

//     // 1. 绑定输入
//     for (size_t i = 0; i < src.inputs.size(); ++i) {
//         cmap.value_map[src.inputs[i]] = bound_inputs[i];
//     }

//     // 2. 复制常量
//     for (const auto& v : src.values) {
//         if (v.kind != ValueKind::Const) continue;

//         auto new_vid = dst.new_value(
//             ValueKind::Const, v.shape, v.dtype, v.device, v.name
//         );
//         dst.set_const(new_vid, src.get_const(v.id));
//         cmap.value_map[v.id] = new_vid;
//     }

//     // 3. 按 node 顺序复制 temp value + node
//     for (const auto& n : src.nodes) {
//         std::vector<ValueId> new_inputs;
//         for (auto in : n.inputs) {
//             new_inputs.push_back(cmap.value_map.at(in));
//         }

//         std::vector<ValueId> new_outputs;
//         for (auto out : n.outputs) {
//             const auto& old_v = src.value(out);
//             auto new_vid = dst.new_value(
//                 old_v.kind, old_v.shape, old_v.dtype, old_v.device, old_v.name
//             );
//             cmap.value_map[out] = new_vid;
//             new_outputs.push_back(new_vid);
//         }

//         auto new_scope = scope_prefix.empty() ? n.scope : (scope_prefix + "/" + n.scope);

//         auto new_nid = dst.new_node(
//             n.op, new_inputs, new_outputs, n.attrs, n.name, new_scope
//         );
//         cmap.node_map[n.id] = new_nid;
//     }

//     // 4. 收集 outputs
//     InlineResult res;
//     res.mapping = std::move(cmap);

//     for (auto out : src.outputs) {
//         res.outputs.push_back(res.mapping.value_map.at(out));
//     }

//     return res;
// }

// } // namespace EC::Gr
