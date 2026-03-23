#pragma once

#include <unordered_map>
#include <memory>

#include "api.hpp"
#include "tensor/buffer.hpp"
#include "graph/graph.hpp"
#include "graph/kind.hpp"
#include "tensor/api.hpp"


namespace EC::Tr
{

struct TraceContext{
public:
    explicit TraceContext(Gr::Graph& g);
    Gr::Graph& graph();
    const Gr::Graph& graph()const;
    // tensor ->? value 
    bool has_mapping(const AT::Tensor& t)const;
    Gr::ValueId get_mapping(const AT::Tensor& t)const;
    void bind_tensor(const AT::Tensor& t,Gr::ValueId vid);

    // 构建值
    Gr::ValueId make_input(const Shape& s,DType dtype,DI device,const std::string& name="");
    Gr::ValueId capture_const(const AT::Tensor& t,const std::string& name = "");
    Gr::ValueId resolve_tensor(const AT::Tensor& t);
    void mark_output(const AT::Tensor& t,const std::string& name = "");

    void push_scope(const std::string& scope);
    void pop_scope();
    std::string current_scope() const;

private:
    Gr::Graph& graph_;
    std::unordered_map<std::shared_ptr<AT::TensorId>,Gr::ValueId> tensor_map_;
    std::vector<std::string> scope_stack_;

};



} // namespace EC::Tr




// namespace EC{

// enum class ExecMode{
//     Eager,
//     Trace
// };

// inline thread_local ExecMode g_mode = ExecMode::Eager;


// namespace Tr{

// struct TensorKey{
//     const Buffer* buf =  nullptr;
//     size_t offset_bytes;
//     bool operator==(const TensorKey& o)const{return buf == o.buf && offset_bytes == o.offset_bytes;}
// };
// struct TensorKeyHash{
//     size_t operator()(const TensorKey& k) const noexcept {
//         size_t h1 = std::hash<const Buffer*>{}(k.buf);
//         size_t h2 = std::hash<size_t>{}(k.offset_bytes);

//         // 经典 hash combine
//         return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
//     }};

// using CatpureKind = Gr::ValueKind;

// struct TraceContext : ITracer{
//     Gr::Graph g;
//     // 非 owning ptr
//     std::unordered_map<TensorKey,Gr::ValueId,TensorKeyHash> capture_map;
//     CatpureKind default_capture = CatpureKind::Const;

//     bool is_tracing() const override {return true;}

//     Gr::ValueId value_of(const AT::Tensor& t)override{
//         if(t.is_symbolic()) return t.sym();
//         return capture_tensor(t,default_capture);
//     }

//     Gr::ValueId make_value_like(const AT::Tensor& like, bool req_grad) override {
//         Gr::TensorMeta meta{like.shape(), like.dtype(), like.device()};
//         return g.new_value(meta, /*kind*/ Gr::ValueKind::Temp, req_grad, "tmp");
//     }
//      void record_op(TOp op,
//                    const std::vector<Gr::ValueId>& ins,
//                    const std::vector<Gr::ValueId>& outs) override {
//         g.new_node(op, ins, outs, "op");
//     }

//     Gr::ValueId capture_tensor(const AT::Tensor& t,CatpureKind kind){
//         TensorKey key{t.buffer_ptr(),t.offset_bytes()};
//         auto it = capture_map.find(key);
//         if(it!=capture_map.end())return it->second;
//         Gr::TensorMeta meta{t.shape(),t.dtype(),t.device()};
//         bool req_grad = (kind == CatpureKind::Param)? t.requires_grad() : false;

//         auto vid = g.new_value(meta,kind,req_grad,"capture");
//         capture_map.emplace(key,vid);
//         return vid;
//     }

//     std::vector<AT::Tensor> captured_tensors_keep_alive;
// };





// template<typename F>
// TraceContext trace(F&& fn){
//     TraceContext ctx;
//     {
//         TraceGuard guard(ctx);
//         fn();
//     }
//     return ctx;
// }
// }


// // inline thread_local Tr::TraceContext* g_trace = nullptr;

// }