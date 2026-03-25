#pragma once

#include <unordered_map>
#include <memory>

#include "tensor/buffer.hpp"
#include "graph/graph.hpp"
#include "graph/kind.hpp"
#include "tensor/api.hpp"


namespace EC::Tr
{

enum class TraceMode{
    Off,
    On
};

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
    Gr::ValueId make_input(const Shape& s,DType dtype,DI device,bool req_grads,const std::string& name="");
    Gr::ValueId capture_const(const AT::Tensor& t,const std::string& name = "");
    Gr::ValueId resolve_tensor(const AT::Tensor& t);
    void mark_output(const AT::Tensor& t,const std::string& name = "");

    void push_scope(const std::string& scope);
    void pop_scope();
    std::string current_scope() const;

private:
    Gr::Graph& graph_;
    std::unordered_map<AT::TensorId,Gr::ValueId> tensor_map_;
    std::vector<std::string> scope_stack_;

};

TraceMode current_trace_mode();
void set_trace_mode(TraceMode mode);

TraceContext* current_tracer();
void set_current_tracer(TraceContext* ctx);

} // namespace EC::Tr

