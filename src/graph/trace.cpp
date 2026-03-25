#include "graph/trace/ctx.hpp"

namespace EC::Tr{

// TraceContext definition

Gr::Graph& TraceContext::graph(){return graph_;}
const Gr::Graph& TraceContext::graph()const{return graph_;}

Gr::ValueId TraceContext::resolve_tensor(const AT::Tensor& t){
    if(t.is_symbolic()) return t.sym();
    if(has_mapping(t)) return get_mapping(t);
    return capture_const(t);
}

bool TraceContext::has_mapping(const AT::Tensor& t)const{
    return tensor_map_.find(t.id()) != tensor_map_.end();
}

Gr::ValueId TraceContext::get_mapping(const AT::Tensor& t)const{
    auto it = tensor_map_.find(t.id());
    if(it == tensor_map_.end())
        throw TraceException("get_mapping: tensor not mapped");
    return it->second;
}

void TraceContext::bind_tensor(const AT::Tensor& t,Gr::ValueId vid){
    // if t.id() has exited?
    tensor_map_[t.id()] = vid;
}

    // 构建值
Gr::ValueId TraceContext::make_input(const Shape& s,DType dtype,DI device,bool req_grads,const std::string& name){
    AT::TensorMeta meta{
        .shape = s,
        .dtype = dtype,
        .device = device,
        .is_contiguous = true,
        .requires_grad = req_grads
    };
    return graph_.new_value(meta,Gr::ValueKind::Input,name);
}
// 将一个普通 eager tensor 固化进 graph 变成 const value
Gr::ValueId TraceContext::capture_const(const AT::Tensor& t,const std::string& name){
    if(has_mapping(t))
        return get_mapping(t);
                                                                // name repeat ok?
    auto vid = graph_.new_value(t.getMeta(),Gr::ValueKind::Const,name.empty()?"const":name);
    graph_.set_const(vid,t);
    bind_tensor(t,vid);
    return vid;
}

void TraceContext::mark_output(const AT::Tensor& t,const std::string& name){
    auto vid = resolve_tensor(t);
    if(!name.empty()){
        graph_.rename_value(vid,name);
    }
    graph_.mark_output(vid);
}

void TraceContext::push_scope(const std::string& scope){
    scope_stack_.push_back(scope);
}
void TraceContext::pop_scope(){
    if(!scope_stack_.empty()){
        scope_stack_.pop_back();
    }
}
std::string TraceContext::current_scope() const{
    if(scope_stack_.empty())return "";
    std::string out = scope_stack_[0];
    for(size_t i = 1;i<scope_stack_.size();++i){
        out+="/";
        out += scope_stack_[i];
    }
    return out;
}

}