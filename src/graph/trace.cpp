#include "graph/trace/ctx.hpp"
#include "graph/trace/guard.hpp"

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
    AT::TensorMeta meta(s,dtype,device,req_grads);
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
// definition end

// 当前文件私有
namespace {
    thread_local TraceMode g_trace_mode = TraceMode::Off;
    thread_local TraceContext* g_current_tracer = nullptr;
}

TraceMode current_trace_mode() {
    return g_trace_mode;
}

void set_trace_mode(TraceMode mode) {
    g_trace_mode = mode;
}

TraceContext* current_tracer() {
    return g_current_tracer;
}

void set_current_tracer(TraceContext* ctx) {
    g_current_tracer = ctx;
}

// TraceGuard

TraceGuard::TraceGuard(TraceContext* ctx)
    :prev_ctx_(current_tracer()),prev_mode_(current_trace_mode())
{
    set_trace_mode(TraceMode::On);
    set_current_tracer(ctx);
}
TraceGuard::~TraceGuard() {
    set_current_tracer(prev_ctx_);
    set_trace_mode(prev_mode_);
}

template <class Fn>
auto trace(Fn&& fn) -> std::pair< Gr::Graph, decltype(fn())>{
    Gr::Graph g;
    TraceContext ctx(g);
    TraceGuard guard(ctx);

    auto out = fn();
    ctx.mark_output(out);
    g.validate();
    return {std::move(g),std::move(out)};
}

}