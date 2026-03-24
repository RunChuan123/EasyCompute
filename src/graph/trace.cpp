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

Gr::ValueId TraceContext::make_input()



}