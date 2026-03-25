
#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include "tensor/api.hpp"
#include "graph/graph.hpp"
#include "ctx.hpp"
#include "guard.hpp"

namespace EC::Tr
{

AT::Tensor input(const Shape& shape,DType dtype,DI device,bool req_grads,const std::string& name = ""){
    auto* ctx = current_tracer();
    if(!ctx) 
        throw std::runtime_error("current mode do not support trace");
    auto vid = ctx->make_input(shape,dtype,device,req_grads,name);
    auto t = AT::Tensor::from_symbol(vid,shape,dtype,device,req_grads);
    ctx->bind_tensor(t,vid);
    return t;
}

// void mark_output(const AT::Tensor& t, const std::string& name = "");

template <class Fn>
auto trace(Fn&& fn) -> std::pair< Gr::Graph, decltype(fn())>;

template <class Fn>
Gr::GraphModule trace_module(
    const std::string& module_name,
    Fn&& fn
);

}
