
#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include "tensor/api.hpp"

namespace EC::Tr
{

AT::Tensor input(
    const Shape& shape,
    DType dtype,
    const std::string& name = ""
);

void mark_output(const AT::Tensor& t, const std::string& name = "");

template <class Fn>
auto trace(Fn&& fn) -> std::pair<GraphIR::Graph, decltype(fn())>;

template <class Fn>
Gr::GraphModule trace_module(
    const std::string& module_name,
    Fn&& fn
);

}
