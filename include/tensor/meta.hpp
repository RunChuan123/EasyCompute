#pragma once 

#include "tensor.hpp"

template<typename T,typename... Tensors>
auto get_data_ptrs_17(Tensors&... tensors) {
    return std::make_tuple(tensors.template data_ptr<T>()...);
}

