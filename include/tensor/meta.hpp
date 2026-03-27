#pragma once 

#include "tensor.hpp"

template<typename T,typename... Tensors>
auto get_data_ptrs_17(Tensors&... tensors) {
    // return std::make_tuple(tensors.template data_ptr<T>()...);
    return std::make_tuple(
        [&]() -> std::conditional_t<std::is_const_v<Tensors>, const T*, T*> {
            if constexpr (std::is_const_v<Tensors>) {
                return tensors.template data_ptr<const T>();
            } else {
                return tensors.template data_ptr<T>();
            }
        }()...
    );
}

