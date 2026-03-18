#pragma once

#include <cstddef>
#include <cstdint>

namespace EC
{

enum class DType:uint8_t{
    f64=0,
    f32,
    f16,
    bf16,
    i64,
    i32,
    i16,
    i8,
    u8,
    bool_,
    NumDType,
    Unknow
};


// 基本类型转 DType
template<typename T>
constexpr DType get_dtype();
// 特化float → f32
template<>
constexpr DType get_dtype<float>() {
    return DType::f32;
}

// 特化int → i32
template<>
constexpr DType get_dtype<int>() {
    return DType::i32;
}

// 特化double → f64
template<>
constexpr DType get_dtype<double>() {
    return DType::f64;
}

// 特化int8_t → i8
template<>
constexpr DType get_dtype<int8_t>() {
    return DType::i8;
}

#ifdef CUDA_ENABLED
template<>
constexpr DType get_dtype<__nv_bfloat>() {
    return DType::i8;
}

#endif

template<typename T>
using DTypeOf = decltype(get_dtype<T>());

constexpr size_t size_dtype(DType dtype) {
    switch (dtype) {
        case DType::f16:   return 2;
        case DType::f32:   return 4;
        case DType::f64:   return 8;
        case DType::i8:    return 1;
        case DType::i16:   return 2;
        case DType::i32:   return 4;
        case DType::i64:   return 8;
        case DType::u8:    return 1;
        case DType::bool_: return 1;
        default:
            throw TypeException("size_dtype: unknown dtype");
    }
}

inline const char* name_dtype(DType dtype) {
    switch (dtype) {
        case DType::f16:   return "f16";
        case DType::f32:   return "f32";
        case DType::f64:   return "f64";
        case DType::i8:    return "i8";
        case DType::i16:   return "i16";
        case DType::i32:   return "i32";
        case DType::i64:   return "i64";
        case DType::u8:    return "u8";
        case DType::bool_: return "bool";
        default:           return "unknown";
    }
}

} // namespace EC
