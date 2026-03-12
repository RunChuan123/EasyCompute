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
    i32,
    i8,
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



// 3. 反向映射：从DType枚举获取类型大小（可选，辅助功能）
constexpr size_t size_dtype(DType dtype) {
    switch (dtype) {
        case DType::f32:  return sizeof(float);
        case DType::i32:  return sizeof(int);
        case DType::f64:  return sizeof(double);
        case DType::i8:   return sizeof(int8_t);
        default:          return 0; 
    }
}

// 4. 反向映射：从DType枚举获取类型名称（可选，调试用）
inline const char* name_dtype(DType dtype) {
    switch (dtype) {
        case DType::f32:  return "float32";
        case DType::i32:  return "int32";
        case DType::f64:  return "double";
        case DType::i8:   return "int8";
        default:          return "unknown";
    }
}

} // namespace EC
