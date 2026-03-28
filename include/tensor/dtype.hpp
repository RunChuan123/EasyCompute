#pragma once

#include <cstddef>
#include <exception>
#include <cstdint>
#include <string>

#include "util/err.hpp"

namespace EC
{

enum class DType:uint8_t{
    f64=0,
    f32,
    f16,
    bf16,
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

#ifdef EC_ENABLE_CUDA
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
        case DType::u8:    return "u8";
        case DType::bool_: return "bool";
        default:           return "unknown";
    }
}

template<typename T>
struct PrimToDType;

template<>
struct PrimToDType<float>{
    static constexpr DType value = DType::f32;
};
template<>
struct PrimToDType<double>{
    static constexpr DType value = DType::f64;
};
template<>
struct PrimToDType<int8_t>{
    static constexpr DType value = DType::i8;
};

template<>
struct PrimToDType<int16_t>{
    static constexpr DType value = DType::i16;
};
template<>
struct PrimToDType<int32_t>{
    static constexpr DType value = DType::i32;
};
template<>
struct PrimToDType<uint8_t>{
    static constexpr DType value = DType::u8;
};

template<typename T>
inline constexpr DType PrimToDtype_v = PrimToDType<std::remove_cv_t<T>>::value;

template<typename T>
inline DType prim_dtype(){
    return PrimToDtype_v<T>;
}

template<typename T>
inline void check_dtype_match(DType dt){
    if(prim_dtype<T>() != dt){
        throw TypeException("type mismatch");
    }
}

// template<typename F>
// decltype(auto) dispatch_dtype(DType dt,F&& f){
//     switch (dt)
//     {
//         case DType::f32: return f.template operator()<float>();
//         case DType::f64: return f.template operator()<double>();
//         case DType::f16: return f.template operator()<_Float16>();
// #ifdef EC_ENABLE_CUDA
//         case DType::bf16: return f.template operator()<__nv_float16>();
// #endif
//         case DType::i32: return f.template operator()<int32_t>();
//         case DType::i16: return f.template operator()<int16_t>();
//         case DType::i8: return f.template operator()<int8_t>();
//         case DType::u8: return f.template operator()<uint8_t>();
//         case DType::bool_: return f.template operator()<bool>();

        
//         default: throw DTypeException("unknow type");
//     }
// }


} // namespace EC
