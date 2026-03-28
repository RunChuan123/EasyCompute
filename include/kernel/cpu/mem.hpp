#pragma once

#include <cstring>
#include <type_traits>
#include <vector>

#include "tensor/api.hpp"
namespace EC::AT::CPU {

template<typename T>
inline void fill(void* dst, size_t n, T value) {
    T* p = static_cast<T*>(dst);
    for (size_t i = 0; i < n; ++i) p[i] = value;
}

inline void memcpy_bytes(void* dst, const void* src, size_t nbytes) {
    if (nbytes == 0) return;
    std::memcpy(dst, src, nbytes);
}

template<typename Src, typename Dst>
inline void cast_contiguous_kernel(const void* src_void, void* dst_void, size_t n) {
    const Src* src = static_cast<const Src*>(src_void);
    Dst* dst = static_cast<Dst*>(dst_void);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<Dst>(src[i]);
    }
}

inline void cast_contiguous(const void* src,
                            DType src_dtype,
                            void* dst,
                            DType dst_dtype,
                            size_t n) {
    if (src_dtype == dst_dtype) {
        std::memcpy(dst, src, n * size_dtype(src_dtype));
        return;
    }

    switch (src_dtype) {
        case DType::f32:
            switch (dst_dtype) {
                case DType::f64: cast_contiguous_kernel<float, double>(src, dst, n); return;
                case DType::i32: cast_contiguous_kernel<float, int32_t>(src, dst, n); return;
                case DType::u8:  cast_contiguous_kernel<float, uint8_t>(src, dst, n); return;
                case DType::bool_: cast_contiguous_kernel<float, bool>(src, dst, n); return;
                default: break;
            }
            break;

        case DType::f64:
            switch (dst_dtype) {
                case DType::f32: cast_contiguous_kernel<double, float>(src, dst, n); return;
                case DType::i32: cast_contiguous_kernel<double, int32_t>(src, dst, n); return;
                default: break;
            }
            break;

        case DType::i32:
            switch (dst_dtype) {
                case DType::f32: cast_contiguous_kernel<int32_t, float>(src, dst, n); return;
                case DType::f64: cast_contiguous_kernel<int32_t, double>(src, dst, n); return;
                default: break;
            }
            break;


        case DType::u8:
            switch (dst_dtype) {
                case DType::f32: cast_contiguous_kernel<uint8_t, float>(src, dst, n); return;
                case DType::i32: cast_contiguous_kernel<uint8_t, int32_t>(src, dst, n); return;
                default: break;
            }
            break;

        case DType::bool_:
            switch (dst_dtype) {
                case DType::f32: cast_contiguous_kernel<bool, float>(src, dst, n); return;
                case DType::i32: cast_contiguous_kernel<bool, int32_t>(src, dst, n); return;
                default: break;
            }
            break;

        default:
            break;
    }

    throw TypeException("cast_contiguous: unsupported dtype conversion");
}

inline void copy_strided_to_contiguous(
    const void* src,
    DType dtype,
    const Shape& src_shape,
    const std::vector<size_t>& src_strides,
    void* dst) {

    const size_t itemsize = size_dtype(dtype);
    const size_t n = src_shape.numel();

    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);

    for (size_t linear = 0; linear < n; ++linear) {
        auto idx = detail::unravel_index(linear, src_shape);

        size_t src_off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            src_off += idx[i] * src_strides[i];
        }

        std::memcpy(dst_bytes + linear * itemsize,
                    src_bytes + src_off * itemsize,
                    itemsize);
    }
}

inline void copy_contiguous_to_strided(
    const void* src,
    DType dtype,
    const Shape& dst_shape,
    const std::vector<size_t>& dst_strides,
    void* dst) {

    const size_t itemsize = size_dtype(dtype);
    const size_t n = dst_shape.numel();

    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);

    for (size_t linear = 0; linear < n; ++linear) {
        auto idx = detail::unravel_index(linear, dst_shape);

        size_t dst_off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            dst_off += idx[i] * dst_strides[i];
        }

        std::memcpy(dst_bytes + dst_off * itemsize,
                    src_bytes + linear * itemsize,
                    itemsize);
    }
}

inline void copy_strided_to_strided(
    const void* src,
    DType dtype,
    const Shape& shape,
    const std::vector<size_t>& src_strides,
    void* dst,
    const std::vector<size_t>& dst_strides) {

    const size_t itemsize = size_dtype(dtype);
    const size_t n = shape.numel();

    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);

    for (size_t linear = 0; linear < n; ++linear) {
        auto idx = detail::unravel_index(linear, shape);

        size_t src_off = 0;
        size_t dst_off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            src_off += idx[i] * src_strides[i];
            dst_off += idx[i] * dst_strides[i];
        }

        std::memcpy(dst_bytes + dst_off * itemsize,
                    src_bytes + src_off * itemsize,
                    itemsize);
    }
}

} // namespace EC::AT::cpu