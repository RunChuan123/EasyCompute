#pragma once

#include <variant>
#include <cstdint>
#include <vector>
#include <mutex>

#include "dtype.hpp"
#include "tensor.hpp"
#include "device/device.hpp"

namespace EC
{

enum class TOp{
    ew_add=0,
    ew_sub,
    ew_mul,
    ew_div,
    sin,
    sin_,
    gemm,
    gemv,
    cat,
    stack,
    reshape,
    permute,
    sigmoid,
    transpose,
    relu,
    det,
    submatrix,
    adjugate,
    inverse,
    lu_decompose_crout,
    lu_decompose_doolittle,

    NumOp

};
namespace AT{


using IValue = std::variant<
float,
double,
size_t,
int8_t,
int32_t,
Shape,
Tensor,
std::pair<Tensor,Tensor>
>;

template<typename T>
T& ivalue_cast(IValue& v) {
    if (!std::holds_alternative<T>(v)) {
        throw TensorException("IValue type mismatch");
    }
    return std::get<T>(v);
}

template<typename T>
const T& ivalue_cast(const IValue& v) {
    if (!std::holds_alternative<T>(v)) {
        throw TensorException("IValue type mismatch");
    }
    return std::get<T>(v);
}

struct KernelContext {
    Device device;
    DType dtype;

    std::vector<IValue> inputs;   
    std::vector<IValue> outputs;  
    std::vector<IValue> attrs;    // 例如 axis, transpose, ex_row, ex_col...
    template<typename T>
    const T& input(size_t i) const {
        if (i >= inputs.size())
        throw ContextException("iIputs index out of range");
        return ivalue_cast<T>(inputs[i]);
    }

    template<typename T>
    T& output(size_t i) {
        if (i >= outputs.size())
        throw ContextException("Output index out of range");
        return ivalue_cast<T>(outputs[i]);
    }

    template<typename T>
    const T& attr(size_t i) const {
        if (i >= attrs.size())
        throw ContextException("Attrs index out of range");
        return ivalue_cast<T>(attrs[i]);
    }
    template<typename T>
    void emplace_output(T&& value) {
        outputs.emplace_back(std::forward<T>(value));
    }
    template<class T>
    void set_output(size_t i, T&& v) {
        if (outputs.size() <= i) outputs.resize(i+1);
        outputs[i] = std::forward<T>(v);
    }
};



using KernelFunc = void(*)(KernelContext&);

static constexpr uint8_t NumDType_ = (int)DType::NumDType;
static constexpr uint8_t NumDevice_ = (int)Device::NumDevice;
static constexpr uint8_t NumOp_ = (int)TOp::NumOp;

extern KernelFunc kernel_table[NumDevice_][NumDType_][NumOp_];

static void not_implemented(KernelContext& ) {
    throw TensorException("Kernel not implemented for this op/dtype/device");
}

inline void init_table_default() {
    for (int dt = 0; dt < NumDType_; ++dt)
      for (int dev = 0; dev < NumDevice_; ++dev)
        for (int op = 0; op < NumOp_; ++op)
          kernel_table[dev][dt][op] = &not_implemented;
}

inline void register_kernel(TOp op, DType dt, Device dev, KernelFunc fn) {
    kernel_table[(uint8_t)dev][(uint8_t)dt][(uint8_t)op] = fn;
}

inline KernelFunc lookup(TOp op, DType dt, Device dev) {
    return kernel_table[(uint8_t)dev][(uint8_t)dt][(uint8_t)op];
}

// 便利一点，单返回直接调用 dispatch，多返回还是用 ctx.ouputs
template<typename R>
inline R dispatch(TOp op, DType dt,Device dev, auto&&... args) {
    KernelContext ctx{dev, dt,{},{},{}};
    (ctx.inputs.emplace_back(std::forward<decltype(args)>(args)), ...);

    auto fn = lookup(op, dt, dev);
    fn(ctx);

    return ctx.output<R>(0);
}

// 由各后端实现
void register_cpu_kernels();
void register_cuda_kernels();

inline void register_all_kernels() {
    static bool done = false;
    if (done) return;
    done = true;

    init_table_default();
    register_cpu_kernels();
#ifdef EC_AT_USE_CUDA
    register_cuda_kernels();
#endif
}

static std::once_flag register_flag;

inline void ensure_kernels_registered() {
    std::call_once(register_flag, [](){
        register_all_kernels();
    });
}


Tensor add(Tensor& a,Tensor& b);
Tensor sub(Tensor& a,Tensor& b);
Tensor mul(Tensor& a,Tensor& b);
Tensor div(Tensor& a,Tensor& b);
Tensor sin(Tensor& a);

inline Tensor operator+(Tensor& a,Tensor& b){return add(a,b);}
inline Tensor operator-(Tensor& a,Tensor& b){return sub(a,b);}
inline Tensor operator*(Tensor& a,Tensor& b){return mul(a,b);}
inline Tensor operator/(Tensor& a,Tensor& b){return div(a,b);}




}
} // namespace EC

