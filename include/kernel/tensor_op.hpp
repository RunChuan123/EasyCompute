#pragma once

#include <variant>
#include <cstdint>
#include <vector>
#include <mutex>

#include "tensor/api.hpp"

namespace EC
{


using IValue = std::variant<
float,
double,
size_t,
int8_t,
int32_t,
Shape,
AT::Tensor,
std::pair<AT::Tensor,AT::Tensor>
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

namespace AT{


struct KernelContext {
    DI device;
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


static constexpr uint8_t NumDType_ = (int)DType::NumDType;
static constexpr uint8_t NumDevice_ = (int)DeviceType::NumDevice;
static constexpr uint8_t NumOp_ = (int)TOp::NumOp;

static void not_implemented(KernelContext& ) {
    throw TensorException("Kernel not implemented for this op/dtype/device");
}

struct KernelTable{
    using KernelFunc = void(*)(KernelContext&);

    KernelFunc kernel_table[NumDevice_][NumDType_][NumOp_];

    KernelTable(){
        // 初始化
        for (int dt = 0; dt < NumDType_; ++dt)
            for (int dev = 0; dev < NumDevice_; ++dev)
                for (int op = 0; op < NumOp_; ++op)
                kernel_table[dev][dt][op] = &not_implemented;
    }

    inline void register_kernel(TOp op, DType dt, DeviceType dev, KernelFunc fn) {
        kernel_table[(uint8_t)dev][(uint8_t)dt][(uint8_t)op] = fn;
    }

    inline KernelFunc lookup(TOp op, DType dt, DeviceType dev) {
        return kernel_table[(uint8_t)dev][(uint8_t)dt][(uint8_t)op];
    }
    inline KernelFunc find(TOp op, DType dt = DType::f32, DeviceType dev = DeviceType::CPU){
        return kernel_table[(uint8_t)dev][(uint8_t)dt][(uint8_t)op];// 后面调整为 OpDesc
    }

    template<typename R>
    inline R dispatch_without_attrs(TOp op, DType dt,DI dev, auto&&... args) {
        KernelContext ctx{dev, dt,{},{},{}};
        (ctx.inputs.emplace_back(std::forward<decltype(args)>(args)), ...);
        auto fn = lookup(op, dt, dev.type());
        fn(ctx); 
        return ctx.output<R>(0);
    }
};

inline KernelTable& GlobalKernelTable(){
    static KernelTable table;
    return table;
} 

void register_cpu_kernels(KernelTable& kt);
void register_cuda_kernels(KernelTable& kt);

inline void register_all_kernels(){
    register_cpu_kernels(GlobalKernelTable());
    register_cuda_kernels(GlobalKernelTable());

}


Tensor add(Tensor& a,Tensor& b);
Tensor sub(Tensor& a,Tensor& b);
Tensor mul(Tensor& a,Tensor& b);
Tensor div(Tensor& a,Tensor& b);
Tensor sin(Tensor& a);
Tensor gemv(const Tensor& A,const Tensor& x,const Tensor& y,float alpha=1.0,float beta=0.0);
Tensor gemm(Tensor& a,Tensor& b);


//     cat,
//     stack,
//     reshape,
//     permute,
//     sigmoid,
//     transpose,
//     relu,
//     det,
//     submatrix,
//     adjugate,
//     inverse,
//     lu_decompose_crout,
//     lu_decompose_doolittle,

inline Tensor operator+(Tensor& a,Tensor& b){return add(a,b);}
inline Tensor operator-(Tensor& a,Tensor& b){return sub(a,b);}
inline Tensor operator*(Tensor& a,Tensor& b){return mul(a,b);}
inline Tensor operator/(Tensor& a,Tensor& b){return div(a,b);}




}
} // namespace EC

