

#include "tensor/tensor_op.hpp"
#include "tensor/tensor.hpp"
#include "graph/trace_api.hpp"

namespace EC{
namespace AT{

KernelFunc kernel_table[NumDevice_][NumDType_][NumOp_]={};


Tensor add(Tensor& a,Tensor& b){
    // Trace
    if(auto* tr = Tr::current_tracer(); tr && tr->is_tracing()){
        auto va = tr->value_of(a);
        auto vb = tr->value_of(b);
        bool req = a.requires_grad() || b.requires_grad();
        auto vout = tr->make_value_like(a,req);
        tr->record_op(TOp::ew_add,{va,vb},{vout});
        return Tensor::from_symbol(vout,a.shape(),a.dtype(),a.device(),req);
    }
    // Eager;
    auto fn = lookup(TOp::ew_add,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor sub(Tensor& a,Tensor& b){
    auto fn = lookup(TOp::ew_sub,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor mul(Tensor& a,Tensor& b){
    
    auto fn = lookup(TOp::ew_mul,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    // c.print();
    return c;
}
Tensor div(Tensor& a,Tensor& b){
    auto fn = lookup(TOp::ew_div,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

Tensor sin(Tensor& a){
    auto fn = lookup(TOp::sin,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

}
}