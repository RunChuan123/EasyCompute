

#include "tensor/tensor.hpp"
#include "graph/trace/api.hpp"

namespace EC{
namespace AT{




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
    auto fn = GlobalKernelTable().lookup(TOp::ew_add,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor sub(Tensor& a,Tensor& b){
    auto fn = GlobalKernelTable().lookup(TOp::ew_sub,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor mul(Tensor& a,Tensor& b){
    
    auto fn = GlobalKernelTable().lookup(TOp::ew_mul,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    // c.print();
    return c;
}
Tensor div(Tensor& a,Tensor& b){
    auto fn = GlobalKernelTable().lookup(TOp::ew_div,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

Tensor sin(Tensor& a){
    auto fn = GlobalKernelTable().lookup(TOp::sin,DType::f32,Device::CPU);
    KernelContext k{a.device(),a.dtype(),{a},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

// f(x) = alpha * Ax + beta * y
Tensor gemv(const Tensor& A,const Tensor& x,const Tensor& y ,float alpha,float beta){

    if(A.shape().get(-1) != x.shape().get(0)){
        throw FunctionException("gemv: a b shape mismatch");
    }
    auto fn = GlobalKernelTable().lookup(TOp::gemv,DType::f32,Device::CPU);
    KernelContext k{A.device(),A.dtype(),{A,x,y,alpha,beta},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

Tensor gemm(Tensor& a,Tensor& b){

}


}
}