

#include "tensor/tensor.hpp"
#include "graph/trace/api.hpp"

namespace EC{
namespace AT{


Tensor add(Tensor& a,Tensor& b){
    // Trace
    if(auto* tr = Tr::current_tracer(); tr && Tr::is_tracing()){
        auto va = tr->resolve_tensor(a);
        auto vb = tr->resolve_tensor(b);
        bool req = a.requires_grad() || b.requires_grad();

        auto vout = tr->graph().new_value(a.getMeta(),Gr::ValueKind::Temp,"add_out");
        // tr->record_op(TOp::ew_add,{va,vb},{vout});
        tr->graph().new_node(TOp::ew_add,{va,vb},{vout},{},"add_op");
        return Tensor::from_symbol(vout,a.getShape(),a.getDtype(),a.getDevice(),req);
    }
    // Eager;
    auto fn = GlobalKernelTable().lookup(TOp::ew_add,a.getDtype(),a.getDevice().type());
    KernelContext k{a.getDevice(),a.getDtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor sub(Tensor& a,Tensor& b){
    // Trace
    if(auto* tr = Tr::current_tracer(); tr && Tr::is_tracing()){
        auto va = tr->resolve_tensor(a);
        auto vb = tr->resolve_tensor(b);
        bool req = a.requires_grad() || b.requires_grad();

        auto vout = tr->graph().new_value(a.getMeta(),Gr::ValueKind::Temp,"sub_out");
        tr->graph().new_node(TOp::ew_sub,{va,vb},{vout},{},"add_op");
        return Tensor::from_symbol(vout,a.getShape(),a.getDtype(),a.getDevice(),req);
    }
    // Eager;
    auto fn = GlobalKernelTable().lookup(TOp::ew_sub,a.getDtype(),a.getDevice().type());
    KernelContext k{a.getDevice(),a.getDtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor mul(Tensor& a,Tensor& b){
    // Trace
    if(auto* tr = Tr::current_tracer(); tr && Tr::is_tracing()){
        auto va = tr->resolve_tensor(a);
        auto vb = tr->resolve_tensor(b);
        bool req = a.requires_grad() || b.requires_grad();
        auto vout = tr->graph().new_value(a.getMeta(),Gr::ValueKind::Temp,"mul_out");

        tr->graph().new_node(TOp::ew_mul,{va,vb},{vout},{},"mul_op");
        return Tensor::from_symbol(vout,a.getShape(),a.getDtype(),a.getDevice(),req);
    }
    // Eager;
    auto fn = GlobalKernelTable().lookup(TOp::ew_mul,a.getDtype(),a.getDevice().type());
    KernelContext k{a.getDevice(),a.getDtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}
Tensor div(Tensor& a,Tensor& b){
    // Trace
    if(auto* tr = Tr::current_tracer(); tr && Tr::is_tracing()){
        auto va = tr->resolve_tensor(a);
        auto vb = tr->resolve_tensor(b);
        bool req = a.requires_grad() || b.requires_grad();
        auto vout = tr->graph().new_value(a.getMeta(),Gr::ValueKind::Temp,"div_out");

        tr->graph().new_node(TOp::ew_div,{va,vb},{vout},{},"div_op");
        return Tensor::from_symbol(vout,a.getShape(),a.getDtype(),a.getDevice(),req);
    }
    // Eager;
    auto fn = GlobalKernelTable().lookup(TOp::ew_div,a.getDtype(),a.getDevice().type());
    KernelContext k{a.getDevice(),a.getDtype(),{a,b},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

Tensor sin(Tensor& a){
    // Trace
    if(auto* tr = Tr::current_tracer(); tr && Tr::is_tracing()){
        auto va = tr->resolve_tensor(a);
        bool req = a.requires_grad();
        auto vout = tr->graph().new_value(a.getMeta(),Gr::ValueKind::Temp,"sin_out");

        tr->graph().new_node(TOp::ew_div,{va},{vout},{},"sin_op");
        return Tensor::from_symbol(vout,a.getShape(),a.getDtype(),a.getDevice(),req);
    }
    // Eager;
    auto fn = GlobalKernelTable().lookup(TOp::sin,a.getDtype(),a.getDevice().type());
    KernelContext k{a.getDevice(),a.getDtype(),{a},{},{}};
    fn(k);
    Tensor c = k.output<Tensor>(0);
    return c;
}

// // f(x) = alpha * Ax + beta * y
// Tensor gemv(const Tensor& A,const Tensor& x,const Tensor& y ,float alpha,float beta){

//     if(A.shape().get(-1) != x.shape().get(0)){
//         throw FunctionException("gemv: a b shape mismatch");
//     }
//     auto fn = GlobalKernelTable().lookup(TOp::gemv,DType::f32,Device::CPU);
//     KernelContext k{A.device(),A.dtype(),{A,x,y,alpha,beta},{},{}};
//     fn(k);
//     Tensor c = k.output<Tensor>(0);
//     return c;
// }

// Tensor gemm(Tensor& a,Tensor& b){

// }


}
}