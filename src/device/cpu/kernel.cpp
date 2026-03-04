
#include <memory>
#include <cassert>
#include <cmath>

#include "tensor/tensor_op.hpp"
#include "tensor/tensor.hpp"
#include "tensor/meta.hpp"
#include "tensor/device/cpu/kernel/matrix_op.hpp"
#include "util/rand.h"

namespace EC{
namespace AT{
    


namespace CPU{

// normal initial
Tensor scalar(float value,DType dt){return Tensor{Shape{1},value,dt,Device::CPU};}
Tensor zeros(Shape s,DType dt){return Tensor{s,0.0f,dt,Device::CPU};}
Tensor vector(std::initializer_list<float> vl,Shape s,DType dt){
    assert(s.rank()==1 || (s.rank() == 2 && s.dims[1]==1));
    assert(vl.size() == s.numel());
    Tensor t = zeros(s,dt);
    float* p = t.data_ptr<float>();
    std::memcpy(p,vl.begin(),s.numel()*sizeof(float));
    return t;
}//concate scaalr

Tensor ones(Shape s,DType dt){return Tensor{s,1.0f,dt,Device::CPU};}
Tensor E(Shape s,DType dt){
    assert(s.is_square());
    Tensor t = zeros(s,dt);
    for(size_t i = 0;i<s.dims[0];i++){
        t.at({i,i}) = 1.0f;
    }
    return t;
}

Tensor uniform(Shape s, float low, float high,DType dt){
    if (low > high) {
        throw TensorException("uniform: low must be <= high");
    }
    Tensor t(std::move(s), 0.0,dt,Device::CPU);
    RandomGenerator::getInstance().filltUniformBatch<float>(t.data_ptr<float>(),t.size(),low,high);
    return t;
}
Tensor normal(Shape s, float mean , float stddev,DType dt){
//     static std::atomic<size_t> c{0};
//   std::cerr << "Tensor::normal " << ++c << "\n";
  
    if (stddev <= 0.0F) {
        throw std::runtime_error("normal: stddev must be > 0");
    }
    Tensor t(::std::move(s), 0.0F, dt,Device::CPU);
    RandomGenerator::getInstance().fillNormalBatch<float>(t.data_ptr<float>(),t.size(),mean,stddev);

    return t;
}

Tensor from_symbol(ValueId vid,Shape s,DType dt, bool req_grad){
    Tensor t{std::move(s),0.0f,dt,Device::CPU,req_grad};
    t.sym_ = vid;
    return t;

}

namespace MatrixTOp{

// matrix operator
void det_kernel(KernelContext& ctx){
    Tensor t = ctx.input<Tensor>(0);
    float determinant = MatrixTOp::det(t);
    ctx.set_output(0,determinant);
    return;
}
void submatrix_kernel(KernelContext& ctx) {
    Tensor t = ctx.input<Tensor>(0);
    Tensor sub = MatrixTOp::submatrix(t,ctx.attr<size_t>(0),ctx.attr<size_t>(1));
    ctx.set_output(0,sub);
    return;
}

void adjugate_kernel(KernelContext& ctx){
    Tensor t = ctx.input<Tensor>(0);
    Tensor adj = MatrixTOp::adjugate(t);
    ctx.set_output(0,adj);
    return;
}

void inverse_kernel(KernelContext& ctx){
    Tensor t = ctx.input<Tensor>(0);
    Tensor inv = MatrixTOp::inverse(t);
    ctx.set_output(0,inv);
    return;
}

}

std::shared_ptr<Buffer> allocate_(size_t bytes,DType dtype){return std::make_shared<Buffer>(bytes,dtype,Device::CPU);}
void fill_(void* data,float value,DType dt,size_t size){
    // assert(dt == DType::f32);
    float* p = static_cast<float*>(data);
    for(size_t i=0;i<size;i++)p[i]=value;
}



Tensor add_imlp(const Tensor& a,const Tensor& b){
    Tensor out{a.shape(),0.0f,a.dtype(),a.device()};
    auto [pa,pb,po] = get_data_ptrs_17<float>(a,b,out);
    for(size_t i=0;i < a.size();i++){
        po[i] = pa[i] + pb[i];
    }
    return out;
} 

void add_kernel(KernelContext& ctx){
    Tensor a = ctx.input<Tensor>(0);
    Tensor b = ctx.input<Tensor>(1);
    Tensor out = add_imlp(a,b);
    ctx.set_output(0,out);
}
Tensor sub_imlp(const Tensor& a,const Tensor& b){
    Tensor out{a.shape(),0.0f,a.dtype(),a.device()};
    auto [pa,pb,po] = get_data_ptrs_17<float>(a,b,out);
    for(size_t i=0;i < a.size();i++){
        po[i] = pa[i] - pb[i];
    }
    return out;
} 

void sub_kernel(KernelContext& ctx){
    Tensor a = ctx.input<Tensor>(0);
    Tensor b = ctx.input<Tensor>(1);
    Tensor out = sub_imlp(a,b);
    ctx.set_output(0,out);
}
Tensor mul_imlp(const Tensor& a,const Tensor& b){
    Tensor out{a.shape(),0.0f,a.dtype(),a.device()};
    auto [pa,pb,po] = get_data_ptrs_17<float>(a,b,out);
    for(size_t i=0;i < a.size();i++){
        po[i] = pa[i] * pb[i];
    }
    return out;
} 

void mul_kernel(KernelContext& ctx){
    Tensor a = ctx.input<Tensor>(0);
    Tensor b = ctx.input<Tensor>(1);
    Tensor out = mul_imlp(a,b);
    ctx.set_output(0,out);
}
Tensor div_imlp(const Tensor& a,const Tensor& b){
    Tensor out{a.shape(),0.0f,a.dtype(),a.device()};
    auto [pa,pb,po] = get_data_ptrs_17<float>(a,b,out);
    for(size_t i=0;i < a.size();i++){
        po[i] = pa[i] / pb[i];
    }
    return out;
} 

void div_kernel(KernelContext& ctx){
    Tensor a = ctx.input<Tensor>(0);
    Tensor b = ctx.input<Tensor>(1);
    Tensor out = add_imlp(a,b);
    ctx.set_output(0,out);
}

Tensor sin_imlp(const Tensor& t){
    Tensor out{t.shape(),0.0f,t.dtype(),t.device()};
    auto [pt,po] = get_data_ptrs_17<float>(t,out);
    for(size_t i=0;i<t.size();i++){
        po[i] = std::sinf(pt[i]);
    }
    return out;
}

void sin__imlp(Tensor& t){
    auto [pt] = get_data_ptrs_17<float>(t);
    for(size_t i=0;i<t.size();i++){
        pt[i] = std::sinf(pt[i]);
    }
}

void sin_kernel(KernelContext& ctx){
    Tensor t = ctx.input<Tensor>(0);
    Tensor out = sin_imlp(t);
    ctx.set_output(0,out);
}

void sin__kernel(KernelContext& ctx){
    Tensor t = ctx.input<Tensor>(0);
    sin__imlp(t);
}


}



void register_cpu_kernels(){
    register_kernel(TOp::det,DType::f32,Device::CPU,CPU::MatrixTOp::det_kernel);
    register_kernel(TOp::inverse,DType::f32,Device::CPU,CPU::MatrixTOp::inverse_kernel);
    register_kernel(TOp::adjugate,DType::f32,Device::CPU,CPU::MatrixTOp::adjugate_kernel);
    register_kernel(TOp::ew_add,DType::f32,Device::CPU,CPU::add_kernel);
    register_kernel(TOp::ew_sub,DType::f32,Device::CPU,CPU::sub_kernel);
    register_kernel(TOp::ew_mul,DType::f32,Device::CPU,CPU::mul_kernel);
    register_kernel(TOp::ew_div,DType::f32,Device::CPU,CPU::div_kernel);
    register_kernel(TOp::ew_div,DType::f32,Device::CPU,CPU::div_kernel);
    register_kernel(TOp::sin,DType::f32,Device::CPU,CPU::sin_kernel);
    register_kernel(TOp::sin_,DType::f32,Device::CPU,CPU::sin__kernel);

}

}
}
