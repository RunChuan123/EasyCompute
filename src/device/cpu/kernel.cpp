
#include <memory>
#include <cassert>
#include <cmath>
#include <cstring>

#include "kernel/tensor_op.hpp"
#include "tensor/tensor.hpp"
#include "tensor/meta.hpp"
#include "util/rand.h"
#include "kernel/cpu/matrix_op.hpp"
#include "backends/task_executor.hpp"

namespace EC{
namespace AT{
    


namespace CPU{

// normal initial
Tensor scalar(float value,DType dt){return Tensor{Shape{1},value,dt,DI::cpu()};}
Tensor zeros(Shape s,DType dt){return Tensor{s,0.0f,dt,DI::cpu()};}
Tensor vector(std::initializer_list<float> vl,Shape s,DType dt){    
    assert(s.rank()==1 || (s.rank() == 2 && s.dims[1]==1));
    assert(vl.size() == s.numel());
    Tensor t = zeros(s,dt);
    float* p = t.data_ptr<float>();
    std::memcpy(p,vl.begin(),s.numel()*sizeof(float));
    return t;
}//concate scaalr

Tensor ones(Shape s,DType dt){return Tensor{s,1.0f,dt,DI::cpu()};}
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
    Tensor t(std::move(s), 0.0,dt,DI::cpu());
    RandomGenerator::getInstance().filltUniformBatch<float>(t.data_ptr<float>(),t.size(),low,high);
    return t;
}
Tensor normal(Shape s, float mean , float stddev,DType dt){
//     static std::atomic<size_t> c{0};
//   std::cerr << "Tensor::normal " << ++c << "\n";
  
    if (stddev <= 0.0F) {
        throw std::runtime_error("normal: stddev must be > 0");
    }
    Tensor t(::std::move(s), 0.0F, dt,DI::cpu());
    RandomGenerator::getInstance().fillNormalBatch<float>(t.data_ptr<float>(),t.size(),mean,stddev);

    return t;
}

Tensor from_symbol(ValueId vid,Shape s,DType dt, bool req_grad){
    Tensor t{std::move(s),0.0f,dt,DI::cpu(),req_grad};
    t.set_sym(vid);
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

std::shared_ptr<Buffer> allocate_(size_t bytes,DType dtype){
    // return std::make_shared<Buffer>(bytes,dtype,DI::cpu());
    return Buffer::make(bytes,dtype,DI::cpu());
}

void fill_(void* data,float value,DType dt,size_t size){
    assert(dt == DType::f32);
    float* p = static_cast<float*>(data);
    for(size_t i=0;i<size;i++)p[i]=value;
}



Tensor add_imlp(const Tensor& a,const Tensor& b){
    Tensor out{a.getShape(),0.0f,a.getDtype(),a.getDevice()};
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
    Tensor out{a.getShape(),0.0f,a.getDtype(),a.getDevice()};
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
    Tensor out{a.getShape(),0.0f,a.getDtype(),a.getDevice()};
    auto [pa,pb,po] = get_data_ptrs_17<float>(a,b,out);
    for(size_t i=0;i < a.size();i++){
        po[i] = pa[i] * pb[i];
    }
    return out;
} 

void mul_into_impl(const Tensor& a, const Tensor& b, Tensor& out) {
    auto [pa, pb, po] = get_data_ptrs_17<float>(a, b, out);
    for (size_t i = 0; i < a.size(); ++i) {
        po[i] = pa[i] * pb[i];
    }
}

// void mul_kernel(KernelContext& ctx){
//     Tensor a = ctx.input<Tensor>(0);
//     Tensor b = ctx.input<Tensor>(1);
//     Tensor out = mul_imlp(a,b);
//     ctx.set_output(0,out);
// }
void mul_kernel(KernelContext& ctx){
    Tensor a = ctx.input<Tensor>(0);
    Tensor b = ctx.input<Tensor>(1);
    // TODO 基本检查先省略

    Tensor out{a.getShape(),0.0f,a.getDtype(),a.getDevice()};
    auto& exec = Task::AsyncTaskExecutor::get_instance();

    std::string task_name = "mul_" + a.id().to_string() + "_" + b.id().to_string();
    exec.registerTask(
        task_name,
        Task::TaskType::Compute,
        Task::Priority::Normal,
        0,
        a.getDevice(),
        {},
        [&a,&b,&out](){
           mul_into_impl(a,b,out);
           return true;
        }
    );
    exec.executeUntilIdle();
    exec.waitTask(task_name);
    ctx.set_output(0,out);

}

Tensor div_imlp(const Tensor& a,const Tensor& b){
    Tensor out{a.getShape(),0.0f,a.getDtype(),a.getDevice()};
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
    Tensor out{t.getShape(),0.0f,t.getDtype(),t.getDevice()};
    auto [pt,po] = get_data_ptrs_17<float>(t,out);
    for(size_t i=0;i<t.size();i++){
        po[i] = std::sin(pt[i]);
    }
    return out;
}
void sin__imlp(Tensor& t){
    auto [pt] = get_data_ptrs_17<float>(t);
    for(size_t i=0;i<t.size();i++){
        pt[i] = std::sin(pt[i]);
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

Tensor gemv_impl(const Tensor& A,const Tensor& x,const Tensor& y,float a,float b){
    // 先做二维矩阵向量乘，
    Shape s = A.getShape().clone().set(-1,1);
    Tensor out{s,0.0f,A.getDtype(),A.getDevice()};
    auto [pa,px,py,po] = get_data_ptrs_17<float>(A,x,y,out);
    size_t out_idx = A.getShape().get(-2);
    size_t inn_idx = A.getShape().get(-1);
    for(size_t i = 0;i<out_idx;i++){
        float accu = 0.0f;
        for(size_t j = 0;j<inn_idx;j++){
            accu += pa[i*inn_idx+j] * px[j];
        }
        po[i] = accu * a + b * py[i];
    }
    return out;
}

void gemv_kernel(KernelContext& ctx){
    Tensor A = ctx.input<Tensor>(0);
    Tensor x = ctx.input<Tensor>(1);
    Tensor y = ctx.input<Tensor>(2);
    float alpha = ctx.input<float>(3);
    float beta = ctx.input<float>(4);
    Tensor out = gemv_impl(A,x,y,alpha,beta);
    ctx.set_output(0,out);
}



}


// Tensor gemv(Tensor& A,Tensor& x,Tensor& y,float alpha,float beta){
//     auto fn = GlobalKernelTable().lookup(TOp::gemv,DType::f32,DI::cpu());
//     KernelContext k{A.device(),A.dtype(),{A,x,y,alpha,beta},{},{}};


void register_cpu_kernels(KernelTable& kt){
    kt.register_kernel(TOp::det,DType::f32,DeviceType::CPU,CPU::MatrixTOp::det_kernel);
    kt.register_kernel(TOp::inverse,DType::f32,DeviceType::CPU,CPU::MatrixTOp::inverse_kernel);
    kt.register_kernel(TOp::adjugate,DType::f32,DeviceType::CPU,CPU::MatrixTOp::adjugate_kernel);
    kt.register_kernel(TOp::ew_add,DType::f32,DeviceType::CPU,CPU::add_kernel);
    kt.register_kernel(TOp::ew_sub,DType::f32,DeviceType::CPU,CPU::sub_kernel);
    kt.register_kernel(TOp::ew_mul,DType::f32,DeviceType::CPU,CPU::mul_kernel);
    kt.register_kernel(TOp::ew_div,DType::f32,DeviceType::CPU,CPU::div_kernel);
    kt.register_kernel(TOp::ew_div,DType::f32,DeviceType::CPU,CPU::div_kernel);
    kt.register_kernel(TOp::sin,DType::f32,DeviceType::CPU,CPU::sin_kernel);
    kt.register_kernel(TOp::sin_,DType::f32,DeviceType::CPU,CPU::sin__kernel);
    kt.register_kernel(TOp::gemv,DType::f32,DeviceType::CPU,CPU::gemv_kernel);

}

}
}
