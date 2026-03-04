
#include <memory>
#include <cassert>

#include "tensor/tensor.hpp"
// #include "tensor/device/cpu/kernel/kernel_naive.hpp"
#include "util/rand.h"

namespace EC::AT{
namespace NV{


Tensor scalar(float value,DType dt){

}
Tensor vector(std::initializer_list<float> vl,Shape s,DType dt){

}
Tensor zeros(Shape s,DType dt){}
Tensor ones(Shape s,DType dt){}
Tensor E(Shape s,DType dt){}

Tensor uniform(Shape s, float low, float high,DType dt){}
Tensor normal(Shape s, float mean , float stddev,DType dt){}
Tensor from_symbol(int32_t vid,Shape s,DType dt, bool req_grad){
    Tensor t{std::move(s),0.0f,dt,Device::NV_GPU,req_grad};
    t.sym_ = vid;
    return t;

}


float det(){}
Tensor submatrix(size_t ex_row,size_t ex_col){}
Tensor adjugate(){}
Tensor inverse(){}
std::pair<Tensor,Tensor> lu_decompose_doolittle(const Tensor& t){}
std::pair<Tensor,Tensor> lu_decompose_crout(const Tensor& t){}


std::shared_ptr<Buffer> allocate_(size_t bytes,DType dtype){}
void fill_(void* data,float value,DType dt,size_t size){}


}

void register_cuda_kernels(){

}
}