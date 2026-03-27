
// #include "backends/device_enabled.hpp"

// #ifdef EC_ENABLE_CUDA


// #include <memory>
// #include <stdio.h>
// #include <cassert>
// #include <device_launch_parameters.h>


// #include "tensor/api.hpp"
// #include "kernel/tensor_op.hpp"

// #include "util/rand.h"



// namespace EC::AT{
// namespace NV{


// Tensor scalar(float value,DType dt){

// }
// Tensor vector(std::initializer_list<float> vl,Shape s,DType dt){

// }
// Tensor zeros(Shape s,DType dt){}
// Tensor ones(Shape s,DType dt){}
// Tensor E(Shape s,DType dt){}

// Tensor uniform(Shape s, float low, float high,DType dt){}
// Tensor normal(Shape s, float mean , float stddev,DType dt){}
// Tensor from_symbol(int32_t vid,Shape s,DType dt, bool req_grad){
//     Tensor t{std::move(s),0.0f,dt,DI::cuda(),req_grad};
//     t.set_sym(vid);
//     return t;

// }


// float det(){}
// Tensor submatrix(size_t ex_row,size_t ex_col){}
// Tensor adjugate(){}
// Tensor inverse(){}
// std::pair<Tensor,Tensor> lu_decompose_doolittle(const Tensor& t){}
// std::pair<Tensor,Tensor> lu_decompose_crout(const Tensor& t){}


// std::shared_ptr<Buffer> allocate_(size_t bytes,DType dtype){}
// void fill_(void* data,float value,DType dt,size_t size){}
// }


// }

// #endif