#pragma once
#include <memory>
#include <cassert>

#include "tensor/tensor.hpp"

namespace EC::AT{

namespace CPU{

    
Tensor scalar(float value,DType dt);
Tensor vector(std::initializer_list<float> vl,Shape s,DType dt);
Tensor zeros(Shape s,DType dt);
Tensor ones(Shape s,DType dt);
Tensor E(Shape s,DType dt);

Tensor uniform(Shape s, float low, float high,DType dt);
Tensor normal(Shape s, float mean , float stddev,DType dt);
Tensor from_symbol(int32_t vid,Shape s,DType dt, bool req_grad);

std::shared_ptr<Buffer> allocate_(size_t bytes,DType dtype);
void fill_(void* data,float value,DType dt,size_t size);

}
}