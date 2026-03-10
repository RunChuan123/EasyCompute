#include <memory>
#include <cassert>
#include <iomanip>

#include "tensor/tensor.hpp"
#include "tensor/shape.hpp"
#include "util/rand.h"
#include "util/err.hpp"
#include "tensor/device/cpu/kernel/kernel.hpp"
#include "tensor/device/nv/kernel/kernel.hpp"

namespace EC::AT{

void indent(size_t n){
    size_t spaces = 2 + n * 2;
    for(size_t k =0 ;k<spaces;k++) std::cout << " ";
}
void print_value(const Tensor& t,const std::vector<size_t>& index,size_t width = 4,size_t prec = 2){
    std::cout << std::setw(width) << std::fixed << std::setprecision(prec) << t.at(index);
}
void print_naive(const Tensor& t,size_t shape_idx,std::vector<size_t>& index,size_t width = 4,size_t prec = 2){

    size_t cur_dim = t.shape().dims[shape_idx];
    for(size_t i =0;i<cur_dim;i++){
        index[shape_idx] = i;
        if(shape_idx == t.shape().dims.size()-1){
            print_value(t,index,width,prec);
            if(i!=cur_dim - 1)std::cout << " ";
        }else{
            std::cout << "[ ";
            print_naive(t,shape_idx+1,index,width,prec);
            std::cout << " ]";

            if(i!= cur_dim-1) {
                std::cout << ",";
                std::cout << "\n";
                indent(shape_idx);
            }
        }
    }
}

void Tensor::print(size_t width,size_t prec)const{
    std::vector<size_t> index(shape().dims.size(),0);
    std::cout << "[ ";
    print_naive(*this,0,index,width,prec);
    std::cout << " ]\n";
}

    
Tensor Tensor::scalar(float value,DType dt, Device dev){
    switch (dev){
        case Device::CPU:   return CPU::scalar(value,dt);
        case Device::NV_GPU: return NV::scalar(value,dt);
        default:
        throw TensorException("Unsupported device");
    }
}
Tensor Tensor::vector(std::initializer_list<float> vl,Shape s,DType dt, Device dev){
    switch (dev){
        case Device::CPU:   return CPU::vector(vl,s,dt);
        case Device::NV_GPU: return NV::vector(vl,s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}//concate scaalr
Tensor Tensor::zeros(Shape s,DType dt, Device dev ){
    switch (dev){
        case Device::CPU:   return CPU::zeros(s,dt);
        case Device::NV_GPU: return NV::zeros(s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}

Tensor Tensor::likes(Tensor& rhs,float v){
    Tensor t{rhs.shape(),v,rhs.dtype(),rhs.device(),rhs.requires_grad()};
    return t;
}
Tensor Tensor::ones(Shape s,DType dt, Device dev){
    switch (dev){
        case Device::CPU:   return CPU::ones(s,dt);
        case Device::NV_GPU: return NV::ones(s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}
Tensor Tensor::E(Shape s,DType dt, Device dev){
    assert(s.is_square());
    switch (dev){
        case Device::CPU:   return CPU::E(s,dt);
        case Device::NV_GPU: return NV::E(s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}

Tensor Tensor::uniform(Shape s, float low, float high,DType dt, Device dev){
    switch (dev){
        case Device::CPU:   return CPU::uniform(s,low,high,dt);
        case Device::NV_GPU: return NV::uniform(s,low,high,dt);
        default:
        throw TensorException("Unsupported device");
    }
}
Tensor Tensor::normal(Shape s, float mean , float stddev,DType dt, Device dev){
    switch (dev){
        case Device::CPU:   return CPU::normal(s,mean,stddev,dt);
        case Device::NV_GPU: return NV::normal(s,mean,stddev,dt);
        default:
        throw TensorException("Unsupported device");
    }
}

Tensor Tensor::from_symbol(ValueId vid,Shape s,DType dt, Device dev, bool req_grad){

        switch (dev){
        case Device::CPU:   return CPU::from_symbol(vid, s, dt,  req_grad);
        case Device::NV_GPU: return NV::from_symbol(vid, s, dt,  req_grad);
        default:
        throw TensorException("Unsupported device");
    }
}


void Tensor::allocate_(DType dtype,Device device){
    size_t bytes = size() * size_DType(dtype);
    data_ = std::make_shared<Buffer>(bytes,dtype,device);

}
void Tensor::fill_(float value){
    switch (device_){
        case Device::CPU: CPU::fill_(data_->ptr,value,dtype_,size());return;
        case Device::NV_GPU: NV::fill_(data_->ptr,value,dtype_,size());return;
        default:
        throw TensorException("Unsupported device");
    }
}

}