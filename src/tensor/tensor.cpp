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

    size_t cur_dim = t.getShape().dims[shape_idx];
    for(size_t i =0;i<cur_dim;i++){
        index[shape_idx] = i;
        if(shape_idx == t.getShape().dims.size()-1){
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
    std::vector<size_t> index(getShape().dims.size(),0);
    std::cout << "[ ";
    print_naive(*this,0,index,width,prec);
    std::cout << " ]\n";
}

template<typename T>
T& Tensor::at(const Shape& index){
    if(dtype_ != get_dtype<T>()) 
        throw TensorException("DType mismatch");
    switch(device_.type()):
    case DeviceType::CPU :
        return data_ptr<T>()[offset(index)];
    case DeviceType::CUDA :
    // TODO
        return data_ptr<T>()[offset(index)];
}
template<typename T>
const T& Tensor::at(const Shape& index) const{
  if(dtype_ != get_dtype<T>()) 
        throw TensorException("DType mismatch");
    switch(device_.type()):
    case DeviceType::CPU :
        return data_ptr<T>()[offset(index)];
    case DeviceType::CUDA :
    // TODO
        return data_ptr<T>()[offset(index)];
}

Tensor Tensor::view(Shape s) const{
    if(s.numel() != numel()) throw TensorException("shape mismatch");
    Tensor out;
    
    out.data_ = data_;
    out.shape_ = s;
    out.device_ = device_;
    out.dtype_ = dtype_;
    return out;
}

size_t Tensor::offset(const Shape& s)const{
    if(s.rank() != rank()){throw TensorException("offset: index shape mismatch");}
    size_t off=0;
    std::vector<size_t> strides = make_strides(shape_);
    for(size_t i=0;i<s.rank();i++){
        size_t idx = s.dims[i];
        size_t dim = shape_.dims[i];
        if(idx < 0 || idx >= dim){throw TensorException("offset: index out of range");}
        off += idx * strides[i];
    }
    return off;
}
Tensor& Tensor::reshape(Shape s){
    if(s.numel() != numel())throw TensorException("reshape failed: mismatch shape!");
    shape_ = s;
    return *this;
}
template<typename T>
T& Tensor::at(const Shape& index) { 
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, double>,"Unsupported data type for Tensor::at");
    if (dtype_ != get_dtype<T>()) { 
        throw TensorException("Type mismatch in Tensor::at!");
    }
    return data_ptr<T>()[offset(index)];
}
template<typename T>
const T& Tensor::at(const Shape& index) const {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, double>,"Unsupported data type for Tensor::at");
    if (dtype_ != get_dtype<T>()) {
        throw TensorException("Type mismatch in Tensor::at!");
    }
    return data_ptr<const T>()[offset(index)];
}
Tensor Tensor::scalar(float value,DType dt, Device dev){
    switch (dev.type()){
        case DeviceType::CPU:   return CPU::scalar(value,dt);
        case DeviceType::CUDA: return NV::scalar(value,dt);
        default:
        throw TensorException("Unsupported device");
    }
}
Tensor Tensor::vector(std::initializer_list<float> vl,Shape s,DType dt, Device dev){
    switch (dev.type()){
        case DeviceType::CPU   return CPU::vector(vl,s,dt);
        case DeviceType::CUDA: return NV::vector(vl,s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}//concate scaalr
Tensor Tensor::zeros(Shape s,DType dt, Device dev ){
    switch (dev.type()){
        case DeviceType::CPU:   return CPU::zeros(s,dt);
        case DeviceType::CUDA: return NV::zeros(s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}

Tensor Tensor::likes(Tensor& rhs,float v){
    Tensor t{rhs.getShape(),v,rhs.dtype(),rhs.device(),rhs.requires_grad()};
    return t;
}
Tensor Tensor::ones(Shape s,DType dt, Device dev){
    switch (dev.type()){
        case DeviceType::CPU:   return CPU::ones(s,dt);
        case DeviceType::CUDA: return NV::ones(s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}
Tensor Tensor::E(Shape s,DType dt, Device dev){
    assert(s.is_square());
    switch (dev.type()){
        case DeviceType::CPU:   return CPU::E(s,dt);
        case DeviceType::CUDA: return NV::E(s,dt);
        default:
        throw TensorException("Unsupported device");
    }
}

Tensor Tensor::uniform(Shape s, float low, float high,DType dt, Device dev){
    switch (dev.type()){
        case DeviceType::CPU:   return CPU::uniform(s,low,high,dt);
        case DeviceType::CUDA: return NV::uniform(s,low,high,dt);
        default:
        throw TensorException("Unsupported device");
    }
}
Tensor Tensor::normal(Shape s, float mean , float stddev,DType dt, Device dev){
    switch (dev.type()){
        case DeviceType::CPU:   return CPU::normal(s,mean,stddev,dt);
        case DeviceType::CUDA: return NV::normal(s,mean,stddev,dt);
        default:
        throw TensorException("Unsupported device");
    }
}

// Tensor Tensor::from_symbol(ValueId vid,Shape s,DType dt, Device dev, bool req_grad){

//         switch (dev.type()){
//         case DeviceType::CPU:   return CPU::from_symbol(vid, s, dt,  req_grad);
//         case DeviceType::CUDA: return NV::from_symbol(vid, s, dt,  req_grad);
//         default:
//         throw TensorException("Unsupported device");
//     }
// }


void Tensor::allocate_(){

    size_t bytes = size() * size_dtype(dtype_);
    data_ = std::make_shared<Buffer>(bytes,dtype_,device);

}
void Tensor::fill_(float value){
    switch (device_.type()){
        case DeviceType::CPU: CPU::fill_(data_->ptr,value,dtype_,size());return;
        case DeviceType::CUDA: NV::fill_(data_->ptr,value,dtype_,size());return;
        default:
        throw TensorException("Unsupported device");
    }
}

}