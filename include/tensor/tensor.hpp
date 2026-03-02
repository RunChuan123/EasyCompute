#pragma once

#include <initializer_list>
#include <memory>
#include <utility>

#include "buffer.hpp"
#include "shape.hpp"
#include "../util/err.hpp"
// #include "../util/rand.h"

namespace EC{
namespace AT{

struct Tensor{

    std::shared_ptr<Buffer> data_;
    Shape shape_ ;
    DType dtype_=DType::f32;
    Device device_=Device::CPU;

    Tensor()=default;
    explicit Tensor(Shape s, float value = 0.0f, DType dtype=DType::f32,Device dev = Device::CPU)
    : shape_(std::move(s)), dtype_(dtype), device_(dev) {
        allocate_(dtype_,device_);
        fill_(value);
    }
    
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    // 访问器
    const Shape& shape() const { return shape_; }
    size_t numel() const { return shape_.numel(); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    template<typename T>
    inline T* data_ptr(){return static_cast<T*>(data_->data_ptr());}

    template<typename T>
    inline const T* data_ptr()const{return static_cast<const T*>(data_->data_ptr());}

    std::vector<size_t> strides()const{return make_strides(shape_);}
    // no data
    inline bool empty()const{return !data_;}
    inline size_t size()const{return shape_.numel();}
    inline size_t rank()const{return shape_.rank();}

    inline bool is_scalar() const { return shape_.is_scalar(); }
    inline bool is_vector() const { return shape_.is_vector(); }
    inline bool is_matrix() const { return shape_.is_matrix(); }
    inline bool is_square() const { return shape_.is_square(); }
    void print(size_t width = 4,size_t prec = 2)const;
    static Tensor scalar(float value,DType dt=DType::f32, Device dev = Device::CPU);
    static Tensor vector(std::initializer_list<float> vl,Shape s,DType dt=DType::f32, Device dev = Device::CPU);
    static Tensor zeros(Shape s,DType dt=DType::f32, Device dev = Device::CPU);
    static Tensor ones(Shape s,DType dt=DType::f32, Device dev = Device::CPU);
    static Tensor E(Shape s,DType dt=DType::f32, Device dev = Device::CPU);
    static Tensor uniform(Shape s, float low = 0.0F, float high = 1.0F,DType dt=DType::f32, Device dev = Device::CPU);
    static Tensor normal(Shape s, float mean = 0.0F, float stddev = 1.0F,DType dt=DType::f32, Device dev = Device::CPU);

    /** 
     * same pointer to the datamemory
     */
    inline Tensor view() const{
        Tensor out;
        out.data_ = data_;
        out.shape_ = shape_;
        out.device_ = device_;
        out.dtype_ = dtype_;
        return out;
    }

    inline size_t offset(const Shape& s)const{
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

    template<typename T>
    T& at(const Shape& index) { 
        // 编译期校验   
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, double>,"Unsupported data type for Tensor::at");
        // 运行时类型校验
        if (dtype_ != get_dtype<T>()) { 
            throw TensorException("Type mismatch in Tensor::at!");
        }
        return data_ptr<T>()[offset(index)];
    }
    template<typename T>
    const T& at(const Shape& index) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, double>,"Unsupported data type for Tensor::at");
        if (dtype_ != get_dtype<T>()) {
            throw TensorException("Type mismatch in Tensor::at!");
        }
        return data_ptr<const T>()[offset(index)];
    }
    // 为float类型提供重载
    float& at(const Shape& index) {return at<float>(index);}
    const float& at(const Shape& index) const {return at<float>(index);}

private:
    void allocate_(DType dtype,Device device);
    void fill_(float value);
};
}
} // namespace EC
