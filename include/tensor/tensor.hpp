#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <optional>
#include <atomic>
#include <thread>

#include "buffer.hpp"
#include "shape.hpp"
#include "../util/err.hpp"

namespace EC::AT{

struct TensorId{
    // uint64_t pid;
    // uint64_t tid;
    uint64_t tensor_id = -1;
    uint64_t dev_id=-1;
    bool operator==(const TensorId& rhs)const{
        return 
            // pid == rhs.pid && 
            // tid == rhs.tid &&
            dev_id == rhs.dev_id &&
            tensor_id == rhs.tensor_id;
    }

    std::string to_string() const{
        // return std::to_string(pid) + "_" + std::to_string(tid) + "_" + std::to_string(dev_id) + "_" + std::to_string(local_id);
        return std::to_string(dev_id) + "_" + std::to_string(tensor_id);
    
    }
};

thread_local std::atomic<uint64_t> t_local_tensor_id_counter(0);

using ValueId = int32_t;

struct Tensor{
public:

    Tensor()=default;
    Tensor(Shape s, float value = 0.0f, DType dtype=DType::f32,
                    Device dev = Device::cpu(),bool requires_grad = false)
    : id_(TensorId{}),shape_(std::move(s)), dtype_(dtype), device_(dev),requires_grad_(requires_grad) {
        id_.tensor_id = t_local_tensor_id_counter.fetch_add(1, std::memory_order_relaxed);
        allocate_();
        fill_(value);
    }
    
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    Shape getShape() const { return shape_; }
    DType getDtype() const { return dtype_; }
    Device getDevice() const { return device_; }
    size_t numel() const { return shape_.numel(); }
    TensorId id()const{return id_;}
    bool requires_grad() const { return requires_grad_; }
    bool is_symbolic() const {return sym_.has_value();}
    int32_t sym() const {return sym_.value(); }
    inline size_t offset_bytes() const { return data_->offset_bytes; }
    bool is_contiguous()const{return data_->is_contiguous;}
    std::vector<size_t> strides()const{return make_strides(shape_);}
    template<typename T>
    T& operator[](size_t index);
    template<typename T>
    const T& operator[](size_t index) const;

    template<typename T>
    inline T* data_ptr(){return static_cast<T*>(data_->data_ptr());}
    template<typename T>
    inline const T* data_ptr()const{return static_cast<const T*>(data_->data_ptr());}
    inline const Buffer* buffer_ptr()const{return data_.get();}
    
    inline bool empty()const{return !data_;}
    inline size_t size()const{return shape_.numel();}
    inline size_t rank()const{return shape_.rank();}
    
    inline bool is_scalar() const { return shape_.is_scalar(); }
    inline bool is_vector() const { return shape_.is_vector(); }
    inline bool is_matrix() const { return shape_.is_matrix(); }
    inline bool is_square() const { return shape_.is_square(); }

    void print(size_t width = 4,size_t prec = 2)const;
    
    static Tensor scalar(float value,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor vector(std::initializer_list<float> vl,Shape s,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor zeros(Shape s,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor ones(Shape s,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor E(Shape s,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor uniform(Shape s, float low = 0.0F, float high = 1.0F,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor normal(Shape s, float mean = 0.0F, float stddev = 1.0F,DType dt=DType::f32, Device dev = Device::cpu());
    static Tensor likes(Tensor& rhs,float v=0.0f);
    static Tensor Empty(Shape s,DType dt=DType::f32, Device dev = Device::cpu());
    // static Tensor from_symbol(ValueId vid,Shape s,DType dt=DType::f32, Device dev = Device::cpu(), bool req_grad=true);// ?
    // 上三角，下三角，等等

    inline Tensor view(Shape s) const;

    Tensor& reshape(Shape s);
    void set_requires_grad(bool i) { requires_grad_ = i; }
    Tensor ravel();
    Tensor flatten();
    Tensor& unsqueeze(size_t dim);
    Tensor& squeeze(size_t dim);
    Tensor& transpose(size_t dim0,size_t dim1);
    Tensor& permute(std::vector<size_t> dims);

    size_t offset(const Shape& s)const;

    template<typename T=float>
    T& at(const Shape& index);
    template<typename T = float>
    const T& at(const Shape& index) const;

    bool clear();
    Tensor clone();
    void copy_(Tensor& src);
    std::vector<Tensor> split(size_t split_size,size_t dim);
    std::vector<Tensor> chunk(size_t chunks,size_t dim);
    void to(Device dev);
    void to(DType dt);


private:

    TensorId id_=TensorId{};
    std::shared_ptr<Buffer> data_;
    Shape shape_ ;
    DType dtype_=DType::f32;
    Device device_=Device::cpu();
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    // 计算图使用
    std::optional<ValueId> sym_;
    void allocate_();
    void fill_(float value=0.0f);

};

}