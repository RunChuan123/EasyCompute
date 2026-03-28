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
#include "util/logger.hpp"

namespace EC::AT{



inline std::atomic<uint64_t> g_tensor_id_counter{1}; // 0 留给 invalid 也行
using ValueId = int32_t;


struct TensorId {
    uint64_t value = invalid();
    static constexpr uint64_t invalid() {return std::numeric_limits<uint64_t>::max();}
    bool is_valid() const {return value != invalid();}
    bool operator==(const TensorId& rhs) const {return value == rhs.value;}
    bool operator!=(const TensorId& rhs) const {return !(*this == rhs);}
    std::string to_string() const {return std::to_string(value);}
};

inline TensorId make_tensor_id() {
    TensorId id;
    id.value = g_tensor_id_counter.fetch_add(1, std::memory_order_relaxed);
    return id;
}

struct TensorMeta{

    TensorMeta()=default;
    TensorMeta(Shape s,DType dt,DI dev,bool req_grad):
        shape(s),dtype(dt),device(dev),is_contiguous(true),requires_grad(req_grad){}
    
        Shape shape;
    DType dtype;
    DI device;
    bool is_contiguous{true};
    bool requires_grad{false};
    size_t numel() const {return shape.numel();}
    size_t itemsize() const {return size_dtype(dtype);}
    size_t nbytes() const {return numel() * itemsize();}
    
    static TensorMeta make_meta(Shape s,DType dtype,DI dev,bool req_grad){
        TensorMeta t(s,dtype,dev,req_grad);
        return t;
    }
};


struct Tensor : std::enable_shared_from_this<Tensor> {
public:

    Tensor()=default;

    Tensor(Shape s, float value = 0.0f, DType dtype=DType::f32,
       DI dev = DI::cpu(), bool requires_grad = false)
    : id_(make_tensor_id()) {
    meta.shape = std::move(s);
    meta.dtype = dtype;
    meta.device = dev;
    meta.is_contiguous = true;
    meta.requires_grad = requires_grad;
    allocate_();
    fill_(value);
}
    
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() =default;

    Shape getShape() const { return meta.shape; }
    DType getDtype() const { return meta.dtype; }
    DI getDevice() const { return meta.device; }
    std::string getID()const{return id_.to_string();}
    TensorMeta getMeta()const {return meta;}
    size_t numel() const { return meta.numel(); }
    bool requires_grad() const { return meta.requires_grad; }
    TensorId id()const{return id_;}
    bool is_symbolic() const {return sym_.has_value();} // graph value
    int32_t sym() const {return sym_.value(); }
    void set_sym(ValueId id){sym_ = id;}
    inline size_t offset_bytes() const { return data_? data_->offset_bytes : 0; }
    bool is_contiguous()const{return data_ ? data_->is_contiguous : true;}
    std::vector<size_t> strides()const{return make_strides(meta.shape);}
    inline void set_requires_grad(bool i) { meta.requires_grad = i; }

    template<typename T>
    T& operator[](size_t index) {return at(Shape({index}));}
    template<typename T>
    const T& operator[](size_t index) const {return at(Shape({index}));}

    inline const Buffer* buffer_ptr() const { return data_.get(); }
    inline std::shared_ptr<Buffer> buffer() const { return data_; }
    
    inline bool empty()const{return !data_;}
    inline size_t size()const{return meta.shape.numel();}
    inline size_t rank()const{return meta.shape.rank();}
    
    inline bool is_scalar() const { return meta.shape.is_scalar(); }
    inline bool is_vector() const { return meta.shape.is_vector(); }
    inline bool is_matrix() const { return meta.shape.is_matrix(); }
    inline bool is_square() const { return meta.shape.is_square(); }

    void print(size_t width = 4,size_t prec = 2)const;
    
    static Tensor scalar(float value,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor vector(std::initializer_list<float> vl,Shape s,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor zeros(Shape s,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor ones(Shape s,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor E(Shape s,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor uniform(Shape s, float low = 0.0F, float high = 1.0F,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor normal(Shape s, float mean = 0.0F, float stddev = 1.0F,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor likes(Tensor& rhs,float v=0.0f);
    // static Tensor Empty(Shape s,DType dt=DType::f32, DI dev = DI::cpu());
    static Tensor from_symbol(ValueId vid,Shape s,DType dt=DType::f32, DI dev = DI::cpu(), bool req_grad=false);// ?

    inline Tensor view(Shape s) const;

    Tensor& reshape(Shape s);
    
    Tensor ravel();
    Tensor flatten();
    Tensor& unsqueeze(size_t dim);
    Tensor& squeeze(size_t dim);
    Tensor& transpose(size_t dim0,size_t dim1);
    Tensor& permute(std::vector<size_t> dims);

    size_t offset(const Shape& s)const;

    template<typename T>
    T& _at(const Shape& index) {
        if (get_dtype<T>() != meta.dtype) {
            throw TypeException("Tensor::at<T> type mismatch with tensor dtype");
        }
        const size_t off = offset(index);

        if (meta.device.type() == DeviceType::CPU) {   
            return data_ptr<T>()[off];
        }

        ensure_host_mirror_();
        data_->mark_host_dirty(); // 非 const 访问按可能写入处理
        return static_cast<T*>(data_->host_data_ptr())[off];
    }

    template<typename T>
    const T& _at(const Shape& index) const {
        if (get_dtype<T>() != meta.dtype) {
            throw TypeException("Tensor::at<T> type mismatch with tensor dtype");
        }
        const size_t off = offset(index);

        if (meta.device.type() == DeviceType::CPU) {
            return data_ptr<T>()[off];
        }

        ensure_host_mirror_();
        return static_cast<const T*>(data_->host_data_ptr())[off];
    }
    decltype(auto) at(const Shape& index);
    
    decltype(auto) at(const Shape& index) const;

    template<typename F>
    decltype(auto) at(const Shape& index, F&& f) {
        return dispatch_dtype(getDtype(), [&]<typename T>() {
            f(data_ptr<T>()[offset(index)]);
        });
    }

    bool clear();
    Tensor clone();
    void copy_(Tensor& src);
    std::vector<Tensor> split(size_t split_size,size_t dim);
    std::vector<Tensor> chunk(size_t chunks,size_t dim);
    void to(DI dev);
    void to(DType dt);
    template<typename T>
    inline T* data_ptr() {
        if (!data_) {
            LOG_WARN("data_ptr try to get pointer from NULL");
            return nullptr;
        }
        data_->flush_host_to_device_if_needed();
        return static_cast<T*>(data_->data_ptr());
    }

    template<typename T>
    inline const T* data_ptr() const {
        if (!data_) {
            LOG_WARN("data_ptr try to get pointer from NULL");
            return nullptr;
        }
        data_->flush_host_to_device_if_needed();
        return static_cast<const T*>(data_->data_ptr());
    }


private:

    TensorId id_=TensorId{};
    std::shared_ptr<Buffer> data_;
    std::shared_ptr<Buffer> tmp_data_; // 临时数据视图，给打印看
    std::shared_ptr<Tensor> grad_;

    TensorMeta meta;
    // 计算图使用
    std::optional<ValueId> sym_;

    void allocate_();
    void bind_unallocated_();
    void fill_(float value=0.0f);

    void ensure_storage_();
    void ensure_host_mirror_() const;


};




}

namespace std {
template <>
struct hash<EC::AT::TensorId> {
    size_t operator()(const EC::AT::TensorId& id) const noexcept {
        return std::hash<uint64_t>{}(id.value);
    }
};
}