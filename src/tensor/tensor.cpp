#include <memory>
#include <cassert>
#include <iomanip>

#include "kernel/cpu/mem.hpp"
#include "tensor/tensor.hpp"
#include "tensor/shape.hpp"
#include "util/rand.h"
#include "util/err.hpp"


namespace EC::AT{

template<typename T>
inline void require_tensor_type(DType dt, const char* where) {
    if (prim_dtype<T>() != dt) {
        throw TypeException(
            std::string(where) +
            ": requested cpp type=" + name_dtype(prim_dtype<T>()) +
            ", but tensor dtype=" + name_dtype(dt)
        );
    }
}

template<typename T>
T& Tensor::at(const Shape& index) {
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
const T& Tensor::at(const Shape& index) const {
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
size_t Tensor::offset(const Shape& s) const {
    if (s.rank() != rank()) throw ShapeException("offset rank mismatch");

    const auto cur_strides = strides();
    size_t linear = 0;
    for (size_t i = 0; i < rank(); ++i) {
        if (s[i] >= meta.shape[i]) throw ShapeException("offset index out of range");
        linear += s[i] * cur_strides[i];
    }
    return linear;
}
/** 
 * 清理 buffer 和 grad 数据
 */
bool Tensor::clear() {
    bool had = static_cast<bool>(data_);
    data_.reset();
    grad_.reset();
    meta.shape = Shape({});
    return had;
}
inline Tensor Tensor::view(Shape s = {}) const{
    if(s.empty())s = meta.shape.clone();
    if(s.numel() != meta.numel()) throw ShapeException("view numel mismatch");
    Tensor out;
    out.id_ = make_tensor_id();
    out.meta = meta;
    out.meta.shape = std::move(s);
    out.data_ = std::make_shared<Buffer>(*data_);
    out.grad_.reset();
    out.sym_.reset();
    return out;
}

void Tensor::ensure_storage_() {
    if (!data_) bind_unallocated_();
    if (!data_->allocated()) data_->allocate();
}

void Tensor::ensure_host_mirror_() const {
    if (!data_) throw TensorException("ensure_host_mirror_: no buffer");
    data_->ensure_host_mirror();
}

void Tensor::allocate_() {
    data_ = Buffer::make(
        meta.nbytes(),
        meta.dtype,
        meta.device,
        64
    );
}

void Tensor::bind_unallocated_() {
    data_ = Buffer::make_unallocated(
        meta.nbytes(),
        meta.dtype,
        meta.device,
        64
    );
}

void Tensor::fill_(float value){
    ensure_storage_();
    auto fill_host = [&](void* ptr){
        switch (meta.dtype) {
            case DType::f32:
                CPU::fill<float>(ptr, meta.numel(), static_cast<float>(value));
                break;
            case DType::f64:
                CPU::fill<double>(ptr, meta.numel(), static_cast<double>(value));
                break;
            case DType::i32:
                CPU::fill<int32_t>(ptr, meta.numel(), static_cast<int32_t>(value));
                break;
            default:
                throw TypeException("fill_ unsupported dtype");
        }
    };
    if (meta.device.type() == DeviceType::CPU) {
        fill_host(data_->data_ptr());
        data_->invalidate_host();
        return;
    }
    ensure_host_mirror_();
    fill_host(data_->host_data_ptr());
    data_->mark_host_dirty();
    data_->flush_host_to_device_if_needed();
}

void Tensor::print(size_t width, size_t prec) const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << meta.shape.to_string()
        << ", dtype=" << static_cast<int>(meta.dtype)
        << ", device=" << meta.device.to_string()
        << ", id=" << id_.to_string()
        << ")\n";

    if (empty()) {
        oss << "<empty>";
        std::cout << oss.str() << std::endl;
        return;
    }

    if (meta.dtype != DType::f32) {
        oss << "<print only implemented for f32 currently>";
        std::cout << oss.str() << std::endl;
        return;
    }

    const float* p = nullptr;
    if (meta.device.type() == DeviceType::CPU) {
        p = data_ptr<float>();
    } else {
        ensure_host_mirror_();
        p = static_cast<const float*>(data_->host_data_ptr());
    }

    oss << std::fixed << std::setprecision(static_cast<int>(prec));

    if (rank() == 0 || numel() == 1) {
        oss << p[0];
    } else if (rank() == 1) {
        oss << "[";
        for (size_t i = 0; i < numel(); ++i) {
            if (i) oss << ", ";
            oss << std::setw(static_cast<int>(width)) << p[i];
        }
        oss << "]";
    } else if (rank() == 2) {
        size_t rows = meta.shape[0];
        size_t cols = meta.shape[1];
        oss << "[\n";
        for (size_t r = 0; r < rows; ++r) {
            oss << "  [";
            for (size_t c = 0; c < cols; ++c) {
                if (c) oss << ", ";
                oss << std::setw(static_cast<int>(width)) << p[r * cols + c];
            }
            oss << "]";
            if (r + 1 != rows) oss << ",";
            oss << "\n";
        }
        oss << "]";
    } else {
        oss << "[";
        size_t limit = std::min<size_t>(numel(), 16);
        for (size_t i = 0; i < limit; ++i) {
            if (i) oss << ", ";
            oss << p[i];
        }
        if (numel() > limit) oss << ", ...";
        oss << "]";
    }

    std::cout << oss.str() << std::endl;
}

Tensor Tensor::scalar(float value, DType dt, DI dev) {
    return Tensor(Shape({1}), value, dt, dev, false);
}

Tensor Tensor::vector(std::initializer_list<float> vl, Shape s, DType dt, DI dev) {
    if (s.numel() != vl.size()) {
        throw ShapeException("vector initializer size mismatch");
    }
    Tensor t = Tensor::Empty(s, dt, dev);

    if (dt != DType::f32) {
        throw TypeException("vector only implemented for f32 currently");
    }

    if (dev.type() == DeviceType::CPU) {
        std::copy(vl.begin(), vl.end(), t.data_ptr<float>());
    } else {
        t.ensure_host_mirror_();
        std::copy(vl.begin(), vl.end(), static_cast<float*>(t.data_->host_data_ptr()));
        t.data_->mark_host_dirty();
        t.data_->flush_host_to_device_if_needed();
    }
    return t;
}
Tensor Tensor::zeros(Shape s, DType dt, DI dev) {
    return Tensor(std::move(s), 0.0f, dt, dev, false);
}

Tensor Tensor::ones(Shape s, DType dt, DI dev) {
    return Tensor(std::move(s), 1.0f, dt, dev, false);
}

Tensor Tensor::E(Shape s, DType dt, DI dev) {
    if (!s.is_matrix() || s[0] != s[1]) {
        throw ShapeException("E() requires square matrix shape");
    }
    Tensor t = Tensor::zeros(s, dt, dev);
    for (size_t i = 0; i < s[0]; ++i) {
        t.at<float>(Shape({i, i})) = 1.0f;
    }
    if (dev.type() != DeviceType::CPU) {
        t.data_->flush_host_to_device_if_needed();
    }
    return t;
}

Tensor Tensor::uniform(Shape s, float low, float high, DType dt, DI dev) {
    Tensor t = Tensor::Empty(s, dt, dev);

    if (dt != DType::f32) {
        throw TypeException("uniform only implemented for f32 currently");
    }

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(low, high);

    if (dev.type() == DeviceType::CPU) {
        float* p = t.data_ptr<float>();
        for (size_t i = 0; i < t.numel(); ++i) p[i] = dis(gen);
    } else {
        t.ensure_host_mirror_();
        float* p = static_cast<float*>(t.data_->host_data_ptr());
        for (size_t i = 0; i < t.numel(); ++i) p[i] = dis(gen);
        t.data_->mark_host_dirty();
        t.data_->flush_host_to_device_if_needed();
    }

    return t;
}

Tensor Tensor::normal(Shape s, float mean, float stddev, DType dt, DI dev) {
    Tensor t = Tensor::Empty(s, dt, dev);

    if (dt != DType::f32) {
        throw TypeException("normal only implemented for f32 currently");
    }

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(mean, stddev);

    if (dev.type() == DeviceType::CPU) {
        float* p = t.data_ptr<float>();
        for (size_t i = 0; i < t.numel(); ++i) p[i] = dis(gen);
    } else {
        t.ensure_host_mirror_();
        float* p = static_cast<float*>(t.data_->host_data_ptr());
        for (size_t i = 0; i < t.numel(); ++i) p[i] = dis(gen);
        t.data_->mark_host_dirty();
        t.data_->flush_host_to_device_if_needed();
    }

    return t;
}

Tensor Tensor::likes(Tensor& rhs, float v) {
    return Tensor(rhs.getShape(), v, rhs.getDtype(), rhs.getDevice(), rhs.requires_grad());
}

Tensor& Tensor::reshape(Shape s) {
    if (s.numel() != meta.numel()) {
        throw ShapeException("reshape numel mismatch");
    }
    if (!is_contiguous()) {
        throw TensorException("reshape requires contiguous tensor");
    }
    meta.shape = std::move(s);
    return *this;
}

Tensor Tensor::ravel() {
    Tensor out;
    out.id_ = make_tensor_id();
    out.meta = meta;
    out.meta.shape = Shape({meta.numel()});
    out.meta.is_contiguous = true;
    out.data_ = std::make_shared<Buffer>(*data_);
    out.data_->is_contiguous = true;
    return out;
}

Tensor Tensor::flatten() {
    return ravel();
}

Tensor& Tensor::unsqueeze(size_t dim) {
    if (dim > rank()) throw ShapeException("unsqueeze dim out of range");
    std::vector<size_t> dims;
    dims.reserve(rank() + 1);
    for (size_t i = 0; i < dim; ++i) dims.push_back(meta.shape[i]);
    dims.push_back(1);
    for (size_t i = dim; i < rank(); ++i) dims.push_back(meta.shape[i]);
    meta.shape = Shape(dims);

    return *this;
}

Tensor& Tensor::squeeze(size_t dim) {
    if (dim >= rank()) throw ShapeException("squeeze dim out of range");
    if (meta.shape[dim] != 1) throw ShapeException("squeeze requires dimension size == 1");

    std::vector<size_t> dims;
    dims.reserve(rank() - 1);
    for (size_t i = 0; i < rank(); ++i) {
        if (i != dim) dims.push_back(meta.shape[i]);
    }
    meta.shape = Shape(dims);

    return *this;
}

Tensor& Tensor::transpose(size_t dim0, size_t dim1) {
    if (dim0 >= rank() || dim1 >= rank()) {
        throw ShapeException("transpose dim out of range");
    }
    if (dim0 == dim1) return *this;

    std::vector<size_t> dims(rank());
    for (size_t i = 0; i < rank(); ++i) dims[i] = i;
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

Tensor& Tensor::permute(std::vector<size_t> dims) {
    if (dims.size() != rank()) throw ShapeException("permute dims rank mismatch");

    std::vector<bool> seen(rank(), false);
    for (auto d : dims) {
        if (d >= rank() || seen[d]) throw ShapeException("permute dims invalid");
        seen[d] = true;
    }

    std::vector<size_t> old_dims(rank());
    for (size_t i = 0; i < rank(); ++i) old_dims[i] = meta.shape[i];

    std::vector<size_t> new_dims(rank());
    for (size_t i = 0; i < rank(); ++i) new_dims[i] = old_dims[dims[i]];

    meta.shape = Shape(new_dims);
    data_->is_contiguous = false;
    meta.is_contiguous = false;
    return *this;
}

Tensor Tensor::clone() {
    data_->flush_host_to_device_if_needed();

    Tensor out = Tensor::Empty(meta.shape, meta.dtype, meta.device);
    out.meta.requires_grad = meta.requires_grad;
    out.meta.is_contiguous = meta.is_contiguous;
    out.data_->is_contiguous = data_->is_contiguous;

    if (meta.nbytes() == 0) return out;

    auto& dm = Dev::DeviceManager::get_instance();
    auto s = dm.createStream(meta.device, 0);
    dm.memcpyAsync(out.data_->data_ptr(), meta.device,
                   data_->data_ptr(), meta.device,
                   meta.nbytes(), s);
    dm.synchronize(s);
    dm.destroyStream(s);

    out.data_->invalidate_host();
    return out;
}

void Tensor::copy_(Tensor& src) {
    if (meta.shape != src.meta.shape) throw ShapeException("copy_ shape mismatch");
    if (meta.dtype != src.meta.dtype) throw TypeException("copy_ dtype mismatch");

    src.data_->flush_host_to_device_if_needed();
    ensure_storage_();

    auto& dm = Dev::DeviceManager::get_instance();

    if (meta.device.type() == DeviceType::CPU && src.meta.device.type() == DeviceType::CPU) {
        std::memcpy(data_->data_ptr(), src.data_->data_ptr(), meta.nbytes());
        data_->invalidate_host();
        return;
    }

    DI stream_dev = (meta.device.type() == DeviceType::CUDA) ? meta.device : src.meta.device;
    auto s = dm.createStream(stream_dev, 0);
    dm.memcpyAsync(data_->data_ptr(), meta.device,
                   src.data_->data_ptr(), src.meta.device,
                   meta.nbytes(), s);
    dm.synchronize(s);
    dm.destroyStream(s);

    data_->invalidate_host();
}

std::vector<Tensor> Tensor::split(size_t split_size, size_t dim) {
    if (dim >= rank()) throw ShapeException("split dim out of range");
    if (split_size == 0) throw ShapeException("split_size must be > 0");
    if (!is_contiguous()) throw TensorException("split currently requires contiguous tensor");

    std::vector<Tensor> outs;
    size_t total = meta.shape[dim];
    size_t start = 0;

    while (start < total) {
        size_t len = std::min(split_size, total - start);

        std::vector<size_t> out_dims(rank());
        for (size_t i = 0; i < rank(); ++i) out_dims[i] = meta.shape[i];
        out_dims[dim] = len;

        size_t inner = 1;
        for (size_t i = dim + 1; i < rank(); ++i) inner *= meta.shape[i];
        size_t outer = 1;
        for (size_t i = 0; i < dim; ++i) outer *= meta.shape[i];

        size_t block_elems = len * inner;
        size_t block_bytes = block_elems * meta.itemsize();
        size_t start_elem = start * inner;

        Tensor out = Tensor::Empty(Shape(out_dims), meta.dtype, meta.device);

        if (meta.device.type() == DeviceType::CPU) {
            char* dst = static_cast<char*>(out.data_->data_ptr());
            const char* src = static_cast<const char*>(data_->data_ptr());
            for (size_t o = 0; o < outer; ++o) {
                size_t src_off = (o * meta.shape[dim] * inner + start_elem) * meta.itemsize();
                size_t dst_off = (o * block_elems) * meta.itemsize();
                std::memcpy(dst + dst_off, src + src_off, block_bytes);
            }
        } else {
            ensure_host_mirror_();
            out.ensure_host_mirror_();
            char* dst = static_cast<char*>(out.data_->host_data_ptr());
            const char* src = static_cast<const char*>(data_->host_data_ptr());
            for (size_t o = 0; o < outer; ++o) {
                size_t src_off = (o * meta.shape[dim] * inner + start_elem) * meta.itemsize();
                size_t dst_off = (o * block_elems) * meta.itemsize();
                std::memcpy(dst + dst_off, src + src_off, block_bytes);
            }
            out.data_->mark_host_dirty();
            out.data_->flush_host_to_device_if_needed();
        }

        outs.push_back(std::move(out));
        start += len;
    }

    return outs;
}

std::vector<Tensor> Tensor::chunk(size_t chunks, size_t dim) {
    if (chunks == 0) throw ShapeException("chunks must be > 0");
    if (dim >= rank()) throw ShapeException("chunk dim out of range");

    size_t total = meta.shape[dim];
    size_t split_size = (total + chunks - 1) / chunks;
    return split(split_size, dim);
}

void Tensor::to(DI dev) {
    if (meta.device == dev) return;

    data_->flush_host_to_device_if_needed();

    auto new_buf = Buffer::make(meta.nbytes(), meta.dtype, dev, 64);
    auto& dm = Dev::DeviceManager::get_instance();

    if (meta.nbytes() > 0) {
        DI stream_dev = (dev.type() == DeviceType::CUDA) ? dev : meta.device;
        auto s = dm.createStream(stream_dev, 0);
        dm.memcpyAsync(new_buf->data_ptr(), dev,
                       data_->data_ptr(), meta.device,
                       meta.nbytes(), s);
        dm.synchronize(s);
        dm.destroyStream(s);
    }

    data_ = new_buf;
    meta.device = dev;
}
// TODO: 后面转为cast_contiguous
void Tensor::to(DType dt) {
    if (meta.dtype == dt) return;

    data_->flush_host_to_device_if_needed();
    ensure_host_mirror_();

    auto new_storage = std::make_shared<Storage>(meta.numel() * size_dtype(dt), meta.device, 64);
    new_storage->allocate();
    new_storage->allocate_host();

    auto new_buf = std::make_shared<Buffer>(
        new_storage, 0, meta.numel() * size_dtype(dt), dt, meta.device, data_->is_contiguous);

    if (meta.dtype == DType::f32 && dt == DType::f32) {
        std::memcpy(new_buf->storage->host_ptr, data_->storage->host_ptr, meta.nbytes());
        new_buf->storage->mark_host_dirty();
        new_buf->storage->flush_host_to_device_if_needed();
    } else {
        throw TypeException("to(DType) currently only implemented for f32->f32/no-op placeholder");
    }

    meta.dtype = dt;
    data_ = new_buf;
}

}