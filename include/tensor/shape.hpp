#pragma once

#include <vector>
#include <iostream>
#include <initializer_list>
#include <cassert>

namespace EC
{
    struct Shape{
        std::vector<size_t> dims;

        Shape():dims(){}
        Shape(const std::vector<size_t>& dims_):dims(dims_){}
        Shape(std::initializer_list<size_t> il) : dims(il) {}

        inline Shape& set(int index,size_t value){get_(index) = value;return *this;}
        inline size_t get(int index)const{return dims[get_valid_index(index)];}
        inline size_t& get_(int index){return dims[get_valid_index(index)];}
        inline const size_t& get_(int index)const{return dims[get_valid_index(index)];}

        inline bool is_scalar()const{return dims.empty();}
        inline bool is_vector()const{return dims.size()==1;}
        inline bool is_matrix()const{return dims.size()==2;}
        inline bool is_square()const{return is_matrix() && dims[0]==dims[1];}

        inline Shape clone() const noexcept {
            return Shape(this->dims);
        }

        inline bool empty()const{ return dims.empty(); }

        inline size_t rank()const{return dims.size();}

        inline size_t numel()const{
            if(is_scalar()) return 1;
            size_t n = 1;
            for(size_t i:dims)n*=i;
            return n;
        }
        
        bool operator==(const Shape& o)const{return dims == o.dims;}

        size_t& operator[] (size_t idx){
            if(idx >= dims.size()) throw std::out_of_range("operator[] out of range");
            return dims[idx];
        }
        const size_t& operator[] (size_t idx)const{
            if(idx >= dims.size()) throw std::out_of_range("operator[] out of range");
            return dims[idx];
        }

        // 语法糖
        size_t rows()const{return dims[0];}
        size_t cols()const{return dims[1];}
        
        inline size_t get_valid_index(int index) const{
            assert(!dims.empty());
            const int abs_index = std::abs(index);
            assert(abs_index <= static_cast<int>(dims.size()));
            return (index < 0)? 
                static_cast<size_t>((int)dims.size() + index) :
                static_cast<size_t>(index);
        }

        inline std::string to_string() const {
            std::ostringstream oss;
            oss << "[ ";
            for(size_t i =0;i < dims.size();i++){
                oss << dims[i];
                if(i != dims.size()-1) oss << ", "; 
            }
            oss << "]";
            return oss.str();
        }
    };

    inline std::ostream& operator<<(std::ostream& os,const Shape& s){
        os << s.to_string();
        return os;
    }

    inline std::vector<size_t> make_strides(const Shape& shape){
        if (shape.dims.empty()) {
            return {0};
        }
        std::vector<size_t> strides{shape.dims.size(),1};
        for (size_t i = shape.dims.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape.dims[i - 1];
        }
        return strides;
    }
    inline size_t compute_offset_contiguous(const Shape& shape, const Shape& index) {
        if (shape.rank() != index.rank()) {
            throw ShapeException("index rank mismatch");
        }
        auto strides = make_strides(shape);
        size_t linear = 0;
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (index[i] >= shape[i]) throw ShapeException("index out of range");
            linear += index[i] * strides[i];
        }
        return linear;
    }
} // namespace EC
