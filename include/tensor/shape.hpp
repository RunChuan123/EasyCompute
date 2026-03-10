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

        inline size_t rank()const{return dims.size();}

        inline size_t numel()const{
            if(is_scalar()) return 1;
            size_t n = 1;
            for(size_t i:dims)n*=i;
            return n;
        }
        
        bool operator==(const Shape& o)const{return dims == o.dims;}

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

    };

    inline std::ostream& operator<<(std::ostream& os,const Shape& s){
        os << "[ ";
        for(size_t i =0;i<s.dims.size();i++){
            os << s.dims[i];
            if(i != s.dims.size()-1) os << ", "; 
        }
        os << "]";
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
} // namespace EC
