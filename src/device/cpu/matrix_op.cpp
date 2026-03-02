#include <cassert>

#include "tensor/tensor_op.hpp"
#include "tensor/tensor.hpp"
#include "tensor/device/cpu/kernel/matrix_op.hpp"
#include "util/rand.h"

#define AP2ZERO (1e-8)

namespace EC::AT::CPU{
namespace MatrixTOp{

// 辅助函数：前向替换解 Ly = b（L是单位下三角矩阵）
Tensor forward_substitution(const Tensor& L, const Tensor& b) {
    assert(L.shape().is_square());
    assert(b.shape().dims[0] == L.shape().dims[0] && b.shape().dims[1] == 1);
    size_t n = L.shape().dims[0];
    Tensor y({n, 1}, 0.0f, L.dtype(),L.device());

    for (size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < i; ++k) {
            sum += L.at({i, k}) * y.at({k, 0});
        }
        y.at({i, 0}) = b.at({i, 0}) - sum;
    }
    return y;
}

// 辅助函数：后向替换解 Ux = y（U是上三角矩阵）
Tensor backward_substitution(const Tensor& U, const Tensor& y) {
    assert(U.shape().is_square());
    assert(y.shape().dims[0] == U.shape().dims[0] && y.shape().dims[1] == 1);
    size_t n = U.shape().dims[0];
    Tensor x({n, 1}, 0.0f, U.dtype(),U.device());

    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        float sum = 0.0f;
        for (size_t k = static_cast<size_t>(i) + 1; k < n; ++k) {
            sum += U.at({static_cast<size_t>(i), k}) * x.at({k, 0});
        }
        size_t row = static_cast<size_t>(i);
        if (std::fabs(U.at({row, row})) < AP2ZERO) {
            throw std::runtime_error("backward substitution failed: zero pivot");
        }
        x.at({row, 0}) = (y.at({row, 0}) - sum) / U.at({row, row});
    }
    return x;
}

// 基于Crout LU分解的行列式计算 O(n³)
float det(const Tensor& t) {
    assert(t.shape().is_square());
    size_t n = t.shape().dims[0];

    // 极小矩阵用拉普拉斯
    if (n == 1) return t.at({0, 0});
    if (n == 2) return t.at({0,0}) * t.at({1,1}) - t.at({0,1}) * t.at({1,0});

    auto [L, U] = lu_decompose_crout(t);
    float determinant = 1.0f;
    for (size_t i = 0; i < n; ++i) {
        determinant *= U.at({i, i});
    }
    return determinant;
}

// 基于逆矩阵的伴随矩阵计算（adj(A) = det(A) * A⁻¹）
Tensor adjugate(const Tensor& t) {
    assert(t.shape().is_square());
    float determinant = det(t);
    Tensor inv = inverse(t);

    size_t n = t.shape().rows();
    Tensor adj({n, n}, 0.0f,t.dtype(), t.device());
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adj.at({i, j}) = inv.at({i, j}) * determinant;
        }
    }
    return adj;
}


Tensor inverse(const Tensor& t) {
    assert(t.shape().is_square());
    size_t n = t.shape().rows();
    float determinant = det(t);
    
    if (std::fabs(determinant) < AP2ZERO) {
        throw std::runtime_error("inverse: singular matrix(det=0) cannot inverse");
    }

    // Crout LU分解
    auto [L, U] = lu_decompose_crout(t);
    Tensor inv({n, n}, 0.0f, t.dtype(),t.device());

    // 对单位矩阵的每一列求解 Ax = e_i，得到逆矩阵的每一列
    for (size_t j = 0; j < n; ++j) {
        // 构造单位向量 e_j（第j列为1，其余为0）
        Tensor b({n, 1}, 0.0f,t.dtype(), t.device());
        b.at({j, 0}) = 1.0f;
        
        // 解 Ly = b → Ux = y → x是逆矩阵的第j列
        Tensor y = forward_substitution(L, b);
        Tensor x = backward_substitution(U, y);
        
        // 将x赋值给逆矩阵的第j列
        for (size_t i = 0; i < n; ++i) {
            inv.at({i, j}) = x.at({i, 0});
        }
    }

    return inv;
}

std::pair<Tensor,Tensor> lu_decompose_crout(const Tensor& t){
    assert(t.shape().is_square());
    size_t n = t.shape().rows();
    Tensor L({n,n},0.0f,t.dtype(),t.device());
    Tensor U({n,n},0.0f,t.dtype(),t.device());

    for(size_t i = 0; i<n;i++){
        L.at({i,i}) = 1;

        for(size_t j = i;j<n;j++){
            float sum = 0.0f;
            for(size_t k=0;k<i;k++){
                sum+=L.at({i,k}) * U.at({k,j});
            }
            U.at({i,j}) = t.at({i,j}) - sum;
        }

        for(size_t j = i+1;j<n;j++){
            float sum = 0.0f;
            for(size_t k =0;k<i;k++){
                sum += L.at({j,k}) * U.at({k,i});
            }
            if(std::fabs(U.at({i,i})) < AP2ZERO){
                throw std::runtime_error("LU decomposition failed: zero pivot(matrix is singular)");
            }
            L.at({j,i}) = (t.at({j,i})-sum) / U.at({i,i});
        }
    }
    return {L,U};
}

std::pair<Tensor,Tensor> lu_decompose_doolittle(const Tensor& t){
    assert(t.shape().is_square());
    size_t n = t.shape().rows();
    Tensor L({n,n},0.0f,t.dtype(),t.device());
    Tensor U({n,n},0.0f,t.dtype(),t.device());

    for(size_t i =0 ;i<n;i++){
        U.at({i,i}) = 1;
        for(size_t j=i;j<n;j++){
            float sum = 0.0f;
            for(size_t k=0;k<i;k++){
                sum += L.at({j,k}) * U.at({k,i});
            }
            L.at({j,i}) = t.at({j,i})- sum;
        }
        for(size_t j = i+1;j<n;j++){
            float sum = 0.0f;
            for(size_t k =0;k<i;k++){
                sum += L.at({i,k}) * U.at({k,j});
            }
            if(std::fabs(L.at({i,i})) < AP2ZERO){
                throw std::runtime_error("LU decomposition failed: zero pivot(matrix is singular)");
            }
            U.at({i,j}) =( t.at({i,j}) - sum)/L.at({i,i});
        }
    }
    return {L,U};
}

Tensor submatrix(const Tensor& t,size_t exclude_row,size_t exclude_col) {
    assert(t.shape().is_square());
    size_t n = t.shape().rows()-1;
    Tensor sub({n,n},0.0f,t.dtype(),t.device());
    size_t sub_i = 0,sub_j =0;
    for(size_t i = 0;i<t.shape().rows();i++){
        if(i == exclude_row) continue;
        sub_j = 0;
        for(size_t j = 0;j<t.shape().cols();j++){
            if(j == exclude_col) continue;
            sub.at({sub_i,sub_j}) = t.at({i,j});
            sub_j++;
        }
        sub_i++;
    }
    return sub;
}

float det_lapras(const Tensor& t) {
    assert(t.shape().is_square());
    size_t n = t.shape().rows();
    if(n == 1)
        return t.at({0,0});
    if(n==2)
        return t.at({0,0}) * t.at({1,1}) - t.at({0,1}) * t.at({1,0});
    float determinant = 0.0f;
    for(size_t j =0;j<n;j++){
        float sign = (j%2==0)? 1.0f:-1.0f;
        determinant += sign * t.at({0,j}) * det_lapras(submatrix(t,0,j));
    }
    return determinant;
}

Tensor adjugate_lapras(const Tensor& t){
    assert(t.shape().is_square());
    size_t n = t.shape().rows();
    Tensor adj({n,n},0.0,t.dtype(),t.device());
    for(size_t i = 0;i<n;++i){
        for(size_t j = 0;j<n;j++){
            float sign = ((i+j)%2==0)? 1.0f:-1.0f;
            float cofactor = sign * det_lapras(submatrix(t,i,j));
            adj.at({i,j}) = cofactor;
        }
    }
    return adj;
}

// A⁻¹ = adj(A)/det(A)）
Tensor inverse_lapras(const Tensor& t){
    float determinant = det_lapras(t);
    if(std::fabs(determinant) < 1e-6){
        throw std::runtime_error("inverse_lapras: singular matrix(det=0) cannot inverse");
    }
    Tensor adj = adjugate_lapras(t);
    size_t n = t.shape().rows();
    Tensor inv({n,n},0.0,t.dtype(),t.device());
    for(size_t i = 0;i<n;i++){
        for(size_t j = 0;j<n;j++){
            inv.at({i,j}) = adj.at({i,j}) / determinant;
        }
    }
    return inv;
}


}
}