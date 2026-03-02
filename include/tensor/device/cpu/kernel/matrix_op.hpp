#pragma once
#include "tensor/tensor.hpp"

namespace EC::AT::CPU{
namespace MatrixTOp{


// 辅助函数：前向替换解 Ly = b（L是单位下三角矩阵）
Tensor forward_substitution(const Tensor& L, const Tensor& b) ;
// 辅助函数：后向替换解 Ux = y（U是上三角矩阵）
Tensor backward_substitution(const Tensor& U, const Tensor& y);

// 基于Crout LU分解的行列式计算 O(n³)
float det(const Tensor& t) ;
// 基于逆矩阵的伴随矩阵计算（adj(A) = det(A) * A⁻¹）
Tensor adjugate(const Tensor& t) ;
Tensor inverse(const Tensor& t) ;
std::pair<Tensor,Tensor> lu_decompose_crout(const Tensor& t);
std::pair<Tensor,Tensor> lu_decompose_doolittle(const Tensor& t);
Tensor submatrix(const Tensor& t,size_t exclude_row,size_t exclude_col) ;
float det_lapras(const Tensor& t) ;
Tensor adjugate_lapras(const Tensor& t);
// A⁻¹ = adj(A)/det(A)）
Tensor inverse_lapras(const Tensor& t);

}
}