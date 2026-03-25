#pragma once

namespace EC
{
    
enum class TOp{
    ew_add=0,
    ew_sub,
    ew_mul,
    ew_div,
    sin,
    sin_,
    gemm,
    gemv,
    cat,
    stack,
    reshape,
    permute,
    sigmoid,
    transpose,
    relu,
    det,
    submatrix,
    adjugate,
    inverse,
    lu_decompose_crout,
    lu_decompose_doolittle,

    NumOp,
    detach

};

// optype _ datatype _ operandnum _ ...
enum class OpDesc{

    add_f32_ab



};

} // namespace EC
