#include <cstdint>


#include "backends/cuda/cuda_runtime.hpp"
#include "kernel/tensor_op.hpp"
#include "kernel/cuda/kernel.cuh"


namespace EC::AT
{
void register_cuda_kernels(KernelTable& kt){
    kt.kernel_table[0][0][0];
}
}