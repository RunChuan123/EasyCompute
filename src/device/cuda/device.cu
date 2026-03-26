#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA

#include <cstdint>

#include "backends/cuda/cuda_runtime.cuh"
#include "kernel/tensor_op.hpp"
#include "kernel/cuda/kernel.cuh"


void register_cuda_kernels(KernelTable& kt){
    // kt.re
}

#endif