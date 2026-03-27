#pragma once

#include "tensor/api.hpp"
#include "graph/api.hpp"
#include "backends/cuda/cuda_runtime.cuh"
#include "backends/cpu/cpu_runtime.hpp"

namespace EC
{
    


void EC_INIT(){

#ifdef EC_ENABLE_CUDA
    Dev::registerCUDABackends();
#endif
    Dev::registerCPUBackend();

    AT::register_all_kernels();


}


} // namespace EC
