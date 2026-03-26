#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA

#include <cstdint>
#include <device_launch_parameters.h>

#include "backends/cuda/cuda_runtime.hpp"
#include "kernel/tensor_op.hpp"
#include "kernel/cuda/kernel.cuh"


#endif