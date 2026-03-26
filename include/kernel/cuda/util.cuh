#pragma once
#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA



#include <cuda_runtime.h>

#define WARP_SIZE 32
#define ActivatedMask 0xffffffff

template<const size_t kWarpSize = WARP_SIZE>
__device__ float warp_reduce_sum_f32(float value){
#pragma unroll
    for(int offset  = kWarpSize >> 1;offset  >=1 ;offset >>=1){
        value += __shfl_xor_sync(ActivatedMask,value,offset )
    }
    return value;
}

template<const size_t kWarpSize = WARP_SIZE>
__device__ float warp_reduce_max_f32(float value){
#pragma unroll
    for(int offset  = kWarpSize >> 1;offset  >=1 ;offset >>=1){
        value = fmax(__shfl_xor_sync(ActivatedMask,value,offset ) ,value); 
    }
    return value;
}

template<const size_t BlockSize = 256>
__global__ void block_reduce_sum_f32(float* a,float* b,size_t N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * BlockSize + tid;
    constexpr int NumWarp = (BlockSize + WARP_SIZE -1) / WARP_SIZE;
    __shared__ float reduce_mem[NumWarp];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum = (idx < N) ? a[idx] : 0.0f;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if(lane == 0){
        reduce_mem[warp] = sum;
    }
    __syncthreads();
    sum = (lane < NumWarp)? reduce_mem[lane]:0.0f;
    if(warp == 0){
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    }
    if(tid == 0){
        atomicAdd(b,sum);
    }
}

#endif