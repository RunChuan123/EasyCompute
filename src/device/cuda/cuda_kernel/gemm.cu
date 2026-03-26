
#include "backends/device_enabled.hpp"

#ifdef EC_ENABLE_CUDA

#include <cuda_runtime.h>

// 每个线程计算 C 中一个元素
__global__ void gemm_f32_v1(float alpha,float* A,float* B,float beta,float* C,int M,int K,int N){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for(int i = 0; i< K;i++){
        sum += A[ty*K + i] * B[i*N + tx];
    }
    C[ty*N + tx] = alpha * sum + beta * C[ty * N + tx];
}

// block 分块计算
template<const size_t BLOCK_SIZE>
__global__ void gemm_f32_v2(float alpha,float* A,float* B,float beta,float* C,int M,int K,int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    A = &A[by*K *BK];
    B = &B[bx * BN];
    C = &C[by*N*BK + bx * BN];

    float tmp = 0;

    for(size_t i = 0;i<K;i+=BK){
        As[ty*BK + tx] = A[ty*K+tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();
        A += BK;
        B += BK * N;
        for(size_t j=0;j < BK;j++){
            tmp += As[ty*BK+j] * Bs[BN * j + tx];
        }
        __syncthreads();
    }
    C[ty*N+tx] = alpha* tmp + beta * C[ty*N+tx];
}

__global__ void gemm_f32_v3(){
    
}
#endif