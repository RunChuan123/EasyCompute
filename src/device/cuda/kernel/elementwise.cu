
#ifdef EC_ENABLE_CUDA

#include <cuda_runtime.h>

#define FLOAT4(addr) (reinterpret_cast<float4*>(addr)[0])

//add

__global__ void element_wise_add_f32x4_full_impl(float* a,float* b,float* out,size_t N){
    size_t idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 4 <= N){
        float4 reg_o;
        float4 reg_a = FLOAT4(a+idx);
        float4 reg_b = FLOAT4(b+idx);
        reg_o.x = reg_a.x + reg_b.x;
        reg_o.y = reg_a.y + reg_b.y;
        reg_o.z = reg_a.z + reg_b.z;
        reg_o.w = reg_a.w + reg_b.w;
        FLOAT4(out + idx) = reg_o;
    }
    if(idx == (N/4) * 4){
        for(size_t i = 0;i<N%4;i++){
            out[idx+i]= a[idx +i] + b[idx + i];
        }
    }
}

template<const size_t BlockSize = 256>
__global__ void element_wise_add_f32x4_loop_impl(float* a,float* b,float* out,size_t N){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t f4cnt = N/4;
    size_t stride = BlockSize * gridDim.x;
    for(size_t l = idx;l < f4cnt;l+=stride){ // l 是第几个float4
        size_t offset = l * 4;
        float4 reg_o;
        float4 reg_a = FLOAT4(a+offset);
        float4 reg_b = FLOAT4(b+offset);
        reg_o.x = reg_a.x + reg_b.x;
        reg_o.y = reg_a.y + reg_b.y;
        reg_o.z = reg_a.z + reg_b.z;
        reg_o.w = reg_a.w + reg_b.w;
        FLOAT4(out + offset) = reg_o;
    }
    if(idx == 0){
        size_t tail = f4cnt*4;
        for(size_t i = tail;i < N;i++){
            out[i] = a[i] + b[i];
        }
    }
}

__global__ void element_wise_sub_f32x4_full_impl(float* a,float* b,float* out,size_t N){
    size_t idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 4 <= N){
        float4 reg_o;
        float4 reg_a = FLOAT4(a+idx);
        float4 reg_b = FLOAT4(b+idx);
        reg_o.x = reg_a.x - reg_b.x;
        reg_o.y = reg_a.y - reg_b.y;
        reg_o.z = reg_a.z - reg_b.z;
        reg_o.w = reg_a.w - reg_b.w;
        FLOAT4(out + idx) = reg_o;
    }
    if(idx == (N/4) * 4){
        for(size_t i = 0;i<N%4;i++){
            out[idx+i]= a[idx +i] - b[idx + i];
        }
    }
}

template<const size_t BlockSize = 256>
__global__ void element_wise_sub_f32x4_loop_impl(float* a,float* b,float* out,size_t N){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t f4cnt = N/4;
    size_t stride = BlockSize * gridDim.x;
    for(size_t l = idx;l < f4cnt;l+=stride){ // l 是第几个float4
        size_t offset = l * 4;
        float4 reg_o;
        float4 reg_a = FLOAT4(a+offset);
        float4 reg_b = FLOAT4(b+offset);
        reg_o.x = reg_a.x - reg_b.x;
        reg_o.y = reg_a.y - reg_b.y;
        reg_o.z = reg_a.z - reg_b.z;
        reg_o.w = reg_a.w - reg_b.w;
        FLOAT4(out + offset) = reg_o;
    }
    if(idx == 0){
        size_t tail = f4cnt*4;
        for(size_t i = tail;i < N;i++){
            out[i] = a[i] - b[i];
        }
    }
}

__global__ void element_wise_mul_f32x4_full_impl(float* a,float* b,float* out,size_t N){
    size_t idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 4 <= N){
        float4 reg_o;
        float4 reg_a = FLOAT4(a+idx);
        float4 reg_b = FLOAT4(b+idx);
        reg_o.x = reg_a.x * reg_b.x;
        reg_o.y = reg_a.y * reg_b.y;
        reg_o.z = reg_a.z * reg_b.z;
        reg_o.w = reg_a.w * reg_b.w;
        FLOAT4(out + idx) = reg_o;
    }
    if(idx == (N/4) * 4){
        for(size_t i = 0;i<N%4;i++){
            out[idx+i]= a[idx +i] * b[idx + i];
        }
    }
}

template<const size_t BlockSize = 256>
__global__ void element_wise_mul_f32x4_loop_impl(float* a,float* b,float* out,size_t N){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t f4cnt = N/4;
    size_t stride = BlockSize * gridDim.x;
    for(size_t l = idx;l < f4cnt;l+=stride){ // l 是第几个float4
        size_t offset = l * 4;
        float4 reg_o;
        float4 reg_a = FLOAT4(a+offset);
        float4 reg_b = FLOAT4(b+offset);
        reg_o.x = reg_a.x * reg_b.x;
        reg_o.y = reg_a.y * reg_b.y;
        reg_o.z = reg_a.z * reg_b.z;
        reg_o.w = reg_a.w * reg_b.w;
        FLOAT4(out + offset) = reg_o;
    }
    if(idx == 0){
        size_t tail = f4cnt*4;
        for(size_t i = tail;i < N;i++){
            out[i] = a[i] * b[i];
        }
    }
}

__global__ void element_wise_div_f32x4_full_impl(float* a,float* b,float* out,size_t N){
    size_t idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 4 <= N){
        float4 reg_o;
        float4 reg_a = FLOAT4(a+idx);
        float4 reg_b = FLOAT4(b+idx);

        reg_o.x = reg_a.x / reg_b.x;
        reg_o.y = reg_a.y / reg_b.y;
        reg_o.z = reg_a.z / reg_b.z;
        reg_o.w = reg_a.w / reg_b.w;
        FLOAT4(out + idx) = reg_o;
    }
    if(idx == (N/4) * 4){
        for(size_t i = 0;i<N%4;i++){
            out[idx+i]= a[idx +i] / b[idx + i];
        }
    }
}

template<const size_t BlockSize = 256>
__global__ void element_wise_div_f32x4_loop_impl(float* a,float* b,float* out,size_t N){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t f4cnt = N/4;
    size_t stride = BlockSize * gridDim.x;
    for(size_t l = idx;l < f4cnt;l+=stride){ // l 是第几个float4
        size_t offset = l * 4;
        float4 reg_o;
        float4 reg_a = FLOAT4(a+offset);
        float4 reg_b = FLOAT4(b+offset);
        reg_o.x = reg_a.x / reg_b.x;
        reg_o.y = reg_a.y / reg_b.y;
        reg_o.z = reg_a.z / reg_b.z;
        reg_o.w = reg_a.w / reg_b.w;
        FLOAT4(out + offset) = reg_o;
    }
    if(idx == 0){
        size_t tail = f4cnt*4;
        for(size_t i = tail;i < N;i++){
            out[i] = a[i] * b[i];
        }
    }
}

__global__ void product(){
    
}
#endif