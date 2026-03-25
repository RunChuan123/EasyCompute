
#ifdef EC_ENABLE_CUDA


#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "../elementwise.cu"
#include "../gemm.cu"

// __global__ void element_wise_add_f32x4_full_impl(float* a,float* b,float* out,size_t N);
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const int n = 1024;
    const size_t bytes = n * sizeof(float);
    float va = 1.0f;
    float vb = 2.0f;
    float vc = 2.0f;
    std::vector<float> h_a(n, va); // 32 * 32
    std::vector<float> h_b(n, vb);
    std::vector<float> h_c(n,vc);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a failed");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b failed");
    checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c failed");

    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "copy a failed");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "copy b failed");
    checkCuda(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice), "copy b failed");

    dim3 block(16 * 16);
    dim3 grid((32 + 16 - 1) / 16, (32 + 16 - 1) / 16);  // (2, 2)
    float alpha = 1.0f;
    float beta = 2.0f;
    gemm_f32_v2<16><<<grid, block>>>(alpha,d_a, d_b,beta, d_c,32,32,32);

    checkCuda(cudaGetLastError(), "kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "kernel sync failed");

    checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "copy c failed");

    for (int i = 0; i < n; ++i) {
        // std::cout << h_c[i];
        if (h_c[i] != va*vb * alpha * 32 + beta * vc){
            throw std::runtime_error("mismatch");
        } 
        
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "test_add_kernel passed\n";
    return EXIT_SUCCESS;
}

#endif