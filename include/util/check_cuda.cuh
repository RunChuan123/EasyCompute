#pragma once

namespace EC
{

inline const char* cuda_get_error_detail(cudaError_t err, const char* file, int line) {
    static char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg), 
             "CUDA Error: %s (code: %d) at %s:%d",
             cudaGetErrorString(err), err, file, line);
    return err_msg;
}


#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t _err = (err);                                            \
        if (_err != cudaSuccess) {                                           \
            throw std::runtime_error(cuda_get_error_detail(_err, __FILE__, __LINE__)); \
        }                                                                    \
    } while (0)

// 宏2：检查CUDA错误（失败则打印日志，不抛异常）→ 非核心逻辑用
#define CUDA_CHECK_WARN(err)                                                 \
    do {                                                                     \
        cudaError_t _err = (err);                                            \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "[WARN] %s\n", cuda_get_error_detail(_err, __FILE__, __LINE__)); \
        }                                                                    \
    } while (0)

// 宏3：检查核函数启动错误（核函数本身无法直接返回错误，需同步检查）
#define CUDA_CHECK_KERNEL()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)

}