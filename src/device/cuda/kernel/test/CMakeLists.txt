cmake_minimum_required(VERSION 3.18)

project(kernel_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CUDA_STANDARD_REQUIRED on)

set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(test_g
    test_g.cu
    # ../elementwise.cu
    # ../gemm.cu
)

target_include_directories(test_g PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
set_target_properties(test_g PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_g COMMAND test_g)