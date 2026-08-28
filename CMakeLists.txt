cmake_minimum_required(VERSION 3.22)

project(EasyCompute VERSION 2.0.0 LANGUAGES CXX)

option(EC_ENABLE_CUDA "Build the CUDA backend when a CUDA compiler is available" ON)
option(EC_BUILD_TESTS "Build EasyCompute tests" ON)
option(EC_BUILD_EXAMPLES "Build EasyCompute examples" ON)
option(EC_ENABLE_SANITIZERS "Enable address and undefined-behaviour sanitizers" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EC_HAS_CUDA OFF)
if(EC_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(EC_HAS_CUDA ON)
  else()
    message(STATUS "CUDA compiler not found; building the CPU backend only")
  endif()
endif()

add_library(easycompute
  src/core/builtin_plugins.cpp
  src/core/capability.cpp
  src/core/dtype.cpp
  src/core/id.cpp
  src/core/int_tuple.cpp
  src/core/layout.cpp
  src/core/plugin.cpp
  src/core/runtime.cpp
  src/compute/dispatch.cpp
  src/compute/registry.cpp
  src/compute/cpu/binary.cpp
  src/memory/allocator.cpp
  src/memory/storage.cpp
  src/tensor/tensor.cpp
)
add_library(EasyCompute::easycompute ALIAS easycompute)

target_include_directories(easycompute PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(easycompute PUBLIC cxx_std_20)
target_compile_options(easycompute PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
)

if(EC_HAS_CUDA)
  target_sources(easycompute PRIVATE
    src/memory/allocator_cuda.cu
    src/compute/cuda/binary.cu
  )
  target_compile_definitions(easycompute PUBLIC EC_HAS_CUDA=1)
  set_target_properties(easycompute PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
else()
  target_compile_definitions(easycompute PUBLIC EC_HAS_CUDA=0)
endif()

if(EC_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(easycompute PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(easycompute PUBLIC -fsanitize=address,undefined)
endif()

if(EC_BUILD_TESTS)
  enable_testing()
  add_executable(easycompute_tests tests/test_main.cpp)
  target_link_libraries(easycompute_tests PRIVATE EasyCompute::easycompute)
  add_test(NAME easycompute_tests COMMAND easycompute_tests)
endif()

if(EC_BUILD_EXAMPLES)
  add_executable(tensor_demo examples/tensor_demo.cpp)
  target_link_libraries(tensor_demo PRIVATE EasyCompute::easycompute)
endif()
