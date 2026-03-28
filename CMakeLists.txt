# EasyCompute/CMakeLists.txt
cmake_minimum_required(VERSION 3.17)  

project(EasyCompute 
        VERSION 1.0 
        LANGUAGES CXX
)

option(EASY_COMPUTE_SANITIZE "Enable ASan/UBSan" OFF)

if(EASY_COMPUTE_SANITIZE)
  add_compile_options(-O0 -g3 -fno-omit-frame-pointer -fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

set(CMAKE_CXX_STANDARD 20)          # 推荐C++17，兼容所有现代特性
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制使用指定的C++标准
set(CMAKE_CXX_EXTENSIONS OFF)       # 禁用编译器扩展（跨平台）


option(EASY_COMPUTE_ENABLE_CUDA "Enable GPU support" ON)
if(EASY_COMPUTE_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # 关键：指定CUDA架构（避免编译警告，适配你的GPU）
    # 可填多个架构：比如 75(RTX20/30)、80(A100)、86(RTX30/40)、89(RTX40)、90(RTX50)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# 输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)  # 可执行文件
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)  # 动态库
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)  # 静态库

include_directories(${PROJECT_SOURCE_DIR}/include)

# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
#     add_link_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
# endif()

add_subdirectory(src)

add_subdirectory(example)  

# 可选：添加示例/测试程序（比如测试Buffer、Shape等）
# add_subdirectory(examples)

