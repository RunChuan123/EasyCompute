# EasyCompute/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)  # 最低CMake版本（兼容C++11+）

# 项目信息：名称+语言（C++）+版本
project(EasyCompute 
        VERSION 1.0 
        LANGUAGES CXX)

        option(EASY_COMPUTE_SANITIZE "Enable ASan/UBSan" OFF)

if(EASY_COMPUTE_SANITIZE)
  add_compile_options(-O0 -g3 -fno-omit-frame-pointer -fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()
# 让 example 作为子目录构建（把你的 exe_main 放到 EasyCompute/example 里）

# 设置C++标准（必须！适配你的C++11+特性：=default、inline、枚举类等）
set(CMAKE_CXX_STANDARD 20)          # 推荐C++17，兼容所有现代特性
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制使用指定的C++标准
set(CMAKE_CXX_EXTENSIONS OFF)       # 禁用编译器扩展（跨平台）

# 可选：若项目支持GPU（CUDA），开启CUDA支持
option(EASY_COMPUTE_ENABLE_GPU "Enable GPU support" ON)
if(EASY_COMPUTE_ENABLE_GPU)
    enable_language(CUDA)
    # 设置CUDA标准（可选）
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # 关键：指定CUDA架构（避免编译警告，适配你的GPU）
    # 可填多个架构：比如 75(RTX20/30)、80(A100)、86(RTX30/40)、89(RTX40)、90(RTX50)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# 输出路径：统一编译产物路径（可选，方便管理）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)  # 可执行文件
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)  # 动态库
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)  # 静态库
include_directories(
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include  # 指向include根目录
)
# 进入src目录编译源码
add_subdirectory(src)


add_subdirectory(example)  

# 可选：添加示例/测试程序（比如测试Buffer、Shape等）
# add_subdirectory(examples)