# cmake_minimum_required(VERSION 3.15)
# project(ExternalProject VERSION 1.0 LANGUAGES CXX)

# # 1. 设置C++标准（和EasyCompute一致）
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# # ========== 核心添加：全局调试选项（-g） ==========
# # 方式1：全局生效（所有目标），推荐开发阶段使用
# # -g：生成调试信息（Mac/Linux通用）
# # -O0：关闭优化（调试时避免代码被优化，方便断点）
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
# # 可选：Mac下若用lldb调试，可添加-g3（更详细的调试信息）
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0")

# # 2. 指定可执行文件输出路径
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# # 3. 关键：指定EasyCompute的头文件路径
# set(EASY_COMPUTE_INCLUDE_DIR "/Users/liubingshuo/Desktop/code/cpp/newp/np6/EasyCompute/include")
# include_directories(${EASY_COMPUTE_INCLUDE_DIR})

# # 4. 关键：找到EasyCompute的库文件
# set(EASY_COMPUTE_LIB_DIR "/Users/liubingshuo/Desktop/code/cpp/newp/np6/EasyCompute/build/lib")
# find_library(
#     EASY_COMPUTE_LIB          
#     NAMES easycompute         
#     PATHS ${EASY_COMPUTE_LIB_DIR}  
#     REQUIRED                  
# )
# # 5. 添加可执行文件
# add_executable(exe_main main.cpp)





# # 6. 强制指定输出路径
# set_target_properties(exe_main PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#     RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
#     RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
# )

# # ========== 方式2：仅给exe_main添加调试选项（推荐，更精准） ==========
# # 替代全局CMAKE_CXX_FLAGS，只给当前可执行文件加调试选项
# target_compile_options(exe_main PRIVATE
#     -g          # 生成调试信息
#     -O0         # 关闭优化（调试必备）
#     -Wall       # 开启警告（可选）
#     -Wextra     # 额外警告（可选）
# )

# # 7. 链接EasyCompute库
# target_link_libraries(exe_main PRIVATE ${EASY_COMPUTE_LIB})

# # 可选：GPU支持
# option(EASY_COMPUTE_ENABLE_GPU "Enable GPU support" OFF)
# if(EASY_COMPUTE_ENABLE_GPU)
#     find_package(CUDAToolkit REQUIRED)
#     target_link_libraries(exe_main PRIVATE CUDA::cudart)
#     # GPU调试：添加CUDA调试选项（可选）
#     target_compile_options(exe_main PRIVATE -G)
# endif()

add_executable(exe_main main.cpp)
target_link_libraries(exe_main PRIVATE easycompute)