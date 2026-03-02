# EasyCompute/src/CMakeLists.txt

# 步骤1：收集所有源码文件（递归查找src下的所有.cpp/.cu文件）
file(GLOB_RECURSE EASY_COMPUTE_SRCS 
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    #  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)  # 若有CUDA源码（.cu）
)
# 步骤2：排除不需要编译的文件（可选，比如测试代码）
# list(REMOVE_ITEM EASY_COMPUTE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

# 步骤3：生成库文件（推荐动态库SHARED，也可改静态库STATIC）
add_library(easycompute SHARED ${EASY_COMPUTE_SRCS})

# 步骤4：设置头文件路径（关键！让外部能找到EasyCompute.hpp）
# PUBLIC：外部项目使用时，也能继承这些头文件路径
target_include_directories(easycompute
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include  # 外部可见的头文件根目录
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}    # 仅源码自身可见（若src有内部头文件）
)

# 步骤5：若启用GPU，链接CUDA库（可选）
if(EASY_COMPUTE_ENABLE_GPU)
    # 查找CUDA库（CMake 3.10+）
    find_package(CUDAToolkit REQUIRED)
    # 链接CUDA库
    target_link_libraries(easycompute PRIVATE CUDA::cudart)
    # 给GPU相关源码添加编译定义（代码中可通过#ifdef EASY_COMPUTE_GPU判断）
    target_compile_definitions(easycompute PRIVATE EASY_COMPUTE_GPU)
endif()

# 步骤6：可选：添加编译选项（警告、优化）
target_compile_options(easycompute
    PRIVATE
        -Wall          # 开启所有警告（Linux/Mac）
        -Wextra        # 额外警告
        # -Werror        # 把警告视为错误（推荐开发阶段开启）
        -O0            # 优化级别（生产环境用O2/O3，调试用O0）
        # MSVC编译器选项（Windows）
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /O2>
)

