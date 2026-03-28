# EasyCompute/src/CMakeLists.txt
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# 收集所有源码文件
file(GLOB_RECURSE EASY_COMPUTE_CPP_SRCS 
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)
file(GLOB_RECURSE EASY_COMPUTE_CUDA_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

# 排除不需要编译的文件 比如测试代码
list(FILTER EASY_COMPUTE_CUDA_SRCS EXCLUDE REGEX ".*/test[^/]*$")

# 生成库文件 动态库SHARED 静态库STATIC
add_library(easycompute SHARED ${EASY_COMPUTE_CPP_SRCS})

# 设置头文件路径,让外部能找到EasyCompute.hpp）
# PUBLIC：外部项目使用时，也能继承这些头文件路径
target_include_directories(easycompute
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include  
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}    # 仅源码自身可见
)

# 若启用GPU，链接CUDA库
if(EASY_COMPUTE_ENABLE_CUDA)
    # 查找CUDA Toolkit
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found, enabling GPU support")

        # 启用 CUDA 语言
        enable_language(CUDA)
        include_directories(/usr/local/cuda/include)
        link_directories(/usr/local/cuda/lib64)
        target_sources(easycompute PRIVATE ${EASY_COMPUTE_CUDA_SRCS})

        # 链接 CUDA 运行时
        target_link_libraries(easycompute PRIVATE CUDA::cudart)

        # 定义宏，代码里可以用 #ifdef EASY_COMPUTE_GPU
        target_compile_definitions(easycompute PUBLIC EC_ENABLE_CUDA)

        # CUDA 编译配置（关键！让 nvcc 正确识别）
        set_target_properties(easycompute PROPERTIES
            CUDA_ARCHITECTURES "75;80;86;89"   # 通用显卡架构
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        # CUDA 编译参数
        target_compile_options(easycompute PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -lineinfo
                -use_fast_math
                -Xcompiler=-Wall
            >
        )
    else()
        message(WARNING "CUDA not found, disabling GPU support")
        set(EASY_COMPUTE_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    endif()
endif()

target_compile_options(easycompute
    PRIVATE
        -Wall
        -Wextra
        -O0
        -Wunused-parameter
        # $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /O2>
)
   


