cmake_minimum_required(VERSION 3.10)
project(how_to_optimize_gemm_in_gpu LANGUAGES CXX)

option(ENABLE_NVIDIA "enable nvidia" ON)
option(ENABLE_ILUVATAR "enable iluvatar" OFF)

# nvidia
if (ENABLE_NVIDIA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)

    add_executable(conv2DBwdFilter ${PROJECT_SOURCE_DIR}/src/conv2DBwdFilter.cu)
    target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda/include)
    target_link_libraries(${PROJECT_NAME} PRIVATE cudnn)
endif()

# iluvatar corex
if (ENABLE_ILUVATAR)
    set(ROOT_PATH "/usr/local/corex/" CACHE PATH "cuda installation root dir")
    set(CUDA_TOOLKIT_ROOT_DIR ${ROOT_PATH} CACHE PATH "root path to cuda/lib64 cuda/bin etc")

    set(COMPILER_PATH ${ROOT_PATH}/bin CACHE PATH "compiler binary path")
    set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/clang++ CACHE PATH "CXX compiler binary path")
    set(CMAKE_C_COMPILER ${COMPILER_PATH}/clang CACHE PATH "C compiler binary path")
    set(CUDA_ARCH "ivcore11" CACHE STRING "CUDA architecture.")

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    include(${PROJECT_SOURCE_DIR}/cmake/CUDA.cmake)

    cuda_add_executable(conv2DBwdFilter ${PROJECT_SOURCE_DIR}/src/conv2DBwdFilter.cu)
    target_include_directories(conv2DBwdFilter PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
    target_link_libraries(conv2DBwdFilter PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudnn.so)
endif()
