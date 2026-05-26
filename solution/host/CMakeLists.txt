cmake_minimum_required(VERSION 3.14)
project(opencl_kernel_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark OpenCL::OpenCL)
target_compile_options(benchmark PRIVATE -O2 -Wall -Wextra)

# Allow passing OpenCL kernel path at build time
target_compile_definitions(benchmark PRIVATE
    KERNEL_PATH="/mnt/workspace/opencl_kernel/solution/kernel/kernel.cl"
)
