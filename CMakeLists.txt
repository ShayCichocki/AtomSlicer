cmake_minimum_required(VERSION 3.10...4.1)
project(slicer)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# Metal port: -DATOM_METAL=ON runs the grid smoothers on the Apple GPU (macOS only).
# Without it the build is plain C++17 as upstream. ATOM_METAL=0 in the environment forces
# the CPU path at run time.
option(ATOM_METAL "Run grid smoothers with Metal compute (macOS)" OFF)

# CUDA port: -DATOM_CUDA=ON runs the same grid kernels on an NVIDIA GPU (needs the CUDA
# toolkit, nvcc). Default target is sm_89 (RTX 40xx / Ada); override with
# -DCMAKE_CUDA_ARCHITECTURES=86 (RTX 30xx), "89;120", or "native" (needs the GPU present
# when CMake configures). ATOM_CUDA=0 in the environment forces the CPU path at run time.
# -DATOM_CUDA_EMU=ON builds the very same CUDA host code and kernels as plain C++ with a CPU
# emulation of the CUDA runtime (src/cuda/cuda_emu.h): no GPU or toolkit needed; used to
# test the CUDA code path on machines without an NVIDIA GPU.
option(ATOM_CUDA "Run grid smoothers with CUDA (NVIDIA)" OFF)
option(ATOM_CUDA_EMU "Build the CUDA backend as CPU emulation (testing without a GPU)" OFF)

if ((ATOM_METAL AND (ATOM_CUDA OR ATOM_CUDA_EMU)) OR (ATOM_CUDA AND ATOM_CUDA_EMU))
    message(FATAL_ERROR "ATOM_METAL, ATOM_CUDA and ATOM_CUDA_EMU are mutually exclusive")
endif()

file(GLOB_RECURSE sources      src/*.cpp src/*.h)
list(FILTER sources EXCLUDE REGEX "/src/cuda/")   # added below only for the CUDA builds

if (ATOM_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND sources src/cuda/gpu_cuda.cu src/cuda/kernels_common.h)
endif()
if (ATOM_CUDA_EMU)
    list(APPEND sources src/cuda/gpu_cuda_emu.cpp src/cuda/kernels_common.h src/cuda/cuda_emu.h)
endif()

if (ATOM_METAL)
    if (NOT APPLE)
        message(FATAL_ERROR "ATOM_METAL needs macOS")
    endif()
    enable_language(OBJCXX)
    # Shaders are compiled at run time (newLibraryWithSource) so the offline Metal
    # toolchain download is not needed; embed the source as a string.
    set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal)
    set(KERNEL_HDR ${CMAKE_CURRENT_BINARY_DIR}/gen/kernels_src.h)
    add_custom_command(
        OUTPUT ${KERNEL_HDR}
        COMMAND ${CMAKE_COMMAND} -DIN=${KERNEL_SRC} -DOUT=${KERNEL_HDR} -P ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/embed.cmake
        DEPENDS ${KERNEL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/embed.cmake
        COMMENT "Embedding Metal kernels")
    list(APPEND sources src/metal/gpu.mm ${KERNEL_HDR})
endif()

add_executable(slicer ${sources})
target_compile_options(slicer PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3 -march=native>)

if (ATOM_METAL)
    target_compile_definitions(slicer PUBLIC ATOM_METAL=1)
    target_include_directories(slicer PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
    set_source_files_properties(src/metal/gpu.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc -O3 -std=c++17")
    target_link_libraries(slicer PRIVATE "-framework Metal" "-framework Foundation")
endif()

if (ATOM_CUDA)
    target_compile_definitions(slicer PUBLIC ATOM_CUDA=1)
    # --fmad=false: no implicit multiply-add fusion; kernels_common.h writes the CPU's fusions
    # out by hand. No fast math: IEEE division and square root, denormals kept.
    target_compile_options(slicer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -lineinfo>)
endif()
if (ATOM_CUDA_EMU)
    target_compile_definitions(slicer PUBLIC ATOM_CUDA=1 ATOM_CUDA_EMU=1)
    set_source_files_properties(src/cuda/gpu_cuda_emu.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# partition-wip: Partition-only harness (reads an ATOM_PART_DUMP file). PARTITION_SRC lets the
# same harness be built against another partition.cpp (e.g. the unmodified one) for A/B runs.
set(PARTITION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/partition.cpp CACHE FILEPATH "partition.cpp for partbench")
add_executable(partbench tools/partbench.cpp ${PARTITION_SRC} src/parallel.cpp)
get_filename_component(PARTITION_INC ${PARTITION_SRC} DIRECTORY)
target_include_directories(partbench PRIVATE ${PARTITION_INC} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(partbench PUBLIC -Wall -O3 -march=native)
