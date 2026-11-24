cmake_minimum_required(VERSION 3.10...4.1)
project(slicer)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# Metal port: -DATOM_METAL=ON runs the grid smoothers on the Apple GPU (macOS only).
# Without it the build is plain C++17 as upstream. ATOM_METAL=0 in the environment forces
# the CPU path at run time.
option(ATOM_METAL "Run grid smoothers with Metal compute (macOS)" OFF)

file(GLOB_RECURSE sources      src/*.cpp src/*.h)

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
