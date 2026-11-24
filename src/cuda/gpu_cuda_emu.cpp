// MIT License
// ATOM_CUDA_EMU build: compiles the CUDA backend (gpu_cuda.cu, host code and kernels) as plain
// C++ against the CPU runtime emulation in cuda_emu.h. Compile with -ffp-contract=off, the
// counterpart of nvcc --fmad=false (CMakeLists.txt sets both).
#ifdef __CUDACC__
#error "gpu_cuda_emu.cpp is the CPU emulation; the CUDA build compiles gpu_cuda.cu directly"
#endif
#include "gpu_cuda.cu"
