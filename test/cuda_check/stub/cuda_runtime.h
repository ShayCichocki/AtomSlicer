// Minimal stand-in for the CUDA toolkit's cuda_runtime.h, ONLY for the clang self-check in
// check_cuda.sh on a machine without the toolkit. Declarations copy the real CUDA 12 runtime
// API signatures for the calls gpu_cuda.cu makes; nothing here is ever linked or run.
#pragma once
#include <stddef.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 12040
#endif
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

#include <math.h>
#include <stdint.h>
#include <limits.h>

struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>

enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNoDevice = 100,
    cudaErrorNoKernelImageForDevice = 209,
};
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
typedef struct CUstream_st *cudaStream_t;

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major, minor;
    int multiProcessorCount;
    // (the real struct has many more fields)
};

extern "C" {
__host__ cudaError_t cudaGetLastError(void);
__host__ const char *cudaGetErrorString(cudaError_t error);
__host__ cudaError_t cudaGetDeviceCount(int *count);
__host__ cudaError_t cudaSetDevice(int device);
__host__ cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
__host__ cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
__host__ cudaError_t cudaStreamCreate(cudaStream_t *pStream);
__host__ cudaError_t cudaStreamSynchronize(cudaStream_t stream);
__host__ cudaError_t cudaDeviceSynchronize(void);
__host__ cudaError_t cudaMalloc(void **devPtr, size_t size);
__host__ cudaError_t cudaFree(void *devPtr);
__host__ cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
__host__ cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream = 0);
// what clang lowers kernel<<<g, b, s, st>>>(...) to (CUDA >= 9.2 launch ABI)
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, struct CUstream_st *stream = 0);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
}

__device__ __forceinline__ unsigned int atomicExch(unsigned int *address, unsigned int val) {
    return __nvvm_atom_xchg_gen_i((int *)address, (int)val);
}
