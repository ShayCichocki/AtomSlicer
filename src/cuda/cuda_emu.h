// MIT License
// CPU emulation of the small part of the CUDA runtime that gpu_cuda.cu uses (ATOM_CUDA_EMU).
// It lets the exact CUDA host code and kernels run on a machine without an NVIDIA GPU:
//   - "device" memory is a separate heap allocation filled with 0xFF bytes (NaN floats,
//     flags = outside|frozen), so a missing host->device copy shows up as wrong output;
//   - host<->device copies are real memcpy calls between the two heaps;
//   - a kernel launch runs every block of the grid (blocks in parallel on the thread pool,
//     threads of a block one after another) with blockIdx/threadIdx/blockDim/gridDim set
//     as thread-local variables, so the kernels read them exactly as on the GPU;
//   - everything is synchronous (a launch returns when the kernel has finished).
// Nothing here is used by the real CUDA build.
#pragma once
#include "../parallel.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

typedef int cudaError_t;
enum : int { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 1, cudaErrorNoDevice = 100 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
typedef struct AtomEmuStream *cudaStream_t;

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint3 { unsigned x, y, z; };

static thread_local uint3 blockIdx, threadIdx;
static thread_local dim3 blockDim, gridDim;

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major, minor;
    int multiProcessorCount;
};

inline const char *cudaGetErrorString(cudaError_t e) {
    switch (e) {
    case cudaSuccess: return "no error";
    case cudaErrorMemoryAllocation: return "out of memory (emulated)";
    case cudaErrorNoDevice: return "no device (emulated)";
    default: return "invalid value (emulated)";
    }
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int d) { return d == 0 ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) {
    std::memset(p, 0, sizeof(*p));
    std::strcpy(p->name, "CPU emulation (ATOM_CUDA_EMU)");
    p->totalGlobalMem = size_t(24) << 30;   // pretend 24 GB, like the target RTX 4090
    p->major = 8; p->minor = 9;
    p->multiProcessorCount = 1;
    return cudaSuccess;
}
inline cudaError_t cudaMemGetInfo(size_t *freeB, size_t *total) { *freeB = size_t(24) << 30; *total = size_t(24) << 30; return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

inline cudaError_t cudaMalloc(void **p, size_t bytes) {
    if (bytes == 0) bytes = 1;
    void *q = std::malloc(bytes);
    if (!q) { *p = nullptr; return cudaErrorMemoryAllocation; }
    std::memset(q, 0xFF, bytes);   // poison: device memory is never what the host had
    *p = q;
    return cudaSuccess;
}
inline cudaError_t cudaFree(void *p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t bytes, cudaMemcpyKind, cudaStream_t) {
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void *dst, int v, size_t bytes, cudaStream_t) {
    std::memset(dst, v, bytes);
    return cudaSuccess;
}

inline unsigned atomicExch(unsigned *addr, unsigned v) { return __atomic_exchange_n(addr, v, __ATOMIC_RELAXED); }

namespace AtomEmu {
// Runs kernel body f once per (block, thread) of a 1-D launch.
template <class F>
void launch(dim3 grid, dim3 block, const F &f) {
    Parallel::For(0, size_t(grid.x), [&](size_t b) {
        gridDim = grid;
        blockDim = block;
        blockIdx.x = unsigned(b); blockIdx.y = 0; blockIdx.z = 0;
        for (unsigned t = 0; t < block.x; ++t) {
            threadIdx.x = t; threadIdx.y = 0; threadIdx.z = 0;
            f();
        }
    });
}
}
