// MIT License
// GPU backend interface for the AtomSlicer grid smoothers.
// Implemented by the Metal backend (-DATOM_METAL=ON, src/metal/gpu.mm) or the CUDA backend
// (-DATOM_CUDA=ON, src/cuda/gpu_cuda.cu; -DATOM_CUDA_EMU=ON runs that same code on the CPU).
// Without either, every call reports "not available" and the callers run their original CPU
// code. At run time ATOM_METAL=0 / ATOM_CUDA=0 in the environment also forces the CPU path.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>
#ifdef ATOM_METAL
#include <unistd.h>
#endif

#if defined(ATOM_METAL) || defined(ATOM_CUDA)
#define ATOM_GPU 1
#endif

namespace Gpu {

enum class Kernel : uint32_t {
    NormalSmooth = 0,
    NormalSmoothFrozen,
    NormalConvergeDistance,
    TriSmoothLayers,
    TriSmoothTangents,      // p0 = 0: smoothTangents, p0 = 1: smoothTangentsAll
    TriSmoothTriphasor,
    TriExtractAtoms,
    Count
};

struct GridDesc {
    void *data;             // cell array (page aligned, see PageAllocator)
    size_t cellCount;
    size_t cellStride;      // sizeof(Data); checked against the kernel's struct
    uint32_t size[3];
    float cellSize;
    float origin[3];
};

#ifdef ATOM_GPU
bool available();
// 2*iterationCount red-black half-sweeps (parity 0,1,0,1,...), as GridHierarchy::smooth.
bool smooth(Kernel k, const GridDesc &g, const float *params, uint32_t nparams, uint32_t iterationCount, bool progress);
// Red-black passes until one pass changes nothing, as GridHierarchy::smoothConvergence.
// Returns the number of passes run, or -1 if the GPU path is not usable.
long smoothConvergence(Kernel k, const GridDesc &g, const float *params, uint32_t nparams);
// Minimum cell count for which the GPU is used (small coarse levels stay on the CPU).
size_t minCells();
// TriPhasor::extractAtoms first loop. maxVal (floats) and maxPos (3 floats per cell) must be
// page aligned and pre-filled with -inf / NaN like the CPU code.
bool extractAtoms(const GridDesc &g, float layerHeight, float range, float *maxVal, float *maxPos);
struct SdfParams {
    uint32_t size[3], raysCount;
    float cellSize, half, offset[3];
    float tri0[3];
};
// Toolpath-speedup change: optional ray tree (BVHTriangle::flattenRayTree); empty = old traversal.
struct SdfRayTree {
    std::vector<float> box;          // 6 floats per ray node
    std::vector<uint32_t> info;      // first, count per ray node (count > 0: leaf)
    std::vector<uint32_t> leaves;    // original leaf node ids
    std::vector<uint32_t> parent;    // original tree parents
};
// SDF constructor loop on the GPU. Writes field[0..cells); a cell whose BVH stack overflowed
// is NaN and must be recomputed on the CPU by the caller.
bool sdf(const std::vector<float> &nodeBox, const std::vector<uint32_t> &nodeInfo, const std::vector<float> &triVerts,
         const SdfParams &p, float *field, bool progress, const SdfRayTree *rayTree = nullptr);
#else
struct SdfParams { uint32_t size[3], raysCount; float cellSize, half, offset[3]; float tri0[3]; };
struct SdfRayTree { std::vector<float> box; std::vector<uint32_t> info, leaves, parent; };
inline bool sdf(const std::vector<float> &, const std::vector<uint32_t> &, const std::vector<float> &, const SdfParams &, float *, bool, const SdfRayTree * = nullptr) { return false; }
inline bool extractAtoms(const GridDesc &, float, float, float *, float *) { return false; }
inline bool available() { return false; }
inline bool smooth(Kernel, const GridDesc &, const float *, uint32_t, uint32_t, bool) { return false; }
inline long smoothConvergence(Kernel, const GridDesc &, const float *, uint32_t) { return -1; }
inline size_t minCells() { return size_t(-1); }
#endif

#ifdef ATOM_METAL
// Page-aligned allocator so std::vector storage can be wrapped as a shared MTLBuffer
// without a copy (newBufferWithBytesNoCopy needs page alignment and page-multiple length).
inline size_t pageSize() { static size_t p = size_t(getpagesize()); return p; }
inline size_t roundToPage(size_t bytes) { size_t p = pageSize(); return (bytes + p - 1) / p * p; }

template <class T>
struct PageAllocator {
    using value_type = T;
    PageAllocator() = default;
    template <class U> PageAllocator(const PageAllocator<U> &) {}
    T *allocate(size_t n) {
        void *p = nullptr;
        if (posix_memalign(&p, pageSize(), roundToPage(n*sizeof(T) ? n*sizeof(T) : 1)) != 0) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T *p, size_t) { free(p); }
    template <class U> bool operator==(const PageAllocator<U> &) const { return true; }
    template <class U> bool operator!=(const PageAllocator<U> &) const { return false; }
};

#endif

}
