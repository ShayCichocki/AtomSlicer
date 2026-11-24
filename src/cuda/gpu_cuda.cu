// MIT License
// CUDA implementation of gpu.h (-DATOM_CUDA=ON), and, compiled as plain C++ through
// gpu_cuda_emu.cpp, the CPU emulation of it (-DATOM_CUDA_EMU=ON). Both builds run this same
// file: same buffers, same copies, same launch grid, same red-black phases, same fallbacks.
// Only the launch mechanism differs (ATOM_LAUNCH below).
//
// Memory: explicit device buffers (cudaMalloc + cudaMemcpyAsync), not managed memory. On
// Windows and WSL2 the driver runs managed memory without on-demand paging
// (concurrentManagedAccess = 0), so every kernel launch would migrate whole allocations and
// the CPU could not touch them while a kernel runs. Explicit copies are predictable there:
// a grid level is uploaded once, stays on the device for all of its 2 x N red-black
// half-sweeps (or all its convergence passes), and is downloaded once. Restrict, prolong and
// the init loops stay on the CPU, as in the Metal port.
//
// Fallback: any failure before the result is downloaded leaves the host data untouched, so
// the caller's CPU code then produces exactly the CPU result. A CUDA error other than an
// allocation failure disables the GPU for the rest of the run.

#include "../metal/gpu.h"
#include "kernels_common.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define ATOM_LAUNCH(kern, grid, block, stream, ...) kern<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#define ATOM_BACKEND "cuda"
#else
#include "cuda_emu.h"
#define ATOM_LAUNCH(kern, grid, block, stream, ...) AtomEmu::launch((grid), (block), [&]() { kern(__VA_ARGS__); })
#define ATOM_BACKEND "cuda-emu"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using namespace AtomK;

// ---------------------------------------------------------------- kernels (thin wrappers)
// One thread per cell of the active colour (grid kernels) or per cell (extract, sdf).
#define ATOM_TID (uint32_t(blockIdx.x)*uint32_t(blockDim.x) + uint32_t(threadIdx.x))

__global__ void k_probe(uint32_t *out) { if (ATOM_TID == 0) out[0] = 0x600DCAFEu; }

__global__ void k_normal_smooth(NData *d, GridInfo g) { normalSmooth(d, g, ATOM_TID); }
__global__ void k_normal_smooth_frozen(NData *d, GridInfo g) { normalSmoothFrozen(d, g, ATOM_TID); }
__global__ void k_tri_smooth_layers(TData *d, GridInfo g) { triSmoothLayers(d, g, ATOM_TID); }
__global__ void k_tri_smooth_tangents(TData *d, GridInfo g) { triSmoothTangents(d, g, ATOM_TID); }
__global__ void k_tri_smooth_triphasor(TData *d, GridInfo g) { triSmoothTriphasor(d, g, ATOM_TID); }

// Convergence pass: runs only if the previous pass of this batch changed something, so the
// device stops on exactly the pass where the CPU loop breaks. flags[p] = 1 if pass p changed a cell.
__global__ void k_normal_converge_distance(NData *d, GridInfo g, uint32_t *flags) {
    if (g.pass > 0 && *(volatile uint32_t *)&flags[g.pass - 1] == 0u) return;
    if (normalConvergeDistance(d, g, ATOM_TID) && *(volatile uint32_t *)&flags[g.pass] == 0u)
        atomicExch(&flags[g.pass], 1u);
}

__global__ void k_tri_extract_atoms(const TData *d, GridInfo g, uint32_t first, uint32_t count, float *maxVal, float *maxPos) {
    uint32_t t = ATOM_TID;
    if (t >= count) return;
    triExtractAtoms(d, g, first + t, maxVal, maxPos);
}

__global__ void k_sdf_cells(const float *box, const uint32_t *info, const float *tris, SdfInfo g, float *field, uint32_t first, uint32_t count) {
    uint32_t t = ATOM_TID;
    if (t >= count) return;
    sdfCell(box, info, tris, g, first + t, field);
}

// ---------------------------------------------------------------- host side
namespace {

const unsigned kBlock = 256;

struct KernelInfo { const char *name; size_t stride; bool convergence; };
const KernelInfo kKernels[] = {          // order of Gpu::Kernel
    {"normal_smooth", sizeof(NData), false},
    {"normal_smooth_frozen", sizeof(NData), false},
    {"normal_converge_distance", sizeof(NData), true},
    {"tri_smooth_layers", sizeof(TData), false},
    {"tri_smooth_tangents", sizeof(TData), false},
    {"tri_smooth_triphasor", sizeof(TData), false},
    {"tri_extract_atoms", sizeof(TData), false},
};
static_assert(sizeof(kKernels)/sizeof(kKernels[0]) == size_t(Gpu::Kernel::Count), "kernel table");
static_assert(sizeof(NData) == 20 && sizeof(TData) == 40, "cell layout");

struct DevBuf {                          // grow-only device buffer, reused across calls
    void *p = nullptr;
    size_t bytes = 0;
};

struct Ctx {
    bool ok = false;
    bool verbose = false;
    cudaStream_t stream = nullptr;
    DevBuf grid, aux0, aux1, aux2, aux3;
};

std::mutex gMutex;                       // calls are sequential in AtomSlicer; this only guards misuse

Ctx &ctx();

bool check(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "[" ATOM_BACKEND "] %s: %s\n", what, cudaGetErrorString(e));
    if (e != cudaErrorMemoryAllocation) {
        fprintf(stderr, "[" ATOM_BACKEND "] disabling the GPU for the rest of the run (CPU from here on)\n");
        ctx().ok = false;
    }
    return false;
}

// After a launch: catches bad launch configurations and missing kernel images.
bool launched(const char *what) { return check(cudaGetLastError(), what); }

void *ensure(DevBuf &b, size_t bytes, const char *what) {
    if (b.bytes >= bytes && b.p) return b.p;
    if (b.p) { cudaFree(b.p); b.p = nullptr; b.bytes = 0; }
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes ? bytes : 1);
    if (e != cudaSuccess) {
        cudaGetLastError();              // clear the (non-sticky) allocation error
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        fprintf(stderr, "[" ATOM_BACKEND "] %s: cannot allocate %.1f MB on the device (%.1f MB free), using CPU\n",
                what, bytes/1048576.0, fr/1048576.0);
        return nullptr;
    }
    b.p = p; b.bytes = bytes;
    return p;
}

void release(DevBuf &b) { if (b.p) cudaFree(b.p); b.p = nullptr; b.bytes = 0; }

Ctx &ctx() {
    static Ctx c;
    static std::once_flag once;
    std::call_once(once, [] {
        const char *env = getenv("ATOM_CUDA");
        if (env && !strcmp(env, "0")) return;
        c.verbose = getenv("ATOM_CUDA_VERBOSE") != nullptr;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            cudaGetLastError();
            fprintf(stderr, "[" ATOM_BACKEND "] no CUDA device, using CPU\n");
            return;
        }
        const char *de = getenv("ATOM_CUDA_DEVICE");
        int dev = de ? atoi(de) : 0;
        if (cudaSetDevice(dev) != cudaSuccess) { fprintf(stderr, "[" ATOM_BACKEND "] cannot select device %d, using CPU\n", dev); return; }
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) { fprintf(stderr, "[" ATOM_BACKEND "] no device properties, using CPU\n"); return; }
        if (cudaStreamCreate(&c.stream) != cudaSuccess) { fprintf(stderr, "[" ATOM_BACKEND "] no stream, using CPU\n"); return; }
        // Probe: a binary built for the wrong architecture fails here, not in the middle of a solve.
        uint32_t *dp = nullptr, hv = 0;
        if (cudaMalloc((void **)&dp, sizeof(uint32_t)) != cudaSuccess) { fprintf(stderr, "[" ATOM_BACKEND "] probe allocation failed, using CPU\n"); return; }
        ATOM_LAUNCH(k_probe, dim3(1), dim3(32), c.stream, dp);
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaMemcpyAsync(&hv, dp, sizeof(uint32_t), cudaMemcpyDeviceToHost, c.stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(c.stream);
        cudaFree(dp);
        if (e != cudaSuccess || hv != 0x600DCAFEu) {
            fprintf(stderr, "[" ATOM_BACKEND "] probe kernel failed (%s): was the binary built for compute capability %d.%d? Using CPU\n",
                    e != cudaSuccess ? cudaGetErrorString(e) : "wrong value", prop.major, prop.minor);
            return;
        }
        c.ok = true;
        fprintf(stderr, "[" ATOM_BACKEND "] using %s (compute %d.%d, %.1f GB)\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem/1073741824.0);
    });
    return c;
}

// ATOM_CUDA_SKIP=normal,tri_smooth_layers,sdf ... keeps kernels whose name starts with a listed
// prefix on the CPU. ATOM_CUDA_EXACT=1 keeps the kernels that use cos/sin/atan2 on the CPU
// (CUDA's cosf/sinf/atan2f differ from the host libm by an ulp now and then).
bool skippedName(const std::string &name) {
    static std::vector<std::string> skip = [] {
        std::vector<std::string> v;
        const char *e = getenv("ATOM_CUDA_SKIP");
        std::string s = e ? e : "";
        const char *x = getenv("ATOM_CUDA_EXACT");
        if (x && strcmp(x, "0")) s += ",tri_smooth_layers,tri_smooth_triphasor,tri_extract_atoms";
        size_t a = 0;
        while (a < s.size()) {
            size_t b = s.find(',', a);
            if (b == std::string::npos) b = s.size();
            if (b > a) v.push_back(s.substr(a, b - a));
            a = b + 1;
        }
        return v;
    }();
    for (auto &p : skip) if (name.compare(0, p.size(), p) == 0) return true;
    return false;
}

bool usable(Gpu::Kernel k, const Gpu::GridDesc &g) {
    Ctx &c = ctx();
    if (!c.ok || size_t(k) >= size_t(Gpu::Kernel::Count) || skippedName(kKernels[size_t(k)].name)) return false;
    if (kKernels[size_t(k)].stride != g.cellStride) {
        fprintf(stderr, "[" ATOM_BACKEND "] %s: cell stride %zu != %zu, using CPU\n", kKernels[size_t(k)].name, g.cellStride, kKernels[size_t(k)].stride);
        return false;
    }
    if (g.cellCount >= (size_t(1) << 32)) return false;    // kernels use 32-bit indices
    return true;
}

GridInfo makeInfo(const Gpu::GridDesc &g, const float *params, uint32_t nparams) {
    GridInfo info;
    memset(&info, 0, sizeof(info));
    info.sx = g.size[0]; info.sy = g.size[1]; info.sz = g.size[2];
    info.cellSize = g.cellSize; info.ox = g.origin[0]; info.oy = g.origin[1]; info.oz = g.origin[2];
    float p[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < nparams && i < 4; ++i) p[i] = params[i];
    info.p0 = p[0]; info.p1 = p[1]; info.p2 = p[2]; info.p3 = p[3];
    return info;
}

// Threads for one colour: rows * ceil(sz/2)
size_t colourThreads(const Gpu::GridDesc &g) { return size_t(g.size[0])*g.size[1]*((g.size[2] + 1)/2); }
dim3 blocksFor(size_t threads) { return dim3(unsigned((threads + kBlock - 1)/kBlock)); }

void progressBar(double pct) {
    static int last = -1;
    int v = int(pct*100);
    if (v == last) return;
    last = v;
    const int W = 40; int l = int(pct*W);
    printf("\r%3d%% [%.*s%*s]", v, l, "||||||||||||||||||||||||||||||||||||||||", W-l, "");
    if (pct >= 1.0) { printf("\n"); last = -1; }
    fflush(stdout);
}

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool launchSmooth(Gpu::Kernel k, void *d, const GridInfo &info, dim3 blocks, cudaStream_t s) {
    switch (k) {
    case Gpu::Kernel::NormalSmooth:       ATOM_LAUNCH(k_normal_smooth, blocks, dim3(kBlock), s, (NData *)d, info); break;
    case Gpu::Kernel::NormalSmoothFrozen: ATOM_LAUNCH(k_normal_smooth_frozen, blocks, dim3(kBlock), s, (NData *)d, info); break;
    case Gpu::Kernel::TriSmoothLayers:    ATOM_LAUNCH(k_tri_smooth_layers, blocks, dim3(kBlock), s, (TData *)d, info); break;
    case Gpu::Kernel::TriSmoothTangents:  ATOM_LAUNCH(k_tri_smooth_tangents, blocks, dim3(kBlock), s, (TData *)d, info); break;
    case Gpu::Kernel::TriSmoothTriphasor: ATOM_LAUNCH(k_tri_smooth_triphasor, blocks, dim3(kBlock), s, (TData *)d, info); break;
    default: return false;
    }
    return launched(kKernels[size_t(k)].name);
}

// Download into host memory the caller will use. If this fails the host grid may be half
// written and the CPU fallback would start from garbage, so stop the run instead.
void downloadOrDie(void *dst, const void *src, size_t bytes, const char *what) {
    Ctx &c = ctx();
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, c.stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(c.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[" ATOM_BACKEND "] %s: download failed (%s); host data may be incomplete, stopping\n", what, cudaGetErrorString(e));
        exit(3);
    }
}

} // namespace

bool Gpu::available() { return ctx().ok; }

size_t Gpu::minCells() {
    static size_t m = [] { const char *e = getenv("ATOM_CUDA_MIN_CELLS"); return e ? size_t(atoll(e)) : size_t(32768); }();
    return m;
}

bool Gpu::smooth(Kernel k, const GridDesc &g, const float *params, uint32_t nparams, uint32_t iterationCount, bool progress) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (!usable(k, g) || kKernels[size_t(k)].convergence) return false;
    Ctx &c = ctx();
    auto t0 = std::chrono::steady_clock::now();
    const size_t bytes = g.cellCount*g.cellStride;
    void *d = ensure(c.grid, bytes, kKernels[size_t(k)].name);
    if (!d) return false;
#if defined(ATOM_CUDA_EMU) && defined(ATOM_CUDA_EMU_FAULT)
    // Negative controls for the emulation test only (never in a real build): 1 = forget the
    // upload, 2 = sweep one colour only. Either must make the output differ from the CPU.
    if (ATOM_CUDA_EMU_FAULT != 1)
#endif
    if (!check(cudaMemcpyAsync(d, g.data, bytes, cudaMemcpyHostToDevice, c.stream), "upload")) return false;
    GridInfo info = makeInfo(g, params, nparams);
    const dim3 blocks = blocksFor(colourThreads(g));
    const uint32_t total = 2*iterationCount;
    const uint32_t step = std::max<uint32_t>(2, total/8);
    for (uint32_t it = 0; it < total; ++it) {
        info.parity = it % 2;
#if defined(ATOM_CUDA_EMU) && defined(ATOM_CUDA_EMU_FAULT)
        if (ATOM_CUDA_EMU_FAULT == 2) info.parity = 0;
#endif
        if (!launchSmooth(k, d, info, blocks, c.stream)) return false;
        if (progress && ((it + 1) % step == 0 || it + 1 == total)) {
            if (!check(cudaStreamSynchronize(c.stream), kKernels[size_t(k)].name)) return false;
            progressBar(double(it + 1)/total);
        }
    }
    if (!check(cudaStreamSynchronize(c.stream), kKernels[size_t(k)].name)) return false;   // all sweeps done
    downloadOrDie(g.data, d, bytes, kKernels[size_t(k)].name);
    if (c.verbose)
        fprintf(stderr, "[" ATOM_BACKEND "] %s %ux%ux%u x%u half-sweeps: %.2f ms (incl. copies)\n", kKernels[size_t(k)].name,
                g.size[0], g.size[1], g.size[2], total, msSince(t0));
    return true;
}

long Gpu::smoothConvergence(Kernel k, const GridDesc &g, const float *params, uint32_t nparams) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (!usable(k, g) || !kKernels[size_t(k)].convergence) return -1;
    Ctx &c = ctx();
    auto t0 = std::chrono::steady_clock::now();
    const size_t bytes = g.cellCount*g.cellStride;
    const uint32_t B = 64;                                  // passes per batch, then one flag read-back
    void *d = ensure(c.grid, bytes, kKernels[size_t(k)].name);
    uint32_t *flags = (uint32_t *)ensure(c.aux0, B*sizeof(uint32_t), "convergence flags");
    if (!d || !flags) return -1;
    if (!check(cudaMemcpyAsync(d, g.data, bytes, cudaMemcpyHostToDevice, c.stream), "upload")) return -1;
    GridInfo info = makeInfo(g, params, nparams);
    const dim3 blocks = blocksFor(colourThreads(g));
    uint32_t hostFlags[B];
    long passes = 0;
    for (;;) {
        if (!check(cudaMemsetAsync(flags, 0, B*sizeof(uint32_t), c.stream), "flags reset")) return -1;
        for (uint32_t p = 0; p < B; ++p) {
            info.parity = uint32_t((passes + p) % 2);
            info.pass = p;
            ATOM_LAUNCH(k_normal_converge_distance, blocks, dim3(kBlock), c.stream, (NData *)d, info, flags);
            if (!launched(kKernels[size_t(k)].name)) return -1;
        }
        if (!check(cudaMemcpyAsync(hostFlags, flags, B*sizeof(uint32_t), cudaMemcpyDeviceToHost, c.stream), "flags read")) return -1;
        if (!check(cudaStreamSynchronize(c.stream), kKernels[size_t(k)].name)) return -1;
        for (uint32_t p = 0; p < B; ++p) {
            if (!hostFlags[p]) {
                passes += p + 1;
                downloadOrDie(g.data, d, bytes, kKernels[size_t(k)].name);
                if (c.verbose)
                    fprintf(stderr, "[" ATOM_BACKEND "] %s %ux%ux%u converged after %ld passes: %.2f ms (incl. copies)\n", kKernels[size_t(k)].name,
                            g.size[0], g.size[1], g.size[2], passes, msSince(t0));
                return passes;
            }
        }
        passes += B;
    }
}

bool Gpu::extractAtoms(const GridDesc &g, float layerHeight, float range, float *maxVal, float *maxPos) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (!usable(Kernel::TriExtractAtoms, g)) return false;
    Ctx &c = ctx();
    auto t0 = std::chrono::steady_clock::now();
    const size_t bytes = g.cellCount*g.cellStride;
    void *d = ensure(c.grid, bytes, "tri_extract_atoms");
    float *dv = (float *)ensure(c.aux0, g.cellCount*sizeof(float), "tri_extract_atoms maxVal");
    float *dp = (float *)ensure(c.aux1, g.cellCount*3*sizeof(float), "tri_extract_atoms maxPos");
    if (!d || !dv || !dp) return false;
    if (!check(cudaMemcpyAsync(d, g.data, bytes, cudaMemcpyHostToDevice, c.stream), "upload")) return false;
    const float params[1] = {range};
    GridInfo info = makeInfo(g, params, 1);
    info.k = 2.0*3.14159265358979323846/double(layerHeight);   // CPU: 2.0*M_PI/layerHeight
    // Chunks keep every launch short (the Windows display watchdog kills kernels after ~2 s).
    const size_t chunk = size_t(1) << 22;
    for (size_t first = 0; first < g.cellCount; first += chunk) {
        uint32_t f = uint32_t(first), n = uint32_t(std::min(chunk, g.cellCount - first));
        ATOM_LAUNCH(k_tri_extract_atoms, blocksFor(n), dim3(kBlock), c.stream, (const TData *)d, info, f, n, dv, dp);
        if (!launched("tri_extract_atoms")) return false;
    }
    if (!check(cudaStreamSynchronize(c.stream), "tri_extract_atoms")) return false;
    downloadOrDie(maxVal, dv, g.cellCount*sizeof(float), "tri_extract_atoms");
    downloadOrDie(maxPos, dp, g.cellCount*3*sizeof(float), "tri_extract_atoms");
    release(c.aux0); release(c.aux1);        // one-off outputs: give the VRAM back
    if (c.verbose) fprintf(stderr, "[" ATOM_BACKEND "] tri_extract_atoms %zu cells: %.2f ms (incl. copies)\n", g.cellCount, msSince(t0));
    return true;
}

bool Gpu::sdf(const std::vector<float> &nodeBox, const std::vector<uint32_t> &nodeInfo, const std::vector<float> &triVerts,
              const SdfParams &p, float *field, bool progress, const SdfRayTree * /*rayTree*/) {
    // The optional ray tree (CPU/Metal speed-up) only prunes candidates and gives identical votes;
    // the CUDA kernel keeps the original ray traversal, so results are the same either way.
    std::lock_guard<std::mutex> lock(gMutex);
    Ctx &c = ctx();
    if (!c.ok || skippedName("sdf")) return false;
    const size_t cells = size_t(p.size[0])*p.size[1]*p.size[2];
    if (cells >= (size_t(1) << 32) || nodeBox.empty() || triVerts.empty()) return false;
    auto t0 = std::chrono::steady_clock::now();
    float *dBox = (float *)ensure(c.aux0, nodeBox.size()*sizeof(float), "sdf nodes");
    uint32_t *dInfo = (uint32_t *)ensure(c.aux1, nodeInfo.size()*sizeof(uint32_t), "sdf node info");
    float *dTris = (float *)ensure(c.aux2, triVerts.size()*sizeof(float), "sdf triangles");
    float *dField = (float *)ensure(c.aux3, cells*sizeof(float), "sdf field");
    if (!dBox || !dInfo || !dTris || !dField) return false;
    if (!check(cudaMemcpyAsync(dBox, nodeBox.data(), nodeBox.size()*sizeof(float), cudaMemcpyHostToDevice, c.stream), "upload") ||
        !check(cudaMemcpyAsync(dInfo, nodeInfo.data(), nodeInfo.size()*sizeof(uint32_t), cudaMemcpyHostToDevice, c.stream), "upload") ||
        !check(cudaMemcpyAsync(dTris, triVerts.data(), triVerts.size()*sizeof(float), cudaMemcpyHostToDevice, c.stream), "upload"))
        return false;
    SdfInfo info;
    memset(&info, 0, sizeof(info));
    info.sx = p.size[0]; info.sy = p.size[1]; info.sz = p.size[2]; info.raysCount = p.raysCount;
    info.cellSize = p.cellSize; info.half = p.half; info.ox = p.offset[0]; info.oy = p.offset[1]; info.oz = p.offset[2];
    info.tri0x = p.tri0[0]; info.tri0y = p.tri0[1]; info.tri0z = p.tri0[2];
    // Chunks of 2^20 cells: short launches (display watchdog) and progress steps.
    const size_t chunk = size_t(1) << 20;
    const size_t nChunks = (cells + chunk - 1)/chunk;
    for (size_t k = 0; k < nChunks; ++k) {
        size_t first = k*chunk;
        uint32_t f = uint32_t(first), n = uint32_t(std::min(chunk, cells - first));
        ATOM_LAUNCH(k_sdf_cells, blocksFor(n), dim3(kBlock), c.stream, dBox, dInfo, dTris, info, dField, f, n);
        if (!launched("sdf_cells")) return false;
        if (progress && (k % 8 == 7 || k + 1 == nChunks)) {
            if (!check(cudaStreamSynchronize(c.stream), "sdf_cells")) return false;
            progressBar(double(k + 1)/nChunks);
        }
    }
    if (!check(cudaStreamSynchronize(c.stream), "sdf_cells")) return false;
    downloadOrDie(field, dField, cells*sizeof(float), "sdf_cells");
    release(c.aux0); release(c.aux1); release(c.aux2); release(c.aux3);
    if (c.verbose) fprintf(stderr, "[" ATOM_BACKEND "] sdf_cells %zu cells, %zu tris: %.2f ms (incl. copies)\n", cells, triVerts.size()/9, msSince(t0));
    return true;
}
