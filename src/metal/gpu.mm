// MIT License
// Metal implementation of gpu.h. Objective-C++ using the Metal framework directly.
// Grid storage is the host std::vector memory wrapped with newBufferWithBytesNoCopy
// (MTLResourceStorageModeShared, unified memory): no copies between CPU and GPU, and the
// CPU restrict/prolong steps work on the same memory between GPU calls.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gpu.h"
#include "kernels_src.h"   // generated: static const char *kAtomKernelSource
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>

namespace {

struct GridInfo {           // must match kernels.metal
    uint32_t sx, sy, sz, parity;
    float cellSize, ox, oy, oz;
    float p0, p1, p2, p3;
    uint32_t pass, pad0, pad1, pad2;
};

struct KernelInfo { const char *name; size_t stride; bool convergence; };
const KernelInfo kKernels[] = {
    {"normal_smooth", 20, false},
    {"normal_smooth_frozen", 20, false},
    {"normal_converge_distance", 20, true},
    {"tri_smooth_layers", 40, false},
    {"tri_smooth_tangents", 40, false},
    {"tri_smooth_triphasor", 40, false},
    {"tri_extract_atoms", 40, false},
    {"sdf_cells", 0, false},
};
static_assert(sizeof(kKernels)/sizeof(kKernels[0]) == size_t(Gpu::Kernel::Count) + 1, "kernel table");
const size_t kSdfKernel = size_t(Gpu::Kernel::Count);   // extra pipeline, not a grid smoother
const size_t kPipeCount = size_t(Gpu::Kernel::Count) + 1;

struct Ctx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipes[kPipeCount] = {};
    bool ok = false;
    bool verbose = false;
};

Ctx &ctx() {
    static Ctx c;
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            const char *env = getenv("ATOM_METAL");
            if (env && !strcmp(env, "0")) return;
            c.verbose = getenv("ATOM_METAL_VERBOSE") != nullptr;
            c.device = MTLCreateSystemDefaultDevice();
            if (!c.device) { fprintf(stderr, "[metal] no device, using CPU\n"); return; }
            c.queue = [c.device newCommandQueue];
            MTLCompileOptions *opt = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                opt.mathMode = MTLMathModeSafe;              // IEEE-ish, no fast-math reassociation
                opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            NSError *err = nil;
            c.library = [c.device newLibraryWithSource:[NSString stringWithUTF8String:kAtomKernelSource] options:opt error:&err];
            if (!c.library) {
                fprintf(stderr, "[metal] shader compile failed, using CPU: %s\n", err ? err.localizedDescription.UTF8String : "?");
                return;
            }
            for (size_t k = 0; k < kPipeCount; ++k) {
                id<MTLFunction> fn = [c.library newFunctionWithName:[NSString stringWithUTF8String:kKernels[k].name]];
                if (!fn) { fprintf(stderr, "[metal] missing kernel %s, using CPU\n", kKernels[k].name); return; }
                c.pipes[k] = [c.device newComputePipelineStateWithFunction:fn error:&err];
                if (!c.pipes[k]) { fprintf(stderr, "[metal] pipeline %s failed, using CPU\n", kKernels[k].name); return; }
            }
            c.ok = true;
            fprintf(stderr, "[metal] using %s\n", c.device.name.UTF8String);
        }
    });
    return c;
}

GridInfo makeInfo(const Gpu::GridDesc &g, const float *params, uint32_t nparams) {
    GridInfo info{};
    info.sx = g.size[0]; info.sy = g.size[1]; info.sz = g.size[2];
    info.cellSize = g.cellSize; info.ox = g.origin[0]; info.oy = g.origin[1]; info.oz = g.origin[2];
    float p[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < nparams && i < 4; ++i) p[i] = params[i];
    info.p0 = p[0]; info.p1 = p[1]; info.p2 = p[2]; info.p3 = p[3];
    return info;
}

id<MTLBuffer> wrap(const Gpu::GridDesc &g) {
    return [ctx().device newBufferWithBytesNoCopy:g.data length:Gpu::roundToPage(g.cellCount*g.cellStride)
                                          options:MTLResourceStorageModeShared deallocator:nil];
}

// ATOM_METAL_SKIP=normal,tri_smooth_layers ... keeps kernels whose name starts with a listed
// prefix on the CPU (used to measure each port on its own).
bool skipped(Gpu::Kernel k) {
    static std::vector<std::string> skip = [] {
        std::vector<std::string> v;
        const char *e = getenv("ATOM_METAL_SKIP");
        std::string s = e ? e : "";
        // ATOM_METAL_EXACT=1: keep the kernels that use cos/sin/atan2 on the CPU. Metal's
        // precise:: functions differ from Apple libm by an ulp now and then, which is enough
        // to flip discrete choices downstream; everything else is bit-identical to the CPU.
        const char *x = getenv("ATOM_METAL_EXACT");
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
    std::string name = kKernels[size_t(k)].name;
    for (auto &p : skip) if (name.compare(0, p.size(), p) == 0) return true;
    return false;
}

bool usable(Gpu::Kernel k, const Gpu::GridDesc &g) {
    Ctx &c = ctx();
    if (!c.ok || skipped(k)) return false;
    if (kKernels[size_t(k)].stride != g.cellStride) {
        fprintf(stderr, "[metal] %s: cell stride %zu != %zu, using CPU\n", kKernels[size_t(k)].name, g.cellStride, kKernels[size_t(k)].stride);
        return false;
    }
    if ((uintptr_t(g.data) % Gpu::pageSize()) != 0) return false;
    if (g.cellCount >= (size_t(1) << 32)) return false;    // kernels use 32-bit indices
    return true;
}

// Threads for one colour: rows * ceil(sz/2)
size_t colourThreads(const Gpu::GridDesc &g) {
    return size_t(g.size[0])*g.size[1]*((g.size[2] + 1)/2);
}

void encodeDispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe, size_t threads) {
    NSUInteger tg = std::min<NSUInteger>(pipe.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

bool finish(id<MTLCommandBuffer> cb, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[metal] %s failed: %s\n", what, cb.error ? cb.error.localizedDescription.UTF8String : "?");
        return false;
    }
    return true;
}

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

} // namespace

bool Gpu::available() { return ctx().ok; }

size_t Gpu::minCells() {
    static size_t m = [] { const char *e = getenv("ATOM_METAL_MIN_CELLS"); return e ? size_t(atoll(e)) : size_t(32768); }();
    return m;
}

bool Gpu::smooth(Kernel k, const GridDesc &g, const float *params, uint32_t nparams, uint32_t iterationCount, bool progress) {
    if (!usable(k, g) || kKernels[size_t(k)].convergence) return false;
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        Ctx &c = ctx();
        id<MTLBuffer> buf = wrap(g);
        if (!buf) return false;
        GridInfo info = makeInfo(g, params, nparams);
        size_t threads = colourThreads(g);
        // Split into a few command buffers only to show progress; the queue keeps them in order.
        const uint32_t total = 2*iterationCount;
        // Keep each command buffer to roughly <= 1e9 cell updates (well under a second) so a
        // huge grid never trips the GPU watchdog; also gives the progress bar its steps.
        uint32_t perCB = uint32_t(std::max<size_t>(1, size_t(1000000000) / std::max<size_t>(threads, 1)));
        if (progress) perCB = std::min(perCB, std::max<uint32_t>(2, total/8));
        perCB = std::min(perCB, total);
        for (uint32_t done = 0; done < total;) {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            [enc setComputePipelineState:c.pipes[size_t(k)]];
            [enc setBuffer:buf offset:0 atIndex:0];
            uint32_t end = std::min(total, done + perCB);
            for (uint32_t it = done; it < end; ++it) {
                info.parity = it % 2;
                [enc setBytes:&info length:sizeof(info) atIndex:1];
                encodeDispatch(enc, c.pipes[size_t(k)], threads);
            }
            [enc endEncoding];
            if (!finish(cb, kKernels[size_t(k)].name)) return false;   // partial work: caller must not retry blindly
            done = end;
            if (progress) progressBar(double(done)/total);
        }
    }
    if (ctx().verbose) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[metal] %s %ux%ux%u x%u half-sweeps: %.2f ms\n", kKernels[size_t(k)].name, g.size[0], g.size[1], g.size[2], 2*iterationCount, ms);
    }
    return true;
}

long Gpu::smoothConvergence(Kernel k, const GridDesc &g, const float *params, uint32_t nparams) {
    if (!usable(k, g) || !kKernels[size_t(k)].convergence) return -1;
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        Ctx &c = ctx();
        id<MTLBuffer> buf = wrap(g);
        if (!buf) return -1;
        // passes per command buffer, capped at ~1e9 cell updates per command buffer
        const uint32_t B = uint32_t(std::max<size_t>(1, std::min<size_t>(64, size_t(1000000000) / std::max<size_t>(colourThreads(g), 1))));
        id<MTLBuffer> flags = [c.device newBufferWithLength:B*sizeof(uint32_t) options:MTLResourceStorageModeShared];
        GridInfo info = makeInfo(g, params, nparams);
        size_t threads = colourThreads(g);
        long passes = 0;
        for (;;) {
            memset(flags.contents, 0, B*sizeof(uint32_t));
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            [enc setComputePipelineState:c.pipes[size_t(k)]];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc setBuffer:flags offset:0 atIndex:2];
            for (uint32_t p = 0; p < B; ++p) {
                info.parity = uint32_t((passes + p) % 2);
                info.pass = p;
                [enc setBytes:&info length:sizeof(info) atIndex:1];
                encodeDispatch(enc, c.pipes[size_t(k)], threads);
            }
            [enc endEncoding];
            if (!finish(cb, kKernels[size_t(k)].name)) return -1;
            const uint32_t *f = static_cast<const uint32_t *>(flags.contents);
            for (uint32_t p = 0; p < B; ++p) {
                if (!f[p]) {
                    passes += p + 1;
                    if (c.verbose) fprintf(stderr, "[metal] %s %ux%ux%u converged after %ld passes: %.2f ms\n", kKernels[size_t(k)].name,
                        g.size[0], g.size[1], g.size[2], passes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    return passes;
                }
            }
            passes += B;
        }
    }
}

bool Gpu::extractAtoms(const GridDesc &g, float layerHeight, float range, float *maxVal, float *maxPos) {
    if (!usable(Kernel::TriExtractAtoms, g)) return false;
    if ((uintptr_t(maxVal) % pageSize()) || (uintptr_t(maxPos) % pageSize())) return false;
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        Ctx &c = ctx();
        id<MTLBuffer> buf = wrap(g);
        id<MTLBuffer> bv = [c.device newBufferWithBytesNoCopy:maxVal length:roundToPage(g.cellCount*sizeof(float)) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bp = [c.device newBufferWithBytesNoCopy:maxPos length:roundToPage(g.cellCount*3*sizeof(float)) options:MTLResourceStorageModeShared deallocator:nil];
        if (!buf || !bv || !bp) return false;
        double k = 2.0*M_PI/double(layerHeight);   // as the CPU: 2.0*M_PI/layerHeight in double
        float khi = float(k), klo = float(k - double(khi));
        const float params[3] = {range, khi, klo};
        GridInfo info = makeInfo(g, params, 3);
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLComputePipelineState> pipe = c.pipes[size_t(Kernel::TriExtractAtoms)];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc setBytes:&info length:sizeof(info) atIndex:1];
        [enc setBuffer:bv offset:0 atIndex:2];
        [enc setBuffer:bp offset:0 atIndex:3];
        encodeDispatch(enc, pipe, g.cellCount);
        [enc endEncoding];
        if (!finish(cb, "tri_extract_atoms")) return false;
    }
    if (ctx().verbose) fprintf(stderr, "[metal] tri_extract_atoms %zu cells: %.2f ms\n", g.cellCount,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

struct SdfInfoGpu {   // must match SdfInfo in kernels.metal
    uint32_t sx, sy, sz, raysCount;
    float cellSize, half, ox, oy, oz;
    float tri0x, tri0y, tri0z;
    float eps_hi, eps_lo;
    uint32_t pad;
};

bool Gpu::sdf(const std::vector<float> &nodeBox, const std::vector<uint32_t> &nodeInfo, const std::vector<float> &triVerts,
              const SdfParams &p, float *field, bool progress, const SdfRayTree *rayTree) {
    Ctx &c = ctx();
    if (!c.ok) return false;
    {
        const char *e = getenv("ATOM_METAL_SKIP");
        if (e && strstr(e, "sdf")) return false;
    }
    size_t cells = size_t(p.size[0])*p.size[1]*p.size[2];
    if (cells >= (size_t(1) << 32) || nodeBox.empty() || triVerts.empty()) return false;
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        id<MTLBuffer> bBox = [c.device newBufferWithBytes:nodeBox.data() length:nodeBox.size()*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bInfo = [c.device newBufferWithBytes:nodeInfo.data() length:nodeInfo.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTris = [c.device newBufferWithBytes:triVerts.data() length:triVerts.size()*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bField = [c.device newBufferWithLength:cells*sizeof(float) options:MTLResourceStorageModeShared];
        if (!bBox || !bInfo || !bTris || !bField) return false;
        // Toolpath-speedup change: ray tree buffers (1-element dummies when not used).
        bool useTree = rayTree && !rayTree->info.empty() && !getenv("ATOM_SDF_NO_RAYTREE");
        static const uint32_t dummy[8] = {0};
        auto buf = [&](const void *d, size_t n) { return [c.device newBufferWithBytes:(n ? d : dummy) length:(n ? n : sizeof(dummy)) options:MTLResourceStorageModeShared]; };
        id<MTLBuffer> bRBox = buf(useTree ? rayTree->box.data() : nullptr, useTree ? rayTree->box.size()*4 : 0);
        id<MTLBuffer> bRInfo = buf(useTree ? rayTree->info.data() : nullptr, useTree ? rayTree->info.size()*4 : 0);
        id<MTLBuffer> bRLeaves = buf(useTree ? rayTree->leaves.data() : nullptr, useTree ? rayTree->leaves.size()*4 : 0);
        id<MTLBuffer> bParent = buf(useTree ? rayTree->parent.data() : nullptr, useTree ? rayTree->parent.size()*4 : 0);
        if (!bRBox || !bRInfo || !bRLeaves || !bParent) return false;
        SdfInfoGpu info{};
        info.pad = useTree ? 1u : 0u;
        info.sx = p.size[0]; info.sy = p.size[1]; info.sz = p.size[2]; info.raysCount = p.raysCount;
        info.cellSize = p.cellSize; info.half = p.half; info.ox = p.offset[0]; info.oy = p.offset[1]; info.oz = p.offset[2];
        info.tri0x = p.tri0[0]; info.tri0y = p.tri0[1]; info.tri0z = p.tri0[2];
        info.eps_hi = float(1e-4); info.eps_lo = float(1e-4 - double(info.eps_hi));
        id<MTLComputePipelineState> pipe = c.pipes[kSdfKernel];
        // Chunks keep each command buffer short (GPU watchdog) and drive the progress bar.
        // Toolpath-speedup change: the kernel takes blocks of 4 x 8 rows x 8 z cells (32 threads
        // each), see sdf_cells; a chunk is 4096 blocks (2^20 cells, as before).
        const size_t nb = size_t((p.size[0]+3)/4) * ((p.size[1]+7)/8) * ((p.size[2]+7)/8);
        const size_t chunk = size_t(1) << 12;
        std::vector<id<MTLCommandBuffer>> cbs;
        for (size_t first = 0; first < nb; first += chunk) {
            uint32_t f = uint32_t(first), n = uint32_t(std::min(chunk, nb - first));
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:bBox offset:0 atIndex:0];
            [enc setBuffer:bInfo offset:0 atIndex:1];
            [enc setBuffer:bTris offset:0 atIndex:2];
            [enc setBytes:&info length:sizeof(info) atIndex:3];
            [enc setBuffer:bField offset:0 atIndex:4];
            [enc setBytes:&f length:sizeof(f) atIndex:5];
            [enc setBytes:&n length:sizeof(n) atIndex:6];
            [enc setBuffer:bRBox offset:0 atIndex:7];
            [enc setBuffer:bRInfo offset:0 atIndex:8];
            [enc setBuffer:bRLeaves offset:0 atIndex:9];
            [enc setBuffer:bParent offset:0 atIndex:10];
            NSUInteger tg = std::min<NSUInteger>(pipe.maxTotalThreadsPerThreadgroup, 64);
            tg = tg >= 32 ? tg - tg % 32 : tg;
            [enc dispatchThreads:MTLSizeMake(size_t(n)*32, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            cbs.push_back(cb);
        }
        for (size_t k = 0; k < cbs.size(); ++k) {
            [cbs[k] waitUntilCompleted];
            if (cbs[k].status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "[metal] sdf_cells failed: %s\n", cbs[k].error ? cbs[k].error.localizedDescription.UTF8String : "?");
                return false;
            }
            if (progress) progressBar(double(k+1)/cbs.size());
        }
        memcpy(field, bField.contents, cells*sizeof(float));
    }
    if (c.verbose) fprintf(stderr, "[metal] sdf_cells %zu cells, %zu tris: %.2f ms\n", cells, triVerts.size()/9,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}
