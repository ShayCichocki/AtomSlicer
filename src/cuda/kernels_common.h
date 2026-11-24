// MIT License
// Per-cell math of the CUDA backend, shared by the real CUDA kernels (gpu_cuda.cu compiled by
// nvcc) and by the CPU emulation build (ATOM_CUDA_EMU, the same gpu_cuda.cu compiled as C++).
//
// Every function here is a line-by-line port of the CPU code it names (and of the Metal
// kernel in src/metal/kernels.metal, which was proven bit-identical where it has no
// transcendental functions). The multiply-adds that Apple clang fuses in the CPU code
// (-ffp-contract=on: a*b + c inside one source expression) are written out with fmaf/fma;
// the file must be compiled with contraction OFF (nvcc --fmad=false, clang -ffp-contract=off)
// so the compiler adds no other fusions.
//
// Rules for this file (it must compile as CUDA device code and as plain C++):
//   - no std:: calls, no exceptions, no virtuals, no recursion, no dynamic allocation;
//   - only the C math functions sqrtf, fabsf, fmaf, fma, cosf, sinf, atan2f (and memcpy);
//   - 32-bit cell indices (the host refuses grids of 2^32 cells or more).
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__CUDACC__)
#define ATOM_HD __host__ __device__ __forceinline__
#else
#define ATOM_HD inline
#endif

namespace AtomK {

constexpr uint8_t FROZEN = 1u;
constexpr uint8_t OUTSIDE = 2u;

// Same layout as NormalField::Data and TriPhasor::Data (the host checks sizeof against the
// grid's cell stride before every call).
struct NData {
    float normal[3];
    float distance;
    uint8_t flags;
};
struct TData {
    float normal[3];
    float tangent[3];
    float phase[3];
    uint8_t flags;
};

// Kernel parameters, passed by value to every grid kernel.
struct GridInfo {
    uint32_t sx, sy, sz;    // grid size
    uint32_t parity;        // active colour: (x+y+z) % 2 == parity
    float cellSize;
    float ox, oy, oz;       // origin
    float p0, p1, p2, p3;   // kernel parameters (layerHeight, ...)
    uint32_t pass;          // pass index inside a convergence batch
    uint32_t pad0;
    double k;               // extractAtoms: 2*pi/layerHeight in double, as the CPU
};

struct SdfInfo {
    uint32_t sx, sy, sz, raysCount;
    float cellSize, half, ox, oy, oz;   // point = ((cellSize*c) + half) + offset
    float tri0x, tri0y, tri0z;          // tris[0].a, the CPU's initial "closest"
};

// Special values by bit pattern (the INFINITY / NAN macros expand to compiler builtins that
// not every device compiler accepts). Same bits as std::numeric_limits<float> on x86 and ARM.
ATOM_HD float bitsToFloat(uint32_t u) { float f; memcpy(&f, &u, sizeof(f)); return f; }
ATOM_HD float posInf() { return bitsToFloat(0x7f800000u); }
ATOM_HD float negInf() { return bitsToFloat(0xff800000u); }
ATOM_HD float qNaN() { return bitsToFloat(0x7fc00000u); }

// ---------------------------------------------------------------- Vec3 (mathutils.h)
struct V3 { float x, y, z; };

ATOM_HD V3 v3(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
ATOM_HD V3 v3load(const float *p) { return v3(p[0], p[1], p[2]); }
ATOM_HD void v3store(float *p, V3 a) { p[0] = a.x; p[1] = a.y; p[2] = a.z; }
ATOM_HD V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
ATOM_HD V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
ATOM_HD V3 vscale(V3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }          // Vec3 * s
// x*x' + y*y' + z*z'  ->  fma(z, z', fma(x, x', y*y'))
ATOM_HD float vdot(V3 a, V3 b) { return fmaf(a.z, b.z, fmaf(a.x, b.x, a.y*b.y)); }
ATOM_HD float vlen(V3 a) { return sqrtf(vdot(a, a)); }
ATOM_HD V3 vnormalize(V3 a) {
    float l = vlen(a);
    V3 r = v3(a.x/l, a.y/l, a.z/l);
    return l > 0 ? r : v3(1, 0, 0);
}
// y*b.z - z*b.y -> fma(y, b.z, -(z*b.y)), same for the others
ATOM_HD V3 vcross(V3 a, V3 b) {
    return v3(fmaf(a.y, b.z, -(a.z*b.y)), -fmaf(a.x, b.z, -(a.z*b.x)), fmaf(a.x, b.y, -(a.y*b.x)));
}
ATOM_HD V3 vorthonormalize(V3 a, V3 n) {
    V3 proj = vscale(n, vdot(a, n));
    return vnormalize(vsub(a, proj));
}

#define ATOM_PI_F float(3.14159265358979323846)
#define ATOM_PI_D 3.14159265358979323846

// ---------------------------------------------------------------- grid helpers
// Thread id -> cell index of the active colour; false when past the end (GridHierarchy::forColour).
ATOM_HD bool cellOf(uint32_t tid, const GridInfo &g, uint32_t c[3], uint32_t &i) {
    uint32_t halfZ = (g.sz + 1) / 2;
    uint32_t row = tid / halfZ;
    uint32_t k = tid - row*halfZ;
    if (row >= g.sx*g.sy) return false;
    uint32_t x = row / g.sy, y = row - x*g.sy;
    uint32_t z = 2*k + ((x + y + g.parity) & 1u);
    if (z >= g.sz) return false;
    c[0] = x; c[1] = y; c[2] = z;
    i = (x*g.sy + y)*g.sz + z;
    return true;
}

// GridHierarchy::getPosition: cellSize*Vec3(c+0.5) + origin
ATOM_HD V3 cellPos(const uint32_t c[3], const GridInfo &g) {
    V3 v = v3(float(c[0]) + 0.5f, float(c[1]) + 0.5f, float(c[2]) + 0.5f);
    V3 s = v3(g.cellSize*v.x, g.cellSize*v.y, g.cellSize*v.z);
    return v3(s.x + g.ox, s.y + g.oy, s.z + g.oz);
}

// Neighbour number dir (0..5) in the CPU order +x, -x, +y, -y, +z, -z (NeighborhoodIter).
// False when it is off the grid. Loops over dir = 0..5 with a constant trip count, so the
// compiler unrolls them and nothing ends up in local memory.
ATOM_HD bool neighbour(const uint32_t c[3], uint32_t i, const GridInfo &g, uint32_t dir, uint32_t &j, uint32_t nc[3]) {
    nc[0] = c[0]; nc[1] = c[1]; nc[2] = c[2];
    switch (dir) {
    case 0: if (c[0] >= g.sx-1) return false; j = i + g.sy*g.sz; nc[0] = c[0]+1; return true;
    case 1: if (c[0] == 0) return false;      j = i - g.sy*g.sz; nc[0] = c[0]-1; return true;
    case 2: if (c[1] >= g.sy-1) return false; j = i + g.sz;      nc[1] = c[1]+1; return true;
    case 3: if (c[1] == 0) return false;      j = i - g.sz;      nc[1] = c[1]-1; return true;
    case 4: if (c[2] >= g.sz-1) return false; j = i + 1;         nc[2] = c[2]+1; return true;
    default: if (c[2] == 0) return false;     j = i - 1;         nc[2] = c[2]-1; return true;
    }
}

#if defined(__CUDACC__)
#define ATOM_UNROLL _Pragma("unroll")
#else
#define ATOM_UNROLL
#endif

// mathutils.h alignPhases with w = 1
ATOM_HD void alignPhases(V3 p_i, V3 p_j, V3 d_i, V3 d_j, float f_i, float f_j, float phi_j, float &x, float &y) {
    V3 s = vadd(p_i, p_j);
    V3 p_ij = v3(s.x*0.5f, s.y*0.5f, s.z*0.5f);
    float a = 2.0f*ATOM_PI_F*f_i * vdot(vsub(p_ij, p_i), d_i);
    float b = fmaf(2.0f*ATOM_PI_F*f_j, vdot(vsub(p_ij, p_j), d_j), phi_j);
    float dd = vdot(d_i, d_j);
    float angle = dd > 0.0f ? b - a : -b - a;
    x = fmaf(fabsf(dd), cosf(angle), x);
    y = fmaf(fabsf(dd), sinf(angle), y);
}

// ---------------------------------------------------------------- NormalField
// NormalField::smooth
ATOM_HD void normalSmooth(NData *d, const GridInfo &g, uint32_t tid) {
    uint32_t c[3], i;
    if (!cellOf(tid, g, c, i)) return;
    uint8_t f = d[i].flags;
    if ((f & OUTSIDE) || (f & FROZEN)) return;
    V3 norm = v3(0, 0, 0);
    ATOM_UNROLL
    for (uint32_t dir = 0; dir < 6; ++dir) {
        uint32_t j, nc[3];
        if (!neighbour(c, i, g, dir, j, nc)) continue;
        if (d[j].flags & OUTSIDE) continue;
        norm = vadd(norm, v3load(d[j].normal));
    }
    v3store(d[i].normal, vnormalize(norm));
}

// NormalField::smoothFrozen
ATOM_HD void normalSmoothFrozen(NData *d, const GridInfo &g, uint32_t tid) {
    uint32_t c[3], i;
    if (!cellOf(tid, g, c, i)) return;
    uint8_t f = d[i].flags;
    if ((f & OUTSIDE) || !(f & FROZEN)) return;
    V3 norm = v3(0, 0, 0);
    ATOM_UNROLL
    for (uint32_t dir = 0; dir < 6; ++dir) {
        uint32_t j, nc[3];
        if (!neighbour(c, i, g, dir, j, nc)) continue;
        uint8_t fn = d[j].flags;
        if ((fn & OUTSIDE) || !(fn & FROZEN)) continue;
        norm = vadd(norm, v3load(d[j].normal));
    }
    // CPU: norm.length2() > 1e-4 (float compared as double; 1e-4 is not a float, so compare in double)
    if (double(vdot(norm, norm)) > 1e-4) {
        v3store(d[i].normal, vnormalize(norm));
    }
}

// NormalField constructor (ConformalTop / ConformalNearest): the smoothConvergence lambda.
// Returns true when the cell changed.
ATOM_HD bool normalConvergeDistance(NData *d, const GridInfo &g, uint32_t tid) {
    uint32_t c[3], i;
    if (!cellOf(tid, g, c, i)) return false;
    if (d[i].flags & OUTSIDE) return false;
    V3 pi = cellPos(c, g);
    float di = d[i].distance;
    V3 ni = v3load(d[i].normal);
    bool changed = false;
    ATOM_UNROLL
    for (uint32_t dir = 0; dir < 6; ++dir) {
        uint32_t j, nc[3];
        if (!neighbour(c, i, g, dir, j, nc)) continue;
        if (d[j].flags & OUTSIDE) continue;
        float distAB = vlen(vsub(pi, cellPos(nc, g)));
        float dn = d[j].distance;
        if (di > dn + distAB) {
            di = dn + distAB;
            ni = v3load(d[j].normal);
            changed = true;
        }
    }
    if (changed) {
        d[i].distance = di;
        v3store(d[i].normal, ni);
    }
    return changed;
}

// ---------------------------------------------------------------- TriPhasor
// TriPhasor::smoothLayers, p0 = layerHeight
ATOM_HD void triSmoothLayers(TData *d, const GridInfo &g, uint32_t tid) {
    uint32_t c[3], i;
    if (!cellOf(tid, g, c, i)) return;
    uint8_t f = d[i].flags;
    if ((f & OUTSIDE) || (f & FROZEN)) return;
    float x = 0, y = 0;
    V3 pi = cellPos(c, g);
    V3 d_i = v3load(d[i].normal);
    float fr = 1.0f/g.p0;
    ATOM_UNROLL
    for (uint32_t dir = 0; dir < 6; ++dir) {
        uint32_t j, nc[3];
        if (!neighbour(c, i, g, dir, j, nc)) continue;
        if (d[j].flags & OUTSIDE) continue;
        V3 d_j = v3load(d[j].normal);
        alignPhases(pi, cellPos(nc, g), d_i, d_j, fr, fr, d[j].phase[2], x, y);
    }
    d[i].phase[2] = atan2f(y, x);
}

// TriPhasor::smoothTangents (p0 = 0) and smoothTangentsAll (p0 = 1)
ATOM_HD void triSmoothTangents(TData *d, const GridInfo &g, uint32_t tid) {
    uint32_t c[3], i;
    if (!cellOf(tid, g, c, i)) return;
    uint8_t f = d[i].flags;
    if (g.p0 == 0.0f && ((f & OUTSIDE) || (f & FROZEN))) return;
    V3 tang = v3(0, 0, 0);
    ATOM_UNROLL
    for (uint32_t dir = 0; dir < 6; ++dir) {
        uint32_t j, nc[3];
        if (!neighbour(c, i, g, dir, j, nc)) continue;
        if (d[j].flags & OUTSIDE) continue;
        V3 t = v3load(d[j].tangent);
        float s = vdot(tang, t) < 0 ? -1.0f : 1.0f;
        tang = vadd(tang, v3(t.x*s, t.y*s, t.z*s));
    }
    v3store(d[i].tangent, vorthonormalize(tang, v3load(d[i].normal)));
}

// TriPhasor::smoothTriphasor, p0 = layerHeight
ATOM_HD void triSmoothTriphasor(TData *d, const GridInfo &g, uint32_t tid) {
    uint32_t c[3], i;
    if (!cellOf(tid, g, c, i)) return;
    uint8_t f = d[i].flags;
    if (f & OUTSIDE) return;
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    V3 pi = cellPos(c, g);
    V3 d_i = v3load(d[i].tangent);
    V3 d_i_b = vnormalize(vcross(v3load(d[i].normal), d_i));
    float ph2 = d[i].phase[2];
    float fr = 1.0f/g.p0;
    ATOM_UNROLL
    for (uint32_t dir = 0; dir < 6; ++dir) {
        uint32_t j, nc[3];
        if (!neighbour(c, i, g, dir, j, nc)) continue;
        if (d[j].flags & OUTSIDE) continue;
        // CPU: fabsf(a - b) > M_PI, a float promoted to double
        if (double(fabsf(ph2 - d[j].phase[2])) > ATOM_PI_D) continue;
        V3 pj = cellPos(nc, g);
        V3 d_j = v3load(d[j].tangent);
        V3 d_j_b = vnormalize(vcross(v3load(d[j].normal), d_j));
        float phj0 = d[j].phase[0], phj1 = d[j].phase[1];
        alignPhases(pi, pj, d_i, d_j, fr, fr, phj0, x0, y0);
        alignPhases(pi, pj, d_i_b, d_j, fr, fr, phj0, x1, y1);
        alignPhases(pi, pj, d_i, d_j_b, fr, fr, phj1, x0, y0);
        alignPhases(pi, pj, d_i_b, d_j_b, fr, fr, phj1, x1, y1);
    }
    d[i].phase[0] = atan2f(y0, x0);
    if (!(f & FROZEN)) {
        d[i].phase[1] = atan2f(y1, x1);
    }
}

// TriPhasor::extractAtoms, first loop: best of 8^3 sub-cell samples. One thread per cell
// (cell = first + tid). p0 = range, g.k = 2*pi/layerHeight (double, as the CPU). Writes every
// cell of the range: outside cells get the CPU's initial -inf / NaN.
ATOM_HD void triExtractAtoms(const TData *d, const GridInfo &g, uint32_t i, float *maxVal, float *maxPos) {
    float best = negInf();
    V3 bestPos = v3(qNaN(), qNaN(), qNaN());
    if (!(d[i].flags & OUTSIDE)) {
        uint32_t syz = g.sy*g.sz;
        uint32_t c[3];
        c[0] = i / syz; uint32_t r = i - c[0]*syz; c[1] = r / g.sz; c[2] = r - c[1]*g.sz;
        V3 base = cellPos(c, g);
        V3 dirT = v3load(d[i].tangent);
        V3 dirN = v3load(d[i].normal);
        V3 dirB = vnormalize(vcross(dirN, dirT));
        float phaseT = d[i].phase[0], phaseB = d[i].phase[1], phaseN = d[i].phase[2];
        float range = g.p0;
        double k = g.k;
        for (uint32_t a = 0; a < 8; ++a) {
            float fx = 2*(float(a)/7.0f)-1;
            for (uint32_t b = 0; b < 8; ++b) {
                float fy = 2*(float(b)/7.0f)-1;
                for (uint32_t cc = 0; cc < 8; ++cc) {
                    float fz = 2*(float(cc)/7.0f)-1;
                    V3 pos = v3(fx*range, fy*range, fz*range);
                    // CPU: cosf(2.0*M_PI/layerHeight*dir.dot(pos)+phase): one double expression,
                    // contracted to fma(k, dot, phase) in double, then rounded to float.
                    float valT = cosf(float(fma(k, double(vdot(dirT, pos)), double(phaseT))));
                    float valB = cosf(float(fma(k, double(vdot(dirB, pos)), double(phaseB))));
                    float valN = cosf(float(fma(k, double(vdot(dirN, pos)), double(phaseN))));
                    float v = valT + valB;
                    v = v + valN;
                    if (v > best) {
                        best = v;
                        bestPos = v3(base.x + pos.x, base.y + pos.y, base.z + pos.z);
                    }
                }
            }
        }
    }
    maxVal[i] = best;
    maxPos[3*i+0] = bestPos.x; maxPos[3*i+1] = bestPos.y; maxPos[3*i+2] = bestPos.z;
}

// ---------------------------------------------------------------- SDF (bvhtriangle.h)
ATOM_HD float stdmin(float a, float b) { return (b < a) ? b : a; }   // std::min
ATOM_HD float stdmax(float a, float b) { return (a < b) ? b : a; }   // std::max
ATOM_HD float comp(V3 a, uint32_t k) { return k == 0 ? a.x : (k == 1 ? a.y : a.z); }
ATOM_HD void setComp(V3 &a, uint32_t k, float v) { if (k == 0) a.x = v; else if (k == 1) a.y = v; else a.z = v; }

ATOM_HD V3 nodeMin(const float *box, uint32_t n) { return v3(box[6*n], box[6*n+1], box[6*n+2]); }
ATOM_HD V3 nodeMax(const float *box, uint32_t n) { return v3(box[6*n+3], box[6*n+4], box[6*n+5]); }
ATOM_HD V3 triV(const float *t, uint32_t k, uint32_t j) { return v3(t[9*k+3*j], t[9*k+3*j+1], t[9*k+3*j+2]); }

// mathutils.h randomUintFromUint / randomFloatFromUint / randomDirectionFromUint
ATOM_HD uint32_t rngUint(uint32_t v) {
    uint32_t state = 747796405u*v + 2891336453u;
    uint32_t word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
ATOM_HD float rngFloat(uint32_t v) { return float(rngUint(v)) / 4294967296.0f; }   // float(UINT32_MAX) == 2^32
ATOM_HD V3 rngDir(uint32_t v) {
    V3 r = v3(rngFloat(v), rngFloat(13u*v+5u), rngFloat(17u*v+7u));
    V3 t = v3(2*r.x, 2*r.y, 2*r.z);
    return vnormalize(v3(t.x - 1, t.y - 1, t.z - 1));
}

ATOM_HD bool pointInAABB(V3 p, V3 mn, V3 mx) {
    for (uint32_t a = 0; a < 3; ++a) if (comp(p, a) < comp(mn, a) || comp(p, a) > comp(mx, a)) return false;
    return true;
}
// dist2 += (m > 0)*m*m + (M > 0)*M*M  ->  dist2 + fma((m>0)*m, m, (M>0)*M*M)
ATOM_HD bool sphereAABB(V3 c, float r2, V3 mn, V3 mx) {
    float dist2 = 0.0f;
    for (uint32_t a = 0; a < 3; ++a) {
        float m = comp(mn, a)-comp(c, a), M = comp(c, a)-comp(mx, a);
        dist2 += fmaf(float(m > 0)*m, m, float(M > 0)*M*M);
    }
    return dist2 <= r2;
}
ATOM_HD float pointAABBDist2(V3 p, V3 mn, V3 mx) {
    V3 cl = p;
    for (uint32_t a = 0; a < 3; ++a) {
        if (comp(p, a) < comp(mn, a)) setComp(cl, a, comp(mn, a));
        else if (comp(p, a) > comp(mx, a)) setComp(cl, a, comp(mx, a));
    }
    V3 dd = vsub(p, cl);
    return vdot(dd, dd);
}
ATOM_HD V3 closestOnSegment(V3 ds, V3 edge, V3 start, V3 end) {
    float l = vdot(ds, edge);
    float s = vdot(edge, edge);
    if (l > 0 && l < s) { float q = l/s; return vadd(start, v3(edge.x*q, edge.y*q, edge.z*q)); }
    return l <= 0 ? start : end;
}
ATOM_HD V3 bestOfTwo(V3 p, V3 a, V3 b) {
    V3 da = vsub(p, a), db = vsub(p, b);
    return vdot(da, da) < vdot(db, db) ? a : b;
}
ATOM_HD V3 closestOnTriangle(V3 p, V3 a, V3 b, V3 c) {
    V3 e0 = vsub(b, a), e1 = vsub(c, b), e2 = vsub(a, c);
    V3 p0 = vsub(p, a), p1 = vsub(p, b), p2 = vsub(p, c);
    V3 n = vnormalize(vcross(e0, e1));
    V3 r0 = vnormalize(vcross(e0, n)), r1 = vnormalize(vcross(e1, n)), r2 = vnormalize(vcross(e2, n));
    float dn = vdot(p0, n), d0 = vdot(p0, r0), d1 = vdot(p1, r1), d2 = vdot(p2, r2);
    float a01 = vdot(r0, r1), a02 = vdot(r0, r2), a12 = vdot(r1, r2);
    uint32_t flag = uint32_t(d2 > 0.0f) | (uint32_t(d1 > 0.0f) << 1) | (uint32_t(d0 > 0.0f) << 2);
    switch (flag) {
    case 0: return vsub(p, v3(n.x*dn, n.y*dn, n.z*dn));
    case 1: return closestOnSegment(p2, e2, c, a);
    case 2: return closestOnSegment(p1, e1, b, c);
    case 4: return closestOnSegment(p0, e0, a, b);
    case 6: return a01 < 0 ? b : bestOfTwo(p, closestOnSegment(p1, e1, b, c), closestOnSegment(p0, e0, a, b));
    case 3: return a12 < 0 ? c : bestOfTwo(p, closestOnSegment(p2, e2, c, a), closestOnSegment(p1, e1, b, c));
    case 5: return a02 < 0 ? a : bestOfTwo(p, closestOnSegment(p0, e0, a, b), closestOnSegment(p2, e2, c, a));
    default: return a;
    }
}
ATOM_HD bool intersectTri(V3 o, V3 dir, V3 a, V3 b, V3 c) {
    V3 edge1 = vsub(b, a), edge2 = vsub(c, a);
    V3 h = vcross(dir, edge2);
    float k = vdot(edge1, h);
    if (k > -0.0001f && k < 0.0001f) return false;
    float f = 1.0f / k;
    V3 s = vsub(o, a);
    float u = f * vdot(s, h);
    if (u < 0 || u > 1) return false;
    V3 q = vcross(s, edge1);
    float v = f * vdot(dir, q);
    if (v < 0 || u + v > 1) return false;
    float t = f * vdot(edge2, q);
    return t > 0.0001f;
}
ATOM_HD bool intersectBox(V3 o, V3 dir, V3 mn, V3 mx) {
    float tx1 = (mn.x - o.x) / dir.x, tx2 = (mx.x - o.x) / dir.x;
    float tmin = stdmin(tx1, tx2), tmax = stdmax(tx1, tx2);
    float ty1 = (mn.y - o.y) / dir.y, ty2 = (mx.y - o.y) / dir.y;
    tmin = stdmax(tmin, stdmin(ty1, ty2)); tmax = stdmin(tmax, stdmax(ty1, ty2));
    float tz1 = (mn.z - o.z) / dir.z, tz2 = (mx.z - o.z) / dir.z;
    tmin = stdmax(tmin, stdmin(tz1, tz2)); tmax = stdmin(tmax, stdmax(tz1, tz2));
    return tmax >= tmin && tmax > 0;
}

#define ATOM_SDF_STACK 64

// SDF constructor, one cell: BVHTriangle::closestPoint (same traversal order) plus the
// ray-parity vote (BVHTriangle::intersectionCount) with the same pseudo-random directions.
// A traversal stack overflow writes NaN; the host redoes those cells on the CPU.
ATOM_HD void sdfCell(const float *box, const uint32_t *info, const float *tris, const SdfInfo &g, uint32_t i, float *field) {
    uint32_t syz = g.sy*g.sz;
    uint32_t x = i / syz, r = i - x*syz, y = r / g.sz, z = r - y*g.sz;
    V3 point = v3(float(x)*g.cellSize, float(y)*g.cellSize, float(z)*g.cellSize);
    point = v3(point.x + g.half, point.y + g.half, point.z + g.half);
    point = v3(point.x + g.ox, point.y + g.oy, point.z + g.oz);

    // closest point
    V3 closest = v3(g.tri0x, g.tri0y, g.tri0z);
    float minD2 = posInf();
    uint32_t stack[ATOM_SDF_STACK];
    uint32_t sp = 0;
    stack[sp++] = 0;
    bool overflow = false;
    while (sp && !overflow) {
        uint32_t idx = stack[--sp];
        while (true) {
            // CPU: sphereAABBIntersection(point, minDistance2 + 1e-4, ...): double sum rounded to float
            float r2 = float(double(minD2) + 1e-4);
            if (!sphereAABB(point, r2, nodeMin(box, idx), nodeMax(box, idx))) break;
            uint32_t triCount = info[2*idx], firstLeft = info[2*idx+1];
            if (triCount > 0) {
                for (uint32_t k = firstLeft; k < firstLeft + triCount; ++k) {
                    V3 tc = closestOnTriangle(point, triV(tris, k, 0), triV(tris, k, 1), triV(tris, k, 2));
                    V3 dd = vsub(tc, point);
                    float d2 = vdot(dd, dd);
                    if (d2 < minD2) { minD2 = d2; closest = tc; }
                }
                break;
            }
            uint32_t l = firstLeft, rr = firstLeft + 1;
            if (sp >= ATOM_SDF_STACK) { overflow = true; break; }
            if (pointInAABB(point, nodeMin(box, l), nodeMax(box, l))) { idx = l; stack[sp++] = rr; }
            else if (pointInAABB(point, nodeMin(box, rr), nodeMax(box, rr))) { idx = rr; stack[sp++] = l; }
            else {
                float ld = pointAABBDist2(point, nodeMin(box, l), nodeMax(box, l));
                float rd = pointAABBDist2(point, nodeMin(box, rr), nodeMax(box, rr));
                idx = ld < rd ? l : rr;
                stack[sp++] = ld < rd ? rr : l;
            }
        }
    }

    // ray parity vote (hit counts are integers, so the traversal order does not matter)
    uint32_t outside = 0;
    uint32_t seed = i;
    for (uint32_t j = 0; j < g.raysCount && !overflow; ++j) {
        V3 dir = rngDir(seed);
        uint32_t hits = 0;
        sp = 0;
        stack[sp++] = 0;
        while (sp) {
            uint32_t n = stack[--sp];
            if (!intersectBox(point, dir, nodeMin(box, n), nodeMax(box, n))) continue;
            uint32_t triCount = info[2*n], firstLeft = info[2*n+1];
            if (triCount > 0) {
                for (uint32_t k = firstLeft; k < firstLeft + triCount; ++k)
                    hits += intersectTri(point, dir, triV(tris, k, 0), triV(tris, k, 1), triV(tris, k, 2)) ? 1u : 0u;
            } else {
                if (sp + 2 > ATOM_SDF_STACK) { overflow = true; break; }
                stack[sp++] = firstLeft + 1;
                stack[sp++] = firstLeft;
            }
        }
        outside += hits % 2;
        seed = rngUint(seed);
    }
    if (overflow) { field[i] = qNaN(); return; }
    float sign = outside > g.raysCount/2 ? -1.0f : 1.0f;
    field[i] = sign * vlen(vsub(point, closest));
}

} // namespace AtomK
