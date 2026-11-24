// MIT License
// Metal compute kernels for the AtomSlicer grid smoothers (red-black Gauss-Seidel half-sweeps).
// Each kernel is a line-by-line port of the CPU per-cell function named next to it; the
// arithmetic is written in the same order as the C++ so results stay within float rounding.
// Compiled at run time (newLibraryWithSource), so no offline Metal toolchain is needed.

#include <metal_stdlib>
using namespace metal;

// No implicit multiply-add fusion: every fma below is written out by hand to mirror what
// Apple clang emits for the CPU code (-O3, -ffp-contract=on: a*b + c inside one source
// expression becomes fma(a, b, c), checking the left operand first).
#pragma METAL fp contract(off)

// ---------------------------------------------------------------- common
struct GridInfo {
    uint sx, sy, sz;      // grid size
    uint parity;          // active colour: (x+y+z) % 2 == parity
    float cellSize;
    float ox, oy, oz;     // origin
    float p0, p1, p2, p3; // kernel parameters (layerHeight, ...)
    uint pass;            // pass index inside a convergence batch
    uint pad0, pad1, pad2;
};

constant uint FROZEN = 1u;
constant uint OUTSIDE = 2u;

// Same layout as the C++ structs (checked on the host through sizeof).
struct NData {            // NormalField::Data
    packed_float3 normal;
    float distance;
    uchar flags;
};

struct TData {            // TriPhasor::Data
    packed_float3 normal;
    packed_float3 tangent;
    float phase[3];
    uchar flags;
};

// Vec3 helpers matching mathutils.h exactly (component order, divisions, fallbacks).
// x*x' + y*y' + z*z'  ->  fma(z, z', fma(x, x', y*y'))
static inline float vdot(float3 a, float3 b) { return fma(a.z, b.z, fma(a.x, b.x, a.y*b.y)); }
static inline float vlen(float3 a) { return sqrt(vdot(a, a)); }
static inline float3 vnormalize(float3 a) {
    float l = vlen(a);
    float3 r = float3(a.x/l, a.y/l, a.z/l);
    return l > 0 ? r : float3(1, 0, 0);
}
static inline float3 vcross(float3 a, float3 b) {
    // y*b.z - z*b.y -> fma(y, b.z, -(z*b.y)), same for the others
    return float3(fma(a.y, b.z, -(a.z*b.y)), -fma(a.x, b.z, -(a.z*b.x)), fma(a.x, b.y, -(a.y*b.x)));
}
static inline float3 vorthonormalize(float3 a, float3 n) {
    float3 proj = n * vdot(a, n);
    return vnormalize(a - proj);
}

// Map thread id -> cell index of the active colour; false when past the end.
static inline bool cellOf(uint tid, constant GridInfo &g, thread uint3 &c, thread uint &i) {
    uint halfZ = (g.sz + 1) / 2;
    uint row = tid / halfZ;
    uint k = tid - row*halfZ;
    if (row >= g.sx*g.sy) return false;
    uint x = row / g.sy, y = row - x*g.sy;
    uint z = 2*k + ((x + y + g.parity) & 1u);
    if (z >= g.sz) return false;
    c = uint3(x, y, z);
    i = (x*g.sy + y)*g.sz + z;
    return true;
}

static inline float3 cellPos(uint3 c, constant GridInfo &g) {
    float3 v = float3(float(c.x) + 0.5f, float(c.y) + 0.5f, float(c.z) + 0.5f);
    float3 s = float3(g.cellSize*v.x, g.cellSize*v.y, g.cellSize*v.z);
    return float3(s.x + g.ox, s.y + g.oy, s.z + g.oz);
}

// Neighbours in the CPU order: +x, -x, +y, -y, +z, -z. Returns the count; idx/pos filled.
static inline uint neighbours(uint3 c, uint i, constant GridInfo &g, thread uint *idx, thread uint3 *nc) {
    uint n = 0;
    uint syz = g.sy*g.sz;
    if (c.x < g.sx-1) { idx[n] = i + syz;  nc[n] = uint3(c.x+1, c.y, c.z); n++; }
    if (c.x > 0)      { idx[n] = i - syz;  nc[n] = uint3(c.x-1, c.y, c.z); n++; }
    if (c.y < g.sy-1) { idx[n] = i + g.sz; nc[n] = uint3(c.x, c.y+1, c.z); n++; }
    if (c.y > 0)      { idx[n] = i - g.sz; nc[n] = uint3(c.x, c.y-1, c.z); n++; }
    if (c.z < g.sz-1) { idx[n] = i + 1;    nc[n] = uint3(c.x, c.y, c.z+1); n++; }
    if (c.z > 0)      { idx[n] = i - 1;    nc[n] = uint3(c.x, c.y, c.z-1); n++; }
    return n;
}

// mathutils.h alignPhases with w = 1
static inline void alignPhases(float3 p_i, float3 p_j, float3 d_i, float3 d_j, float f_i, float f_j, float phi_j, thread float &x, thread float &y) {
    float3 s = p_i + p_j;
    float3 p_ij = float3(s.x*0.5f, s.y*0.5f, s.z*0.5f);
    float a = 2.0f*M_PI_F*f_i * vdot(p_ij - p_i, d_i);
    float b = fma(2.0f*M_PI_F*f_j, vdot(p_ij - p_j, d_j), phi_j);
    float dd = vdot(d_i, d_j);
    float angle = dd > 0.0f ? b - a : -b - a;
    x = fma(fabs(dd), precise::cos(angle), x);
    y = fma(fabs(dd), precise::sin(angle), y);
}

// Convergence bookkeeping: a pass runs only if the previous pass of this batch changed
// something, so the GPU stops exactly where the CPU loop would break.
static inline bool passEnabled(constant GridInfo &g, device atomic_uint *flags) {
    return g.pass == 0 || atomic_load_explicit(&flags[g.pass-1], memory_order_relaxed) != 0;
}

// ---------------------------------------------------------------- NormalField
// NormalField::smooth
kernel void normal_smooth(device NData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint3 c; uint i;
    if (!cellOf(tid, g, c, i)) return;
    uchar f = d[i].flags;
    if ((f & OUTSIDE) || (f & FROZEN)) return;
    uint idx[6]; uint3 nc[6];
    uint n = neighbours(c, i, g, idx, nc);
    float3 norm = float3(0);
    for (uint k = 0; k < n; ++k) {
        if (d[idx[k]].flags & OUTSIDE) continue;
        norm += float3(d[idx[k]].normal);
    }
    d[i].normal = packed_float3(vnormalize(norm));
}

// NormalField::smoothFrozen
kernel void normal_smooth_frozen(device NData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint3 c; uint i;
    if (!cellOf(tid, g, c, i)) return;
    uchar f = d[i].flags;
    if ((f & OUTSIDE) || !(f & FROZEN)) return;
    uint idx[6]; uint3 nc[6];
    uint n = neighbours(c, i, g, idx, nc);
    float3 norm = float3(0);
    for (uint k = 0; k < n; ++k) {
        uchar fn = d[idx[k]].flags;
        if ((fn & OUTSIDE) || !(fn & FROZEN)) continue;
        norm += float3(d[idx[k]].normal);
    }
    if (vdot(norm, norm) > 1e-4f) {
        d[i].normal = packed_float3(vnormalize(norm));
    }
}

// NormalField constructor, ConformalTop / ConformalNearest: smoothConvergence lambda
kernel void normal_converge_distance(device NData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]],
                                     device atomic_uint *flags [[buffer(2)]],
                                     uint tid [[thread_position_in_grid]]) {
    if (!passEnabled(g, flags)) return;
    bool changed = false;
    uint3 c; uint i;
    if (cellOf(tid, g, c, i) && !(d[i].flags & OUTSIDE)) {
        uint idx[6]; uint3 nc[6];
        uint n = neighbours(c, i, g, idx, nc);
        float3 pi = cellPos(c, g);
        float di = d[i].distance;
        float3 ni = float3(d[i].normal);
        for (uint k = 0; k < n; ++k) {
            if (d[idx[k]].flags & OUTSIDE) continue;
            float distAB = vlen(pi - cellPos(nc[k], g));
            float dn = d[idx[k]].distance;
            if (di > dn + distAB) {
                di = dn + distAB;
                ni = float3(d[idx[k]].normal);
                changed = true;
            }
        }
        if (changed) {
            d[i].distance = di;
            d[i].normal = packed_float3(ni);
        }
    }
    if (simd_any(changed) && simd_is_first()) {
        atomic_store_explicit(&flags[g.pass], 1u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- TriPhasor
// TriPhasor::smoothLayers, p0 = layerHeight
kernel void tri_smooth_layers(device TData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint3 c; uint i;
    if (!cellOf(tid, g, c, i)) return;
    uchar f = d[i].flags;
    if ((f & OUTSIDE) || (f & FROZEN)) return;
    uint idx[6]; uint3 nc[6];
    uint n = neighbours(c, i, g, idx, nc);
    float x = 0, y = 0;
    float3 pi = cellPos(c, g);
    float3 d_i = float3(d[i].normal);
    float fr = 1.0f/g.p0;
    for (uint k = 0; k < n; ++k) {
        if (d[idx[k]].flags & OUTSIDE) continue;
        float3 d_j = float3(d[idx[k]].normal);
        alignPhases(pi, cellPos(nc[k], g), d_i, d_j, fr, fr, d[idx[k]].phase[2], x, y);
    }
    d[i].phase[2] = precise::atan2(y, x);
}

// TriPhasor::smoothTangents (p0 = 0) and smoothTangentsAll (p0 = 1)
kernel void tri_smooth_tangents(device TData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint3 c; uint i;
    if (!cellOf(tid, g, c, i)) return;
    uchar f = d[i].flags;
    if (g.p0 == 0.0f && ((f & OUTSIDE) || (f & FROZEN))) return;
    uint idx[6]; uint3 nc[6];
    uint n = neighbours(c, i, g, idx, nc);
    float3 tang = float3(0);
    for (uint k = 0; k < n; ++k) {
        if (d[idx[k]].flags & OUTSIDE) continue;
        float3 t = float3(d[idx[k]].tangent);
        float s = vdot(tang, t) < 0 ? -1.0f : 1.0f;
        tang += float3(t.x*s, t.y*s, t.z*s);
    }
    d[i].tangent = packed_float3(vorthonormalize(tang, float3(d[i].normal)));
}

// TriPhasor::smoothTriphasor, p0 = layerHeight
kernel void tri_smooth_triphasor(device TData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint3 c; uint i;
    if (!cellOf(tid, g, c, i)) return;
    uchar f = d[i].flags;
    if (f & OUTSIDE) return;
    uint idx[6]; uint3 nc[6];
    uint n = neighbours(c, i, g, idx, nc);
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    float3 pi = cellPos(c, g);
    float3 d_i = float3(d[i].tangent);
    float3 d_i_b = vnormalize(vcross(float3(d[i].normal), d_i));
    float ph2 = d[i].phase[2];
    float fr = 1.0f/g.p0;
    for (uint k = 0; k < n; ++k) {
        uint j = idx[k];
        if (d[j].flags & OUTSIDE) continue;
        // CPU: fabsf(a - b) > M_PI (double); for floats that is >= the float just above pi.
        if (fabs(ph2 - d[j].phase[2]) >= M_PI_F) continue;
        float3 pj = cellPos(nc[k], g);
        float3 d_j = float3(d[j].tangent);
        float3 d_j_b = vnormalize(vcross(float3(d[j].normal), d_j));
        float phj0 = d[j].phase[0], phj1 = d[j].phase[1];
        alignPhases(pi, pj, d_i, d_j, fr, fr, phj0, x0, y0);
        alignPhases(pi, pj, d_i_b, d_j, fr, fr, phj0, x1, y1);
        alignPhases(pi, pj, d_i, d_j_b, fr, fr, phj1, x0, y0);
        alignPhases(pi, pj, d_i_b, d_j_b, fr, fr, phj1, x1, y1);
    }
    d[i].phase[0] = precise::atan2(y0, x0);
    if (!(f & FROZEN)) {
        d[i].phase[1] = precise::atan2(y1, x1);
    }
}

// TriPhasor::extractAtoms, first loop: best of 8^3 sub-cell samples of the tri-phasor
// field. p0 = range, p1/p2 = 2*pi/layerHeight as a float pair (the CPU computes the cosine
// argument in double, so it is rebuilt here in float-float before rounding to float).
static inline float phaseArg(float dotv, float ph, float khi, float klo) {
    float p = khi*dotv;
    float e = fma(khi, dotv, -p);
    e = fma(klo, dotv, e);
    float s = p + ph;
    float bb = s - p;
    float err = (p - (s - bb)) + (ph - bb);
    return s + (err + e);
}

kernel void tri_extract_atoms(device const TData *d [[buffer(0)]], constant GridInfo &g [[buffer(1)]],
                              device float *maxVal [[buffer(2)]], device float *maxPos [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    uint syz = g.sy*g.sz;
    if (i >= g.sx*syz) return;
    if (d[i].flags & OUTSIDE) return;
    uint x = i / syz, r = i - x*syz, y = r / g.sz, z = r - y*g.sz;
    float3 base = cellPos(uint3(x, y, z), g);
    float3 dirT = float3(d[i].tangent);
    float3 dirN = float3(d[i].normal);
    float3 dirB = vnormalize(vcross(dirN, dirT));
    float phaseT = d[i].phase[0], phaseB = d[i].phase[1], phaseN = d[i].phase[2];
    float range = g.p0, khi = g.p1, klo = g.p2;
    float best = -INFINITY;
    float3 bestPos = float3(NAN);
    for (uint a = 0; a < 8; ++a) {
        float fx = 2*(float(a)/7.0f)-1;
        for (uint b = 0; b < 8; ++b) {
            float fy = 2*(float(b)/7.0f)-1;
            for (uint c = 0; c < 8; ++c) {
                float fz = 2*(float(c)/7.0f)-1;
                float3 pos = float3(fx*range, fy*range, fz*range);
                float valT = precise::cos(phaseArg(vdot(dirT, pos), phaseT, khi, klo));
                float valB = precise::cos(phaseArg(vdot(dirB, pos), phaseB, khi, klo));
                float valN = precise::cos(phaseArg(vdot(dirN, pos), phaseN, khi, klo));
                float v = valT + valB;
                v = v + valN;
                if (v > best) {
                    best = v;
                    bestPos = float3(base.x + pos.x, base.y + pos.y, base.z + pos.z);
                }
            }
        }
    }
    maxVal[i] = best;
    maxPos[3*i+0] = bestPos.x; maxPos[3*i+1] = bestPos.y; maxPos[3*i+2] = bestPos.z;
}

// ---------------------------------------------------------------- SDF
// SDF constructor: per cell, closest point on the mesh (BVHTriangle::closestPoint, same
// traversal order) and the ray-parity vote (BVHTriangle::intersectionCount) with the same
// pseudo-random directions. Stack overflow writes NaN and the host redoes that cell on CPU.
struct SdfInfo {
    uint sx, sy, sz, raysCount;
    float cellSize, halfCell, ox, oy, oz;   // point = ((cellSize*c) + halfCell) + offset
    float tri0x, tri0y, tri0z;          // tris[0].a, the CPU's initial "closest"
    float eps_hi, eps_lo;               // 1e-4 (double) as a float pair
    uint pad;
};

static inline float stdmin(float a, float b) { return (b < a) ? b : a; }   // std::min
static inline float stdmax(float a, float b) { return (a < b) ? b : a; }   // std::max

static inline float3 nodeMin(device const float *box, uint n) { return float3(box[6*n], box[6*n+1], box[6*n+2]); }
static inline float3 nodeMax(device const float *box, uint n) { return float3(box[6*n+3], box[6*n+4], box[6*n+5]); }
static inline float3 triV(device const float *t, uint k, uint j) { return float3(t[9*k+3*j], t[9*k+3*j+1], t[9*k+3*j+2]); }

static inline uint rngUint(uint v) {
    uint state = 747796405u*v + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
static inline float rngFloat(uint v) { return float(rngUint(v)) / 4294967296.0f; }
static inline float3 rngDir(uint v) {
    float3 r = float3(rngFloat(v), rngFloat(13u*v+5u), rngFloat(17u*v+7u));
    float3 t = float3(2*r.x, 2*r.y, 2*r.z);
    return vnormalize(float3(t.x - 1, t.y - 1, t.z - 1));
}

static inline bool pointInAABB(float3 p, float3 mn, float3 mx) {
    for (uint a = 0; a < 3; ++a) if (p[a] < mn[a] || p[a] > mx[a]) return false;
    return true;
}
static inline bool sphereAABB(float3 c, float r2, float3 mn, float3 mx) {
    float dist2 = 0.0f;
    for (uint a = 0; a < 3; ++a) {
        float m = mn[a]-c[a], M = c[a]-mx[a];
        dist2 += fma(float(m > 0)*m, m, float(M > 0)*M*M);
    }
    return dist2 <= r2;
}
static inline float pointAABBDist2(float3 p, float3 mn, float3 mx) {
    float3 cl = p;
    for (uint a = 0; a < 3; ++a) {
        if (p[a] < mn[a]) cl[a] = mn[a];
        else if (p[a] > mx[a]) cl[a] = mx[a];
    }
    float3 d = p - cl;
    return vdot(d, d);
}
static inline float3 closestOnSegment(float3 ds, float3 edge, float3 start, float3 end) {
    float l = vdot(ds, edge);
    float s = vdot(edge, edge);
    if (l > 0 && l < s) { float q = l/s; return start + float3(edge.x*q, edge.y*q, edge.z*q); }
    return l <= 0 ? start : end;
}
static inline float3 bestOfTwo(float3 p, float3 a, float3 b) {
    float3 da = p - a, db = p - b;
    return vdot(da, da) < vdot(db, db) ? a : b;
}
static inline float3 closestOnTriangle(float3 p, float3 a, float3 b, float3 c) {
    float3 e0 = b-a, e1 = c-b, e2 = a-c;
    float3 p0 = p-a, p1 = p-b, p2 = p-c;
    float3 n = vnormalize(vcross(e0, e1));
    float3 r0 = vnormalize(vcross(e0, n)), r1 = vnormalize(vcross(e1, n)), r2 = vnormalize(vcross(e2, n));
    float d = vdot(p0, n), d0 = vdot(p0, r0), d1 = vdot(p1, r1), d2 = vdot(p2, r2);
    float a01 = vdot(r0, r1), a02 = vdot(r0, r2), a12 = vdot(r1, r2);
    uint flag = uint(d2 > 0.0f) | (uint(d1 > 0.0f) << 1) | (uint(d0 > 0.0f) << 2);
    switch (flag) {
    case 0: return p - float3(n.x*d, n.y*d, n.z*d);
    case 1: return closestOnSegment(p2, e2, c, a);
    case 2: return closestOnSegment(p1, e1, b, c);
    case 4: return closestOnSegment(p0, e0, a, b);
    case 6: return a01 < 0 ? b : bestOfTwo(p, closestOnSegment(p1, e1, b, c), closestOnSegment(p0, e0, a, b));
    case 3: return a12 < 0 ? c : bestOfTwo(p, closestOnSegment(p2, e2, c, a), closestOnSegment(p1, e1, b, c));
    case 5: return a02 < 0 ? a : bestOfTwo(p, closestOnSegment(p0, e0, a, b), closestOnSegment(p2, e2, c, a));
    default: return a;
    }
}
static inline bool intersectTri(float3 o, float3 dir, float3 a, float3 b, float3 c) {
    float3 edge1 = b - a, edge2 = c - a;
    float3 h = vcross(dir, edge2);
    float k = vdot(edge1, h);
    if (k > -0.0001f && k < 0.0001f) return false;
    float f = 1.0f / k;
    float3 s = o - a;
    float u = f * vdot(s, h);
    if (u < 0 || u > 1) return false;
    float3 q = vcross(s, edge1);
    float v = f * vdot(dir, q);
    if (v < 0 || u + v > 1) return false;
    float t = f * vdot(edge2, q);
    return t > 0.0001f;
}
static inline bool intersectBox(float3 o, float3 dir, float3 mn, float3 mx) {
    float tx1 = (mn.x - o.x) / dir.x, tx2 = (mx.x - o.x) / dir.x;
    float tmin = stdmin(tx1, tx2), tmax = stdmax(tx1, tx2);
    float ty1 = (mn.y - o.y) / dir.y, ty2 = (mx.y - o.y) / dir.y;
    tmin = stdmax(tmin, stdmin(ty1, ty2)); tmax = stdmin(tmax, stdmax(ty1, ty2));
    float tz1 = (mn.z - o.z) / dir.z, tz2 = (mx.z - o.z) / dir.z;
    tmin = stdmax(tmin, stdmin(tz1, tz2)); tmax = stdmin(tmax, stdmax(tz1, tz2));
    return tmax >= tmin && tmax > 0;
}

#define SDF_STACK 64

kernel void sdf_cells(device const float *box [[buffer(0)]], device const uint *info [[buffer(1)]],
                      device const float *tris [[buffer(2)]], constant SdfInfo &g [[buffer(3)]],
                      device float *field [[buffer(4)]], constant uint &first [[buffer(5)]],
                      constant uint &count [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    uint i = first + tid;
    uint syz = g.sy*g.sz;
    uint x = i / syz, r = i - x*syz, y = r / g.sz, z = r - y*g.sz;
    float3 point = float3(float(x)*g.cellSize, float(y)*g.cellSize, float(z)*g.cellSize);
    point = float3(point.x + g.halfCell, point.y + g.halfCell, point.z + g.halfCell);
    point = float3(point.x + g.ox, point.y + g.oy, point.z + g.oz);

    // closest point
    float3 closest = float3(g.tri0x, g.tri0y, g.tri0z);
    float minD2 = INFINITY;
    uint stack[SDF_STACK];
    uint sp = 0;
    stack[sp++] = 0;
    bool overflow = false;
    while (sp && !overflow) {
        uint idx = stack[--sp];
        while (true) {
            float r2;
            if (isinf(minD2)) r2 = minD2;
            else {   // float(double(minD2) + 1e-4)
                float s = minD2 + g.eps_hi;
                float bb = s - minD2;
                float err = (minD2 - (s - bb)) + (g.eps_hi - bb);
                r2 = s + (err + g.eps_lo);
            }
            if (!sphereAABB(point, r2, nodeMin(box, idx), nodeMax(box, idx))) break;
            uint triCount = info[2*idx], firstLeft = info[2*idx+1];
            if (triCount > 0) {
                for (uint k = firstLeft; k < firstLeft + triCount; ++k) {
                    float3 tc = closestOnTriangle(point, triV(tris, k, 0), triV(tris, k, 1), triV(tris, k, 2));
                    float3 dd = tc - point;
                    float d2 = vdot(dd, dd);
                    if (d2 < minD2) { minD2 = d2; closest = tc; }
                }
                break;
            }
            uint l = firstLeft, rr = firstLeft + 1;
            if (sp >= SDF_STACK) { overflow = true; break; }
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

    // ray parity vote
    uint outside = 0;
    uint seed = i;
    for (uint j = 0; j < g.raysCount && !overflow; ++j) {
        float3 dir = rngDir(seed);
        uint hits = 0;
        sp = 0;
        stack[sp++] = 0;
        while (sp) {
            uint n = stack[--sp];
            if (!intersectBox(point, dir, nodeMin(box, n), nodeMax(box, n))) continue;
            uint triCount = info[2*n], firstLeft = info[2*n+1];
            if (triCount > 0) {
                for (uint k = firstLeft; k < firstLeft + triCount; ++k)
                    hits += intersectTri(point, dir, triV(tris, k, 0), triV(tris, k, 1), triV(tris, k, 2)) ? 1u : 0u;
            } else {
                if (sp + 2 > SDF_STACK) { overflow = true; break; }
                stack[sp++] = firstLeft + 1;
                stack[sp++] = firstLeft;
            }
        }
        outside += hits % 2;
        seed = rngUint(seed);
    }
    if (overflow) { field[i] = NAN; return; }
    float sign = outside > g.raysCount/2 ? -1.0f : 1.0f;
    float3 delta = point - closest;
    field[i] = sign * vlen(delta);
}
