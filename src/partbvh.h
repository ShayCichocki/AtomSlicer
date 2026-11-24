// MIT License
// Copyright 2026 Giovanni Cocco and Inria
//
// partition-wip: a copy of BVHPoint used only by Partition, rewritten for speed with the
// SAME tree and the SAME query results in the SAME order (so the partition is unchanged
// bit for bit). See five-axis-printer/docs/PARTITION_SPEEDUP.md.
//
// What is different from BVHPoint, and why each is exact:
//  - Partition only removes points in batches followed by compact(), and only queries
//    after compact(). So at query and build time no removed point is ever in the id list,
//    and every `removedPoints.count()` test in BVHPoint is false there. They are dropped;
//    removal is a flag array used only by compact().
//  - Each point's id and coordinates are stored together (PE) in tree order, so the build
//    and the leaf scans read memory in order. Same values, same comparisons, same swaps.
//  - Indices are 32-bit; the traversal stacks are reused buffers instead of a new
//    std::vector per query; queries write into a caller buffer.
//  - The split loop is BVHPoint's, with the two children's bounds gathered during it
//    (NEON min/max) instead of two more passes. Bounds over a different order can only differ
//    in the sign of a zero, which no later comparison or non-zero result depends on.
//  - Subtrees are built in parallel; node numbering changes but every node keeps its two
//    children adjacent and in BVHPoint's order, and queries only follow that structure.
//  - Cone queries read each node's bounding-sphere center and apex offset from a table filled
//    after the build with coneAABBMayIntersect's own statements (bitwise the same floats).
//  - The float expressions in the tests (sphere/AABB, cone/AABB, point in cone) are copied
//    verbatim.

#pragma once
#include "mathutils.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <functional>
#if defined(__ARM_NEON) && !defined(PARTBVH_NO_NEON)
#include <arm_neon.h>
#define PARTBVH_NEON 1
#endif

class PartBVH {
public:
    static constexpr uint32_t NONE = 0xffffffffu;

    PartBVH(const std::vector<Vec3> &points) : points(points), removed(points.size(), 0) {
        pe.resize(points.size());
        for (size_t i = 0; i < pe.size(); ++i) {
            pe[i].p = points[i];
            pe[i].id = uint32_t(i);
        }
        build();
    }

    // Mark only; the tree is rebuilt by compact(). (BVHPoint::removePoint(i, false).)
    void removePoint(size_t idx) { removed[idx] = 1; }
    bool isRemoved(size_t idx) const { return removed[idx]; }

    // BVHPoint::anyPointsInCone, returns NONE for size_t(-1).
    uint32_t anyPointsInCone(const Vec3 &origin, const Vec3 &direction, float halfAngle) const {
        float cosHalfAngle = cosf(halfAngle);
        // coneAABBMayIntersect evaluates these per node in BVHPoint; same values.
        const float sinHalf = sinf(halfAngle);
        const float cosHalf = cosf(halfAngle);
        // Per-node bounding sphere and apex offset, precomputed by finish() with the same
        // expressions (bitwise the same floats) when the half angle matches.
        const bool useSph = halfAngle == sphHalfAngle && sph.size() == nodes.size();
        std::vector<uint32_t> &stack = coneStack;
        stack.clear();
        stack.push_back(0);
        while (stack.size()) {
            uint32_t idx = stack.back();
            stack.pop_back();
            while (true) {
                const Node &nd = nodes[idx];
                if (useSph) {
                    const NodeSphere &ns = sph[idx];
                    Vec3 po = ns.center - origin + direction*ns.offset;
                    float length = po.length();
                    if (!(length < std::numeric_limits<float>::epsilon()) && !(direction.dot(po)/length >= cosHalf)) break;
                } else if (!coneAABBMayIntersect(origin, direction, sinHalf, cosHalf, nd.minAABB, nd.maxAABB)) break;
                if (nd.isLeaf()) {
                    for (uint32_t offset = nd.firstLeft, last = offset + nd.pointNumber; offset < last; ++offset) {
                        if (pointInCone(pe[offset].p, origin, direction, cosHalfAngle)) return pe[offset].id;
                    }
                    break;
                }
                uint32_t leftIdx = nd.firstLeft;
                uint32_t rightIdx = leftIdx + 1;
                float leftDist2, rightDist2;
                if (useSph) {
                    leftDist2 = (sph[leftIdx].center-origin).length2();
                    rightDist2 = (sph[rightIdx].center-origin).length2();
                } else {
                    leftDist2 = ((nodes[leftIdx].minAABB+nodes[leftIdx].maxAABB)*0.5f-origin).length2();
                    rightDist2 = ((nodes[rightIdx].minAABB+nodes[rightIdx].maxAABB)*0.5f-origin).length2();
                }
                idx = leftDist2 < rightDist2 ? leftIdx : rightIdx;
                stack.push_back(leftDist2 < rightDist2 ? rightIdx : leftIdx);
            }
        }
        return NONE;
    }

    // BVHPoint::pointsInSphereIndex, appending to `index` (cleared first), same order.
    void pointsInSphereIndex(const Vec3 &center, float radius, std::vector<uint32_t> &index) const {
        index.clear();
        pointsInSphereAppend(center, radius, index, sphereStack);
    }

    // Thread-safe form (caller's stack); appends to `index`.
    void pointsInSphereAppend(const Vec3 &center, float radius, std::vector<uint32_t> &index, std::vector<uint32_t> &stack) const {
        float radius2 = radius*radius;
        stack.clear();
        stack.push_back(0);
        while (stack.size()) {
            uint32_t idx = stack.back();
            stack.pop_back();
            while (true) {
                const Node &nd = nodes[idx];
                if (!sphereAABBIntersection(center, radius2 + EPSILON, nd.minAABB, nd.maxAABB)) break;
                if (nd.isLeaf()) {
                    for (uint32_t offset = nd.firstLeft, last = offset + nd.pointNumber; offset < last; ++offset) {
                        float distance2 = (pe[offset].p-center).length2();
                        if (distance2 < radius2) {
                            index.push_back(pe[offset].id);
                        }
                    }
                    break;
                }
                uint32_t leftIdx = nd.firstLeft;
                uint32_t rightIdx = leftIdx + 1;

                if (pointAABBIntersection(center, nodes[leftIdx].minAABB-Vec3(radius), nodes[leftIdx].maxAABB+Vec3(radius))) {
                    idx = leftIdx;
                    stack.push_back(rightIdx);
                } else if (pointAABBIntersection(center, nodes[rightIdx].minAABB-Vec3(radius), nodes[rightIdx].maxAABB+Vec3(radius))) {
                    idx = rightIdx;
                    stack.push_back(leftIdx);
                } else {
                    float leftDist2 = pointAABBDistance2(center, nodes[leftIdx].minAABB, nodes[leftIdx].maxAABB);
                    float rightDist2 = pointAABBDistance2(center, nodes[rightIdx].minAABB, nodes[rightIdx].maxAABB);
                    idx = leftDist2 < rightDist2 ? leftIdx : rightIdx;
                    stack.push_back(leftDist2 < rightDist2 ? rightIdx : leftIdx);
                }
            }
        }
    }

    // BVHPoint::compact: drop removed ids keeping order, rebuild.
    void compact() {
        // The root's bounds are gathered here, over the kept points (min/max: the order only
        // matters for the sign of a zero bound, see split()).
        Vec3 mn(std::numeric_limits<float>::infinity());
        Vec3 mx(-std::numeric_limits<float>::infinity());
        const size_t n = pe.size();
        const size_t T = (parallelRunner && parallelWidth > 1 && n > 65536) ? 4*parallelWidth : 1;
        if (T == 1) {
            size_t idx = 0;
            for (size_t i = 0; i < n; ++i) {
                if (removed[pe[i].id]) continue;
                const Vec3 &p = pe[i].p;
                mn.x = std::min(mn.x, p.x); mx.x = std::max(mx.x, p.x);
                mn.y = std::min(mn.y, p.y); mx.y = std::max(mx.y, p.y);
                mn.z = std::min(mn.z, p.z); mx.z = std::max(mx.z, p.z);
                pe[idx++] = pe[i];
            }
            pe.resize(idx);
        } else {
            // stable parallel filter into a second buffer
            std::vector<size_t> kept(T + 1, 0);
            std::vector<Vec3> cmn(T, mn), cmx(T, mx);
            runParallel(T, [&](size_t t) {
                size_t c = 0;
                for (size_t i = n*t/T, e = n*(t+1)/T; i < e; ++i) c += !removed[pe[i].id];
                kept[t+1] = c;
            });
            for (size_t t = 0; t < T; ++t) kept[t+1] += kept[t];
            peTmp.resize(kept[T]);
            runParallel(T, [&](size_t t) {
                size_t o = kept[t];
                Vec3 a = cmn[t], b = cmx[t];
                for (size_t i = n*t/T, e = n*(t+1)/T; i < e; ++i) {
                    if (removed[pe[i].id]) continue;
                    const Vec3 &p = pe[i].p;
                    a.x = std::min(a.x, p.x); b.x = std::max(b.x, p.x);
                    a.y = std::min(a.y, p.y); b.y = std::max(b.y, p.y);
                    a.z = std::min(a.z, p.z); b.z = std::max(b.z, p.z);
                    peTmp[o++] = pe[i];
                }
                cmn[t] = a; cmx[t] = b;
            });
            for (size_t t = 0; t < T; ++t) {
                mn.x = std::min(mn.x, cmn[t].x); mx.x = std::max(mx.x, cmx[t].x);
                mn.y = std::min(mn.y, cmn[t].y); mx.y = std::max(mx.y, cmx[t].y);
                mn.z = std::min(mn.z, cmn[t].z); mx.z = std::max(mx.z, cmx[t].z);
            }
            pe.swap(peTmp);
        }
        build(&mn, &mx);
    }

    size_t size() const { return pe.size(); }

private:
    struct Node {
        Vec3 minAABB;
        Vec3 maxAABB;
        uint32_t pointNumber;
        uint32_t firstLeft;
        bool isLeaf() const { return pointNumber > 0; }
    };

    struct NodeSphere { Vec3 center; float offset; };   // offset = radius/sin(halfAngle)
    std::vector<NodeSphere> sph;
    float sphHalfAngle = -1.0f;

    static constexpr uint32_t MAX_POINTS_PER_NODE = 32;
    static constexpr float EPSILON = 1e-4f;

    const std::vector<Vec3> &points;
    std::vector<uint8_t> removed;
    std::vector<Node> nodes;
    // pe[k] = {position, id} of the k-th point in tree order (BVHPoint's pointsID[k]).
    struct alignas(16) PE { Vec3 p; uint32_t id; };
    std::vector<PE> pe, peTmp;
    mutable std::vector<uint32_t> sphereStack, coneStack;

    // Same tree as BVHPoint::build(). The node *numbering* differs from a serial build when
    // subtrees are built in parallel, but every node keeps its two children adjacent and in
    // the same order (child "firstLeft" = the >= split side, firstLeft+1 = the < split side),
    // and the queries only follow that structure, so their results are identical. Each
    // node's split only reads and permutes its own range, so subtrees are independent.
    void build(const Vec3 *rootMin = nullptr, const Vec3 *rootMax = nullptr) {
        nodes.clear();
        nodes.push_back(Node());
        nodes[0].firstLeft = 0;
        nodes[0].pointNumber = uint32_t(pe.size());
        if (rootMin) { nodes[0].minAABB = *rootMin; nodes[0].maxAABB = *rootMax; }
        else bounds(nodes[0]);

        // Serial phase: split breadth-first until there are enough independent subtrees.
        const size_t want = parallelSubtrees();
        std::vector<uint32_t> frontier{0}, next;
        // The nodes of one level have disjoint ranges, so they are split in parallel too.
        std::vector<Node> rights, lefts;
        std::vector<uint8_t> didSplit;
        while (want > 1 && frontier.size() && frontier.size() < want) {
            rights.resize(frontier.size());
            lefts.resize(frontier.size());
            didSplit.assign(frontier.size(), 0);
            auto one = [&](size_t k) { didSplit[k] = split(nodes[frontier[k]], rights[k], lefts[k]); };
            if (frontier.size() == 1) one(0); else runParallel(frontier.size(), one);
            next.clear();
            for (size_t k = 0; k < frontier.size(); ++k) {
                if (!didSplit[k]) continue;
                uint32_t idx = frontier[k];
                nodes[idx].firstLeft = uint32_t(nodes.size());
                nodes[idx].pointNumber = 0;
                nodes.push_back(rights[k]);
                nodes.push_back(lefts[k]);
                next.push_back(uint32_t(nodes.size()-2));
                next.push_back(uint32_t(nodes.size()-1));
            }
            frontier.swap(next);
        }
        if (want <= 1) {
            std::vector<Node> local;
            subtree(nodes[0], local);
            nodes.swap(local);
            finish();
            return;
        }

        // Parallel phase: build each frontier subtree on its own, then splice.
        std::vector<std::vector<Node>> locals(frontier.size());
        runParallel(frontier.size(), [&](size_t k) { subtree(nodes[frontier[k]], locals[k]); });
        for (size_t k = 0; k < frontier.size(); ++k) {
            std::vector<Node> &local = locals[k];
            const uint32_t base = uint32_t(nodes.size()) - 1;   // local index L -> base + L
            for (size_t L = 1; L < local.size(); ++L) {
                Node nd = local[L];
                if (!nd.isLeaf()) nd.firstLeft += base;
                nodes.push_back(nd);
            }
            Node root = local[0];
            if (!root.isLeaf()) root.firstLeft += base;
            nodes[frontier[k]] = root;
        }
        finish();
    }

    // Exactly the center/radius statements of coneAABBMayIntersect, once per node.
    static void nodeSphere(const Vec3 &minAABB, const Vec3 &maxAABB, float sinHalfAngle, NodeSphere &out) {
        Vec3 center = (minAABB+maxAABB)*0.5f;
        float radius = 0.5f*(maxAABB-minAABB).length()+EPSILON;
        out.center = center;
        out.offset = radius/sinHalfAngle;
    }

    void finish() {
        if (sphHalfAngle < 0) { sph.clear(); return; }
        sph.resize(nodes.size());
        const float sinHalf = sinf(sphHalfAngle);
        auto range = [&](size_t t, size_t T) {
            for (size_t k = nodes.size()*t/T, e = nodes.size()*(t+1)/T; k < e; ++k) nodeSphere(nodes[k].minAABB, nodes[k].maxAABB, sinHalf, sph[k]);
        };
        if (parallelRunner && parallelWidth > 1 && nodes.size() > 8192) {
            size_t T = 4*parallelWidth;
            runParallel(T, [&](size_t t) { range(t, T); });
        } else range(0, 1);
    }

public:
    // Precompute node spheres for cone queries with this half angle (call before queries).
    void setConeHalfAngle(float halfAngle) { sphHalfAngle = halfAngle; finish(); }
private:

    // Set by Partition: how many subtrees to build in parallel and how (1 = serial).
public:
    std::function<void(size_t, const std::function<void(size_t)> &)> parallelRunner;
    size_t parallelWidth = 1;
private:
    size_t parallelSubtrees() const { return parallelRunner && parallelWidth > 1 && pe.size() > 65536 ? 4*parallelWidth : 1; }
    void runParallel(size_t n, const std::function<void(size_t)> &f) { parallelRunner(n, f); }

    void bounds(Node &nd) const {
        Vec3 mn(std::numeric_limits<float>::infinity());
        Vec3 mx(-std::numeric_limits<float>::infinity());
        for (uint32_t offset = nd.firstLeft, last = offset + nd.pointNumber; offset < last; ++offset) {
            const Vec3 &p = pe[offset].p;
            mn.x = std::min(mn.x, p.x); mx.x = std::max(mx.x, p.x);
            mn.y = std::min(mn.y, p.y); mx.y = std::max(mx.y, p.y);
            mn.z = std::min(mn.z, p.z); mx.z = std::max(mx.z, p.z);
        }
        nd.minAABB = mn;
        nd.maxAABB = mx;
    }

    // One step of BVHPoint::subdivideNodes on node nd (whose bounds are set). Returns false
    // when BVHPoint leaves it a leaf. On true, `right` is the child BVHPoint stores at
    // firstLeft and `left` the one at firstLeft+1, both with bounds.
    bool split(const Node &nd, Node &right, Node &left) {
        // BVHPoint counts non-removed points up to 33; none are removed here.
        if (nd.pointNumber <= MAX_POINTS_PER_NODE) return false;
        Vec3 extent = nd.maxAABB - nd.minAABB;
        uint8_t axis = 0;
        for (uint8_t i = 1; i < 3; ++i) {
            if (extent[i] > extent[axis])
                axis = i;
        }
        float splitValue = nd.minAABB[axis] + 0.5f*extent[axis];

        // Same Hoare-style loop as BVHPoint (size_t i/j semantics kept), with
        // the children's bounds gathered on the way:
        // a point is final when it is accepted at i (left side, ascending) or swapped to j
        // (right side, descending). min/max over a different order can only change the sign
        // of a zero bound, which no later comparison or non-zero result depends on (see
        // PARTITION_SPEEDUP.md).
        size_t i = nd.firstLeft;
        size_t j = i + nd.pointNumber - 1;
        const float inf = std::numeric_limits<float>::infinity();
#if PARTBVH_NEON
        // lanes x, y, z (lane 3 carries the id bits and is ignored)
        float32x4_t lmn = vdupq_n_f32(inf), lmx = vdupq_n_f32(-inf);
        float32x4_t rmn = lmn, rmx = lmx;
        PE *a = pe.data();
        while (i <= j) {
            const PE x = a[i];
            const float32x4_t v = vld1q_f32(&x.p.x);
            if ((&x.p.x)[axis] < splitValue) {
                lmn = vminq_f32(lmn, v); lmx = vmaxq_f32(lmx, v);
                i += 1;
            } else {
                a[i] = a[j];
                a[j] = x;
                rmn = vminq_f32(rmn, v); rmx = vmaxq_f32(rmx, v);
                if (!j) break;
                j -= 1;
            }
        }
        float lmnx = vgetq_lane_f32(lmn, 0), lmny = vgetq_lane_f32(lmn, 1), lmnz = vgetq_lane_f32(lmn, 2);
        float lmxx = vgetq_lane_f32(lmx, 0), lmxy = vgetq_lane_f32(lmx, 1), lmxz = vgetq_lane_f32(lmx, 2);
        float rmnx = vgetq_lane_f32(rmn, 0), rmny = vgetq_lane_f32(rmn, 1), rmnz = vgetq_lane_f32(rmn, 2);
        float rmxx = vgetq_lane_f32(rmx, 0), rmxy = vgetq_lane_f32(rmx, 1), rmxz = vgetq_lane_f32(rmx, 2);
#else
        float lmnx = inf, lmny = inf, lmnz = inf, lmxx = -inf, lmxy = -inf, lmxz = -inf;
        float rmnx = inf, rmny = inf, rmnz = inf, rmxx = -inf, rmxy = -inf, rmxz = -inf;
        PE *a = pe.data();
        while (i <= j) {
            const PE x = a[i];
            if ((&x.p.x)[axis] < splitValue) {
                lmnx = std::min(lmnx, x.p.x); lmxx = std::max(lmxx, x.p.x);
                lmny = std::min(lmny, x.p.y); lmxy = std::max(lmxy, x.p.y);
                lmnz = std::min(lmnz, x.p.z); lmxz = std::max(lmxz, x.p.z);
                i += 1;
            } else {
                a[i] = a[j];
                a[j] = x;
                rmnx = std::min(rmnx, x.p.x); rmxx = std::max(rmxx, x.p.x);
                rmny = std::min(rmny, x.p.y); rmxy = std::max(rmxy, x.p.y);
                rmnz = std::min(rmnz, x.p.z); rmxz = std::max(rmxz, x.p.z);
                if (!j) break;
                j -= 1;
            }
        }
#endif

        uint32_t leftCount = uint32_t(i - nd.firstLeft);
        if (!(leftCount > 0 && leftCount < nd.pointNumber)) return false;
        left.firstLeft = nd.firstLeft;
        left.pointNumber = leftCount;
        right.firstLeft = uint32_t(i);
        right.pointNumber = nd.pointNumber - leftCount;
        left.minAABB = Vec3(lmnx, lmny, lmnz);  left.maxAABB = Vec3(lmxx, lmxy, lmxz);
        right.minAABB = Vec3(rmnx, rmny, rmnz); right.maxAABB = Vec3(rmxx, rmxy, rmxz);
        return true;
    }

    // BVHPoint::subdivideNodes from `root` into `out` (out[0] = root), same DFS order.
    void subtree(const Node &root, std::vector<Node> &out) {
        out.clear();
        out.push_back(root);
        std::vector<uint32_t> stack{0};
        while (stack.size()) {
            uint32_t idx = stack.back();
            stack.pop_back();
            Node right, left;
            if (!split(out[idx], right, left)) continue;
            out[idx].firstLeft = uint32_t(out.size());
            out[idx].pointNumber = 0;
            stack.push_back(uint32_t(out.size()));
            stack.push_back(uint32_t(out.size()+1));
            out.push_back(right);
            out.push_back(left);
        }
    }

    bool sphereAABBIntersection(const Vec3 &center, float radius2, const Vec3 &minAABB, const Vec3 &maxAABB) const {
        float dist2 = 0.0f;
        for (uint8_t axis = 0; axis < 3; ++axis) {
            float m = minAABB[axis]-center[axis], M = center[axis]-maxAABB[axis];
            dist2 += (m > 0)*m*m + (M > 0)*M*M;
        }
        return dist2 <= radius2;
    }

    bool pointAABBIntersection(const Vec3 &point, const Vec3 &minAABB, const Vec3 &maxAABB) const {
        for (uint8_t axis = 0; axis < 3; ++axis) {
            if (point[axis] < minAABB[axis] || point[axis] > maxAABB[axis])
                return false;
        }
        return true;
    }

    float pointAABBDistance2(const Vec3 &point, const Vec3 &minAABB, const Vec3 &maxAABB) const {
        Vec3 closest(point);
        for (uint8_t axis = 0; axis < 3; ++axis) {
            if (point[axis] < minAABB[axis]) {
                closest[axis] = minAABB[axis];
            } else if (point[axis] > maxAABB[axis]) {
                closest[axis] = maxAABB[axis];
            }
        }
        return (point-closest).length2();
    }

    bool pointInCone(const Vec3 &point, const Vec3 &origin, const Vec3 &direction, float cosHalfAngle) const {
        Vec3 po = point-origin;
        float length = po.length();
        if (length < std::numeric_limits<float>::min()) return false;
        return direction.dot(po)/length >= cosHalfAngle;
    }

    bool coneAABBMayIntersect(const Vec3 &origin, const Vec3 &direction, float sinHalfAngle, float cosHalfAngle, const Vec3 &minAABB, const Vec3 &maxAABB) const {
        Vec3 center = (minAABB+maxAABB)*0.5f;
        float radius = 0.5f*(maxAABB-minAABB).length()+EPSILON;
        Vec3 po = center - origin + direction*(radius/sinHalfAngle);
        float length = po.length();
        if (length < std::numeric_limits<float>::epsilon()) return true;
        return direction.dot(po)/length >= cosHalfAngle;
    }
};
