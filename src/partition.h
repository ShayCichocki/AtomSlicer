// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once

#include "bvhpoint.h"
#include "partbvh.h"
#include <unordered_set>
#include "machine.h"
#include <cassert>
#include <string>

class Partition {
public:
    Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float layerHeight, float nozzleConeAngle = 95.0f, bool upCollisionCheck = false, bool quiet = false);
    Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const char *indexCSVPath);
    const std::vector<std::vector<size_t>> &getLayers() const { return layers; }

    void saveLayersCSV(const char *path) const;
    void saveLayersIndexCSV(const char *path) const;
    void saveLayersPLY(const char *path) const;
    void savePartitionToPLY(const char *path) const;

private:
    bool isNotLocallyBlocked(size_t i) const;
    bool isNeighboor(size_t i, size_t j) const;
    size_t getBlocker(size_t i) const;
    // partition-wip caches (exact, see PARTITION_SPEEDUP.md):
    size_t blockerOf(size_t i) const;      // == getBlocker(i) for the current tree
    bool isBlocked(size_t i) const;        // == (getBlocker(i) != size_t(-1))
    bool notLocallyBlocked(size_t i) const; // == isNotLocallyBlocked(i)
    void treeChanged() { ++treeVersion; }
    void floodNeighbours(size_t i, uint32_t &begin, uint32_t &count) const;
    void setNeighbours(size_t i, uint32_t &begin, uint32_t &count) const;
    bool componentAccessible(size_t start, std::vector<uint32_t> &comp) const;
    size_t countGen(const std::vector<uint32_t> &v) const { size_t c = 0; for (auto g : v) c += g == floodGen; return c; }
    void flood(size_t start, std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked) const;
    void getBlockers(std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked, std::unordered_set<size_t> &blocker) const;
    size_t getHigherBlocker(std::unordered_set<size_t> &blocker) const;
    void savePoints(const std::unordered_set<size_t> &layer, const std::string &path) const;

    float layerHeight;
    std::vector<std::vector<size_t>> layers;
    const std::vector<Vec3> &points;
    const std::vector<Vec3> &normals;
    PartBVH *bvh;
    // reused query buffers (one per call site that can be live at the same time)
    mutable std::vector<uint32_t> nbLocal, nbFlood, nbUpdate;
    // getBlocker result per atom, valid for tree version blockerVersion[i] (0 = never computed).
    // Across versions it still answers "is blocked?": an empty cone stays empty as atoms are only
    // removed, and a blocker that is still in the tree still blocks.
    mutable std::vector<uint32_t> blockerId, blockerVersion;
    // isNotLocallyBlocked: 0 unknown, 1 not blocked (stays so), w+2 = blocked by atom w.
    mutable std::vector<uint32_t> localWitness;
    uint32_t treeVersion = 1;
    bool verify = false;
    mutable std::vector<uint32_t> nbVersion, nbBegin, nbCount, nbArena;
    mutable std::vector<uint32_t> inLayer, inBlocked;
    mutable uint32_t floodGen = 0;
    mutable std::vector<size_t> floodStack;
    mutable std::vector<uint8_t> snKnown;
    mutable std::vector<size_t> snBegin;
    mutable std::vector<uint32_t> snCount, snArena;
    mutable size_t snCursor = 0;
    mutable std::vector<uint32_t> inComp, inCompBlocked, inTerminal, compBlocked, compStack, compAtoms, compTerminals;
    mutable uint32_t compGen = 0;
    std::vector<size_t> layerOrder;
    std::vector<uint8_t> inLV;
    // witness -> atoms it locally blocks (singly linked, entries may be stale)
    mutable std::vector<int32_t> depHead, depNext;
    mutable std::vector<uint32_t> depAtom;
    void addDependent(size_t witness, size_t atom) const;
    uint32_t localBlocker(size_t i, std::vector<uint32_t> &buf, std::vector<uint32_t> &stack) const;
    std::vector<uint32_t> cands, candWitness;
    void updateViables(const std::unordered_set<size_t> &layer, std::unordered_set<size_t> &locallyViables);
    static constexpr uint32_t NONE32 = 0xffffffffu;
    struct RankJ { uint32_t rank; uint32_t j; uint32_t pos = 0; RankJ(uint32_t r, uint32_t jj) : rank(r), j(jj) {} };
    std::vector<RankJ> rankOf;
    struct KeyK { uint64_t key; uint32_t k; };
    std::vector<KeyK> gridKeys;
    std::vector<uint32_t> gridSlots;
    std::vector<uint32_t> verifySeq, newViables, layerPos, candStamp, oldWitness, posStamp, posInQuery;
    uint32_t updGen = 0, posGen = 0;
    std::vector<std::vector<uint32_t>> updLists;
    std::vector<uint32_t> updOffsets;
    float nozzleConeAngle;
    bool upCollisionCheck;
};