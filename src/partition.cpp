// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "partition.h"
#include "parallel.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <chrono>

// partition-wip: ATOM_PART_PROFILE=1 prints where the time went (wall clock per phase).
namespace {
struct PhaseClock {
    using C = std::chrono::steady_clock;
    bool on; C::time_point t;
    double acc[8] = {0};
    size_t n[8] = {0};
    PhaseClock() : on(getenv("ATOM_PART_PROFILE") != nullptr) {}
    void start() { if (on) t = C::now(); }
    void stop(int k) { if (on) { auto u = C::now(); acc[k] += std::chrono::duration<double>(u - t).count(); n[k]++; t = u; } }
};
}

void Partition::savePoints(const std::unordered_set<size_t> &layer, const std::string &path) const {
    std::ofstream plyFile(path);
    plyFile << "ply\nformat ascii 1.0\nelement vertex " << layer.size() << "\n"
            << "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
            << "end_header\n";
        
    for (auto i : layer) {
        plyFile << points[i][0] << " " << points[i][1] << " " << points[i][2] << std::endl;
    }
}

bool Partition::isNotLocallyBlocked(size_t i) const {
    const float cosHalfAngle = cosf(0.5f*nozzleConeAngle);
    bvh->pointsInSphereIndex(points[i], 2.0f*layerHeight, nbLocal);
    for (size_t j : nbLocal) {
        if (i == j) continue;
        Vec3 v = points[j]-points[i];
        float length = v.length();
        Vec3 n = !upCollisionCheck ? normals[i] : Vec3(0, 0, 1);
        float cosAngle = v.dot(n)/length;
        if (length <= 2.0f*layerHeight && cosAngle >= cosHalfAngle) {
            return false;
        }
    }
    return true;
}

size_t Partition::blockerOf(size_t i) const {
    // An empty cone stays empty (atoms only leave), whatever the tree: no query needed.
    if (blockerVersion[i] == treeVersion || (blockerVersion[i] && blockerId[i] == PartBVH::NONE)) {
        size_t b = blockerId[i] == PartBVH::NONE ? size_t(-1) : size_t(blockerId[i]);
        if (verify && b != getBlocker(i)) { std::cerr << "partition verify: blockerOf mismatch\n"; abort(); }
        return b;
    }
    size_t b = getBlocker(i);
    blockerId[i] = b == size_t(-1) ? PartBVH::NONE : uint32_t(b);
    blockerVersion[i] = treeVersion;
    return b;
}

bool Partition::isBlocked(size_t i) const {
    bool res;
    if (blockerVersion[i] == treeVersion) {
        res = blockerId[i] != PartBVH::NONE;
    } else if (blockerVersion[i] && blockerId[i] == PartBVH::NONE) {
        res = false;                          // the cone was empty and atoms only leave
    } else if (blockerVersion[i] && !bvh->isRemoved(blockerId[i])) {
        res = true;                           // the old blocker is still there
    } else {
        return blockerOf(i) != size_t(-1);
    }
    if (verify && res != (getBlocker(i) != size_t(-1))) { std::cerr << "partition verify: isBlocked mismatch\n"; abort(); }
    return res;
}

bool Partition::notLocallyBlocked(size_t i) const {
    uint32_t w = localWitness[i];
    bool res;
    if (w == 1) {
        res = true;
    } else if (w >= 2 && !bvh->isRemoved(w - 2)) {
        res = false;
    } else {
        // Same test as isNotLocallyBlocked, remembering the first blocking atom.
        const float cosHalfAngle = cosf(0.5f*nozzleConeAngle);
        bvh->pointsInSphereIndex(points[i], 2.0f*layerHeight, nbLocal);
        for (size_t j : nbLocal) {
            if (i == j) continue;
            Vec3 v = points[j]-points[i];
            float length = v.length();
            Vec3 n = !upCollisionCheck ? normals[i] : Vec3(0, 0, 1);
            float cosAngle = v.dot(n)/length;
            if (length <= 2.0f*layerHeight && cosAngle >= cosHalfAngle) {
                localWitness[i] = uint32_t(j) + 2;
                depHead.size() ? addDependent(j, i) : void();
                return false;
            }
        }
        localWitness[i] = 1;
        return true;
    }
    if (verify && res != isNotLocallyBlocked(i)) { std::cerr << "partition verify: notLocallyBlocked mismatch\n"; abort(); }
    return res;
}

// isNotLocallyBlocked's loop, returning the first blocking atom (NONE32 if none); thread-safe.
uint32_t Partition::localBlocker(size_t i, std::vector<uint32_t> &buf, std::vector<uint32_t> &stack) const {
    const float cosHalfAngle = cosf(0.5f*nozzleConeAngle);
    buf.clear();
    bvh->pointsInSphereAppend(points[i], 2.0f*layerHeight, buf, stack);
    for (size_t j : buf) {
        if (i == j) continue;
        Vec3 v = points[j]-points[i];
        float length = v.length();
        Vec3 n = !upCollisionCheck ? normals[i] : Vec3(0, 0, 1);
        float cosAngle = v.dot(n)/length;
        if (length <= 2.0f*layerHeight && cosAngle >= cosHalfAngle) {
            return uint32_t(j);
        }
    }
    return NONE32;
}

void Partition::addDependent(size_t witness, size_t atom) const {
    depAtom.push_back(uint32_t(atom));
    depNext.push_back(depHead[witness]);
    depHead[witness] = int32_t(depAtom.size() - 1);
}

void Partition::updateViables(const std::unordered_set<size_t> &layer, std::unordered_set<size_t> &locallyViables) {
    layerOrder.assign(layer.begin(), layer.end());
    ++updGen;
    for (size_t k = 0; k < layerOrder.size(); ++k) layerPos[layerOrder[k]] = uint32_t(k);
    newViables.clear();
    cands.clear();
    for (size_t w : layerOrder) {
        for (int32_t node = depHead[w]; node >= 0; node = depNext[node]) {
            uint32_t j = depAtom[node];
            if (localWitness[j] != w + 2 || bvh->isRemoved(j) || inLV[j] || candStamp[j] == updGen) continue;
            candStamp[j] = updGen;
            oldWitness[j] = uint32_t(w);
            cands.push_back(j);
        }
        depHead[w] = -1;   // w is gone; its dependents have moved on
    }
    // Their witness is gone, so isNotLocallyBlocked must be evaluated again: a pure read of the
    // tree per atom, done in parallel; the results are applied in candidate order.
    candWitness.resize(cands.size());
    {
        const size_t C = cands.size();
        const size_t tasks = C < 64 ? 1 : std::min<size_t>(4*Parallel::threadCount(), (C+15)/16);
        auto run = [&](size_t t) {
            std::vector<uint32_t> buf, stack;
            for (size_t q = C*t/tasks, e = C*(t+1)/tasks; q < e; ++q) candWitness[q] = localBlocker(cands[q], buf, stack);
        };
        if (tasks == 1) run(0); else Parallel::ForEachTask(tasks, run);
    }
    for (size_t q = 0; q < cands.size(); ++q) {
        uint32_t j = cands[q], w2 = candWitness[q];
        if (w2 == NONE32) {
            localWitness[j] = 1;
            newViables.push_back(j);
        } else {
            localWitness[j] = w2 + 2;
            addDependent(w2, j);
        }
        if (verify && (w2 == NONE32) != isNotLocallyBlocked(j)) { std::cerr << "partition verify: localBlocker mismatch\n"; abort(); }
    }
    if (verify) {
        // the original walk, for comparison
        verifySeq.clear();
        ++posGen;
        for (size_t i : layerOrder) {
            bvh->pointsInSphereIndex(points[i], 2.0f*layerHeight + 1e-3, nbUpdate);
            for (uint32_t j : nbUpdate) {
                if (!inLV[j] && posStamp[j] != posGen && notLocallyBlocked(j)) { posStamp[j] = posGen; verifySeq.push_back(j); }
            }
        }
    }
    if (newViables.empty()) {
        if (verify && verifySeq.size()) { std::cerr << "partition verify: updateViables missed atoms\n"; abort(); }
        return;
    }
    // rank = first layer atom whose sphere (the original's test, same expression) holds j
    const float radius = 2.0f*layerHeight + 1e-3;
    const float radius2 = radius*radius;
    rankOf.clear();
    // Candidates come from a grid of the layer atoms (cells a bit larger than the radius, so
    // every atom within the radius is in the 27 cells around j); the test itself is the
    // original's expression.
    const float cell = radius * 1.01f;
    auto key = [&](int64_t cx, int64_t cy, int64_t cz) { return uint64_t((cx + (1<<20)) | ((cy + (1<<20)) << 21) | ((cz + (1<<20)) << 42)); };
    auto cellOf = [&](const Vec3 &p, int64_t &cx, int64_t &cy, int64_t &cz) { cx = int64_t(floorf(p.x/cell)); cy = int64_t(floorf(p.y/cell)); cz = int64_t(floorf(p.z/cell)); };
    gridKeys.clear();
    for (uint32_t k = 0; k < layerOrder.size(); ++k) {
        int64_t cx, cy, cz;
        cellOf(points[layerOrder[k]], cx, cy, cz);
        gridKeys.push_back({key(cx, cy, cz), k});
    }
    std::sort(gridKeys.begin(), gridKeys.end(), [](const KeyK &a, const KeyK &b) { return a.key < b.key || (a.key == b.key && a.k < b.k); });
    // open-addressing index: cell key -> first entry in gridKeys
    size_t cap = 64;
    while (cap < 2*gridKeys.size()) cap <<= 1;
    gridSlots.assign(cap, NONE32);
    auto slotOf = [&](uint64_t kk) { return size_t((kk * 0x9E3779B97F4A7C15ull) >> 20) & (cap - 1); };
    for (size_t q = 0; q < gridKeys.size(); ++q) {
        if (q && gridKeys[q].key == gridKeys[q-1].key) continue;
        size_t sl = slotOf(gridKeys[q].key);
        while (gridSlots[sl] != NONE32) sl = (sl + 1) & (cap - 1);
        gridSlots[sl] = uint32_t(q);
    }
    auto findCell = [&](uint64_t kk) -> uint32_t {
        for (size_t sl = slotOf(kk); gridSlots[sl] != NONE32; sl = (sl + 1) & (cap - 1)) {
            if (gridKeys[gridSlots[sl]].key == kk) return gridSlots[sl];
        }
        return NONE32;
    };
    for (uint32_t j : newViables) {
        int64_t cx, cy, cz;
        cellOf(points[j], cx, cy, cz);
        uint32_t rank = NONE32;
        for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            uint64_t kk = key(cx+dx, cy+dy, cz+dz);
            uint32_t first = findCell(kk);
            if (first == NONE32) continue;
            for (auto it = gridKeys.begin() + first; it != gridKeys.end() && it->key == kk; ++it) {
                if (it->k >= rank) break;   // sorted by k inside a cell
                float distance2 = (points[j]-points[layerOrder[it->k]]).length2();
                if (distance2 < radius2) { rank = it->k; break; }
            }
        }
        if (rank == NONE32) { std::cerr << "partition: updateViables found no sphere for a new viable atom\n"; abort(); }
        rankOf.push_back({rank, j});
    }
    std::stable_sort(rankOf.begin(), rankOf.end(), [](const RankJ &a, const RankJ &b) { return a.rank < b.rank; });
    for (size_t s0 = 0; s0 < rankOf.size(); ) {
        size_t s1 = s0 + 1;
        while (s1 < rankOf.size() && rankOf[s1].rank == rankOf[s0].rank) ++s1;
        if (s1 - s0 > 1) {
            // order inside the sphere query of that layer atom
            bvh->pointsInSphereIndex(points[layerOrder[rankOf[s0].rank]], 2.0f*layerHeight + 1e-3, nbUpdate);
            ++posGen;
            for (size_t q = 0; q < nbUpdate.size(); ++q) { posStamp[nbUpdate[q]] = posGen; posInQuery[nbUpdate[q]] = uint32_t(q); }
            for (size_t q = s0; q < s1; ++q) {
                if (posStamp[rankOf[q].j] != posGen) { std::cerr << "partition: updateViables sphere mismatch\n"; abort(); }
                rankOf[q].pos = posInQuery[rankOf[q].j];
            }
            std::sort(rankOf.begin() + s0, rankOf.begin() + s1, [](const RankJ &a, const RankJ &b) { return a.pos < b.pos; });
        }
        s0 = s1;
    }
    if (verify) {
        bool ok = verifySeq.size() == rankOf.size();
        for (size_t q = 0; ok && q < rankOf.size(); ++q) ok = verifySeq[q] == rankOf[q].j;
        if (!ok) { std::cerr << "partition verify: updateViables order mismatch\n"; abort(); }
    }
    for (const RankJ &r : rankOf) {
        locallyViables.insert(r.j);
        inLV[r.j] = 1;
    }
}

bool Partition::isNeighboor(size_t i, size_t j) const {
    Vec3 v = points[j]-points[i];
    return fabsf(v.dot(normals[i])) < 0.5f*layerHeight && fabsf(v.dot(normals[j])) < 0.5f*layerHeight && v.length() <= 1.5f*layerHeight;
}

size_t Partition::getBlocker(size_t i) const {
    Vec3 n = !upCollisionCheck ? normals[i] : Vec3(0, 0, 1);
    uint32_t b = bvh->anyPointsInCone(points[i]+n*1e-3, n, nozzleConeAngle * 0.5);
    return b == PartBVH::NONE ? size_t(-1) : size_t(b);
}

// Neighbours of i that flood() may step to: the sphere query in its order, filtered by
// isNeighboor (a pure function of the two atoms, tested first in flood's condition apart from
// the set lookups, so filtering up front gives the same sequence). Cached per tree version.
void Partition::floodNeighbours(size_t i, uint32_t &begin, uint32_t &count) const {
    if (nbVersion[i] != treeVersion) {
        bvh->pointsInSphereIndex(points[i], 1.5f*layerHeight, nbFlood);
        nbVersion[i] = treeVersion;
        nbBegin[i] = uint32_t(nbArena.size());
        for (size_t j : nbFlood) {
            if (isNeighboor(i, j)) nbArena.push_back(uint32_t(j));
        }
        nbCount[i] = uint32_t(nbArena.size()) - nbBegin[i];
    }
    begin = nbBegin[i];
    count = nbCount[i];
}

// Neighbours of i as a set, in no particular order: atoms alive when first asked for, minus
// those removed since (filter with bvh->isRemoved). Same membership as floodNeighbours in any
// tree version, because the sphere test and isNeighboor are pure functions of the atoms and
// atoms only ever leave.
void Partition::setNeighbours(size_t i, uint32_t &begin, uint32_t &count) const {
    if (!snKnown[i]) {
        bvh->pointsInSphereIndex(points[i], 1.5f*layerHeight, nbFlood);
        snBegin[i] = snArena.size();
        for (size_t j : nbFlood) {
            if (isNeighboor(i, j)) snArena.push_back(uint32_t(j));
        }
        snCount[i] = uint32_t(snArena.size() - snBegin[i]);
        snKnown[i] = 1;
    }
    begin = uint32_t(0);
    count = snCount[i];
    snCursor = snBegin[i];
}

// The loop branch's test of one start: flood(start) then getBlockers, answering only "is the
// blocker set empty?" and listing the flooded atoms (as a set). flood's layer is the connected
// component of `start` among unblocked atoms (isNeighboor is symmetric), and whether
// getBlockers finds a blocker does not depend on the order the blocked atoms are visited in
// (a chain stops early on an existing blocker only once the set is already non-empty), so
// neither depends on the order neighbours are visited in. Every chain is still walked to its
// end so the "bidirectional constraint" exit happens exactly when getBlockers would hit it.
bool Partition::componentAccessible(size_t start, std::vector<uint32_t> &comp) const {
    ++compGen;
    comp.clear();
    std::vector<uint32_t> &blockedList = compBlocked;
    blockedList.clear();
    std::vector<uint32_t> &stack = compStack;
    stack.clear();
    comp.push_back(uint32_t(start));
    inComp[start] = compGen;
    stack.push_back(uint32_t(start));
    while (stack.size()) {
        uint32_t i = stack.back();
        stack.pop_back();
        uint32_t b0, count;
        setNeighbours(i, b0, count);
        const size_t base = snCursor;
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t j = snArena[base + k];
            if (bvh->isRemoved(j)) continue;
            if (inComp[j] == compGen || inCompBlocked[j] == compGen) continue;
            if (!isBlocked(j)) {
                inComp[j] = compGen;
                comp.push_back(j);
                stack.push_back(j);
            } else {
                inCompBlocked[j] = compGen;
                blockedList.push_back(j);
            }
        }
    }
    bool accessible = true;
    compTerminals.clear();
    for (uint32_t blockedAtom : blockedList) {
        size_t blockerAtom = blockerOf(blockedAtom);
        size_t attempted = 0;
        while (inCompBlocked[blockerAtom] != compGen && inComp[blockerAtom] != compGen) {
            size_t nextBlocker = blockerOf(blockerAtom);
            if (attempted++ > points.size()) {
                std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
                exit(1);
            }
            if (nextBlocker == size_t(-1)) {
                accessible = false;
                // getBlockers would insert this atom into `blocker` (once)
                if (inTerminal[blockerAtom] != compGen) {
                    inTerminal[blockerAtom] = compGen;
                    compTerminals.push_back(uint32_t(blockerAtom));
                }
                break;
            }
            blockerAtom = nextBlocker;
        }
    }
    return accessible;
}

void Partition::flood(size_t start, std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked) const {
    layer.clear();
    blocked.clear();
    // Membership mirrors of `layer` and `blocked` for this flood (and the getBlockers call
    // that follows it), so the hash sets are only written, in the same order as before.
    ++floodGen;
    std::vector<size_t> &stack = floodStack;
    stack.clear();

    layer.insert(start);
    inLayer[start] = floodGen;
    stack.push_back(start);
    while (stack.size()) {
        size_t i = stack.back();
        stack.pop_back();
        uint32_t begin, count;
        floodNeighbours(i, begin, count);
        for (uint32_t k = 0; k < count; ++k) {
            size_t j = nbArena[begin + k];
            if (inLayer[j] != floodGen && inBlocked[j] != floodGen) {
                if (!isBlocked(j)) {
                    layer.insert(j);
                    inLayer[j] = floodGen;
                    stack.push_back(j);
                } else {
                    blocked.insert(j);
                    inBlocked[j] = floodGen;
                }
            }
        }
    }
}

void Partition::getBlockers(std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked, std::unordered_set<size_t> &blocker) const {
    blocker.clear();
    // layer/blocked are the sets the flood() just before filled; their mirrors are current.
    if (verify && (layer.size() != countGen(inLayer) || blocked.size() != countGen(inBlocked))) { std::cerr << "partition verify: flood mirrors stale\n"; abort(); }
    for (auto blockedAtom : blocked) {
        size_t blockerAtom = blockerOf(blockedAtom);
        size_t attempted = 0;
        while (!blocker.count(blockerAtom) && inBlocked[blockerAtom] != floodGen && inLayer[blockerAtom] != floodGen) {
            size_t nextBlocker = blockerOf(blockerAtom);
            if (attempted++ > points.size()) {
                std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
                exit(1);
            }
            if (nextBlocker == size_t(-1)) {
                blocker.insert(blockerAtom);
                break;
            }
            blockerAtom = nextBlocker;
        }
    }
}

size_t Partition::getHigherBlocker(std::unordered_set<size_t> &blocker) const {
    size_t best = *blocker.begin();
    for (auto atom : blocker) {
        if (points[atom][2] > points[best][2]) {
            best = atom;
        }
    }
    return best;
}

void Partition::saveLayersPLY(const char *path) const {
    for (size_t i = 0; i < layers.size(); ++i) {
        std::ofstream plyFile(std::string(path)+ std::to_string(i) +".ply");
        plyFile << "ply\nformat ascii 1.0\nelement vertex " << layers[i].size() << "\n"
                << "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
                << "end_header\n";
            
        for (auto j : layers[i]) {
            plyFile << points[j][0] << " " << points[j][1] << " " << points[j][2] << std::endl;
        }
    }
}

void Partition::savePartitionToPLY(const char *path) const {
    std::vector<Vec3> pos;
    std::vector<uint32_t> idx;
    std::vector<std::array<uint8_t, 3>> col;

    for (size_t i = 0; i < layers.size(); ++i) {            
        for (auto j : layers[i]) {
            pos.push_back(points[j]);
            idx.push_back(i);
            col.push_back(turbo(float(i)/(layers.size()-1)));
        }
    }

    std::ofstream plyFile(path);   
    plyFile << "ply\nformat ascii 1.0\nelement vertex " << pos.size() << "\n"
            << "property float x\nproperty float y\nproperty float z\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            << "property uint order\n"
            << "end_header\n";
    for (size_t i = 0; i < pos.size(); ++i) {
        plyFile << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << " "
                << uint32_t(col[i][0]) << " " << uint32_t(col[i][1]) << " " << uint32_t(col[i][2]) << " "
                << idx[i] << std::endl;
    }
}

void Partition::saveLayersIndexCSV(const char *path) const {
    std::ofstream layersFile(path);
    for (auto &layer : layers) {
        for (auto i : layer) {
            layersFile << i << std::endl;
        }
        layersFile << "===" << std::endl;
    }
}

Partition::Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const char *indexCSVPath) : points(points), normals(normals) {
    std::ifstream file(indexCSVPath);
    std::string line;
    size_t idx;
    std::vector<size_t> layer;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> idx)) {
            layers.push_back(layer);
            layer.clear();
            continue;
        }
        layer.emplace_back(idx);
    }
}

void Partition::saveLayersCSV(const char *path) const {
    std::ofstream layersFile(path);
    for (auto &layer : layers) {
        for (auto i : layer) {
            layersFile << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << std::endl;
        }
        layersFile << "===" << std::endl;
    }
}

Partition::Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float layerHeight, float nozzleConeAngle, bool upCollisionCheck, bool quiet) : layerHeight(layerHeight), points(points), normals(normals), bvh(new PartBVH(points)), nozzleConeAngle(nozzleConeAngle/180.0f*M_PI), upCollisionCheck(upCollisionCheck) {
    if (!points.size()) return;
    // partition-wip: rebuild the BVH subtrees on the thread pool (same tree, see partbvh.h).
    bvh->parallelWidth = Parallel::threadCount();
    bvh->parallelRunner = [](size_t n, const std::function<void(size_t)> &f) { Parallel::ForEachTask(n, f); };
    bvh->setConeHalfAngle(this->nozzleConeAngle * 0.5);   // the value getBlocker passes
    blockerId.assign(points.size(), PartBVH::NONE);
    blockerVersion.assign(points.size(), 0);
    localWitness.assign(points.size(), 0);
    nbVersion.assign(points.size(), 0);
    nbBegin.assign(points.size(), 0);
    nbCount.assign(points.size(), 0);
    inLayer.assign(points.size(), 0);
    inBlocked.assign(points.size(), 0);
    snKnown.assign(points.size(), 0);
    snBegin.assign(points.size(), 0);
    snCount.assign(points.size(), 0);
    inComp.assign(points.size(), 0);
    inCompBlocked.assign(points.size(), 0);
    inTerminal.assign(points.size(), 0);
    inLV.assign(points.size(), 0);   // mirror of locallyViables membership
    depHead.assign(points.size(), -1);
    layerPos.assign(points.size(), 0);
    candStamp.assign(points.size(), 0);
    oldWitness.assign(points.size(), 0);
    posStamp.assign(points.size(), 0);
    posInQuery.assign(points.size(), 0);
    verify = getenv("ATOM_PART_VERIFY") != nullptr;
    std::unordered_set<size_t> layer;
    std::unordered_set<size_t> blocked;
    std::unordered_set<size_t> blocker;
    std::unordered_set<size_t> locallyViables;
    std::vector<std::unordered_set<size_t>> attemptedLayers;
    size_t added = 0;
    size_t backtracks = 0;

    {
        // isNotLocallyBlocked for every atom: pure reads of the tree, in parallel; then the
        // inserts in index order as before.
        const size_t N = points.size();
        std::vector<uint32_t> first(N);
        const size_t tasks = N < 4096 ? 1 : 4*Parallel::threadCount();
        auto run = [&](size_t t) {
            std::vector<uint32_t> buf, stack;
            for (size_t i = N*t/tasks, e = N*(t+1)/tasks; i < e; ++i) first[i] = localBlocker(i, buf, stack);
        };
        if (tasks == 1) run(0); else Parallel::ForEachTask(tasks, run);
        for (size_t i = 0; i < N; ++i) {
            if (first[i] == NONE32) {
                localWitness[i] = 1;
                locallyViables.insert(i);
                inLV[i] = 1;
            } else {
                localWitness[i] = first[i] + 2;
                addDependent(first[i], i);
            }
            if (verify && (first[i] == NONE32) != isNotLocallyBlocked(i)) { std::cerr << "partition verify: initial localBlocker mismatch\n"; abort(); }
        }
    }

    if (!locallyViables.size()) {
        std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
        exit(1);
    }

    // Start from the higher atom
    size_t startPoint = *locallyViables.begin();
    for (auto i : locallyViables) {
        if (points[i][2] > points[startPoint][2]) {
            if (!isBlocked(i)) {
                startPoint = i;
            }
        }
    }

    PhaseClock clk;
    clk.start();
    clk.stop(7);
    while (locallyViables.size()) {
        clk.start();
        // Flood the layer and keep track of the non viable neighbours
        flood(startPoint, layer, blocked);

        // Get the blockers that are not in the layer
        getBlockers(layer, blocked, blocker);
        clk.stop(0);

        if (getenv("ATOM_PART_TRACE")) fprintf(stderr, "it v%u start %zu layer %zu blocked %zu blocker %zu\n", treeVersion, startPoint, layer.size(), blocked.size(), blocker.size());
        startPoint = size_t(-1);
        if (blocker.size()) {
            bool loop = false;
            for (auto &attempt : attemptedLayers) {
                if (attempt == layer) {
                    loop = true;
                    break;
                }
            }

            clk.stop(1);
            if (loop) {
                // If there is a cycle check if there is a fully accessible layer
                std::unordered_set<size_t> starts(locallyViables);
                std::unordered_set<size_t> layerTMP;
                for (size_t atom : layer) {
                    starts.erase(atom);
                }
                for (const auto &attempted : attemptedLayers) {
                    for (size_t atom : attempted) {
                        starts.erase(atom);
                    }
                }
                
                clk.stop(6);
                size_t loopFloods = 0;
                while (starts.size()) {
                    size_t startPoint = *starts.begin();
                    if (isBlocked(startPoint)) {
                        starts.erase(startPoint);
                        continue;
                    }
                    
                    loopFloods++;
                    if (componentAccessible(startPoint, compAtoms)) {
                        // Redo it the original way for the layer's exact atom order.
                        flood(startPoint, layerTMP, blocked);
                        getBlockers(layerTMP, blocked, blocker);
                        if (verify && (blocker.size() || layerTMP.size() != compAtoms.size())) { std::cerr << "partition verify: componentAccessible mismatch\n"; abort(); }
                        layer = std::move(layerTMP);
                        break;
                    }
                    // The original flood/getBlockers left layerTMP, blocked and blocker holding
                    // this candidate's sets. Their contents are never read again, but their
                    // bucket counts carry over (clear() keeps buckets) and decide the iteration
                    // order of later layers, so give them the same sizes the same way.
                    layerTMP.clear();
                    for (uint32_t a : compAtoms) layerTMP.insert(a);
                    blocked.clear();
                    for (uint32_t a : compBlocked) blocked.insert(a);
                    blocker.clear();
                    for (uint32_t a : compTerminals) blocker.insert(a);
                    if (verify) {
                        std::unordered_set<size_t> tl, tb, tk;
                        tl.rehash(layerTMP.bucket_count()); tb.rehash(blocked.bucket_count()); tk.rehash(blocker.bucket_count());
                        flood(startPoint, tl, tb);
                        getBlockers(tl, tb, tk);
                        if (tl != layerTMP || tb != blocked || tk != blocker) { std::cerr << "partition verify: component sets mismatch\n"; abort(); }
                    }

                    // erase order does not change the iteration order of what is left
                    for (uint32_t atom : compAtoms) {
                        starts.erase(atom);
                    }
                }
                

                clk.n[7] += loopFloods;
                if (getenv("ATOM_PART_TRACE")) fprintf(stderr, "loop v%u floods %zu layer %zu first %zu\n", treeVersion, loopFloods, layer.size(), *layer.begin());
                // Add the current layer if there are not fully accessible ones
                layers.emplace_back(std::vector<size_t>(layer.begin(), layer.end()));
                attemptedLayers.clear();
                clk.stop(2);
            } else {
                // Start from the higher blocker
                startPoint = getHigherBlocker(blocker);
                attemptedLayers.push_back(layer);
            }
        } else {
            layers.emplace_back(std::vector<size_t>(layer.begin(), layer.end()));
            attemptedLayers.clear();
        }

        if (startPoint == size_t(-1)) {
            // Update the BVH and the locally viables
            for (auto i : layer) {
                bvh->removePoint(i);
                locallyViables.erase(i);
                inLV[i] = 0;
            }
            clk.start();
            bvh->compact();
            treeChanged();
            nbArena.clear();
            clk.stop(3);
            // New locally viable atoms. The original queries a sphere of 2h+1e-3 around every
            // layer atom and inserts each atom j that is not viable yet and now passes
            // isNotLocallyBlocked, in order of first appearance. An alive atom that is not
            // viable has a blocking witness w (localWitness); it can only pass now if w was
            // just removed, i.e. w is in this layer, and then j is inside w's sphere. So the
            // candidates are the dependents of the layer atoms. Their insertion order is that
            // of the first layer atom (in layer order) whose sphere holds them, and, only when
            // two share it, the order inside that sphere query, which is then run.
            updateViables(layer, locallyViables);
            clk.stop(4);
            // Continue from the closest to the first atom of the last added layer
            float closestDist = std::numeric_limits<float>::infinity();
            bool found = false;
            for (auto i : locallyViables) {
                float distance = (points[*layer.begin()]-points[i]).length2();
                if (distance < closestDist) {
                    if (!isBlocked(i)) {
                        closestDist = distance;
                        startPoint = i;
                        found = true;
                    }
                }
            }
            if (!found && locallyViables.size()) {
                std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
                exit(1);
            }

            clk.stop(5);
            added += layer.size();
            printProgress(float(added)/points.size());
        } else {
            backtracks++;
        }
    }
    assert(added == points.size());

    // Filter too small layers
    layers.erase(
        std::remove_if(layers.begin(), layers.end(),
            [](const auto& layer) { return layer.size() <= 8; }),
        layers.end()
    );

    std::reverse(layers.begin(), layers.end());
    delete bvh;
    bvh = nullptr;

    if (!quiet) std::cout << "Backtracks: " << backtracks << std::endl;
    if (clk.on) {
        const char *names[] = {"flood+getBlockers", "attempt compare", "loop branch", "compact/rebuild", "viables update", "closest start", "", ""};
        printf("  loop-branch setup   %9.3f s, floods in loop branches %zu\n", clk.acc[6], clk.n[7]);
        for (int k = 0; k < 6; ++k) printf("  %-18s %9.3f s  (%zu)\n", names[k], clk.acc[k], clk.n[k]);
    }
}