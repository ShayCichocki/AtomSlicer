// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include <functional>
#include <array>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>

// Metal-port change: the hot loops (For, ForAny, ForAny2, ArgMin) are templates so the
// per-element body is inlined; only one std::function call per chunk remains.
// Chunking (one equal chunk per thread) is unchanged, so results are identical.
namespace Parallel {
    bool inWorker();
    // Ranges this small run inline on the calling thread: waking the pool costs more than the work.
    constexpr size_t kSerialBelow = 256;

    namespace detail {
        size_t threadCount();
        void runChunks(size_t chunks, const std::function<void(size_t)> &chunk);
    }

    template <class F>
    void For(size_t start, size_t stop, F &&foo) {
        if (stop - start < kSerialBelow) {
            for (size_t i = start; i < stop; ++i) foo(i);
            return;
        }
        size_t N = detail::threadCount();
        size_t grain = (stop-start)/N + ((stop-start)%N ? 1 : 0);
        detail::runChunks(N, [&](size_t t) {
            size_t begin = start+t*grain, end = std::min(start+(t+1)*grain, stop);
            for (size_t i = begin; i < end; ++i) {
                foo(i);
            }
        });
    }

    // Pool size (ATOM_THREADS=n caps it; default hardware_concurrency).
    inline size_t threadCount() { return detail::threadCount(); }
    // True on a pool worker thread; Parallel calls made there run serially (same chunks).
    bool inWorker();
    // Runs task(0..count-1) as independent pool tasks (dynamic scheduling: an idle worker
    // takes the next one). Results must be written per index; the caller orders them.
    void ForEachTask(size_t count, const std::function<void(size_t)> &task);

    void ForProgress(size_t start, size_t stop, std::function<void(size_t)> foo, bool quiet = false);
    void ForGrainProgress(size_t start, size_t stop, std::function<void(size_t)> foo, size_t grain = 4096, bool quiet = false);

    template <class F>
    bool ForAny(size_t start, size_t stop, F &&foo) {
        if (stop - start < kSerialBelow) {
            bool res = false;
            for (size_t i = start; i < stop; ++i) res |= foo(i);
            return res;
        }
        size_t N = detail::threadCount();
        size_t grain = (stop-start)/N + ((stop-start)%N ? 1 : 0);
        std::vector<uint8_t> booleans(N);
        detail::runChunks(N, [&](size_t t) {
            size_t begin = start+t*grain, end = std::min(start+(t+1)*grain, stop);
            bool res = false;
            for (size_t i = begin; i < end; ++i) {
                res |= foo(i);
            }
            booleans[t] = res;
        });
        bool result = false;
        for (auto b : booleans) result |= b;
        return result;
    }

    template <class F>
    std::array<bool, 2> ForAny2(size_t start, size_t stop, F &&foo) {
        if (stop - start < kSerialBelow) {
            std::array<bool, 2> res{false, false};
            for (size_t i = start; i < stop; ++i) { auto v = foo(i); res[0] |= v[0]; res[1] |= v[1]; }
            return res;
        }
        size_t N = detail::threadCount();
        size_t grain = (stop-start)/N + ((stop-start)%N ? 1 : 0);
        if (inWorker()) {
            // Toolpath-speedup change: run inline on a worker in lockstep order (element k of
            // every chunk, then k+1, ...), i.e. the order the N concurrent chunks take on the
            // pool. Its one caller (Stitcher::connectRegion) writes, in a rare branch, a value
            // that other elements of the same pass read, so the order matters: chunk-by-chunk
            // order gave a 0.1 mm road change on one hook layer, lockstep matches the pool.
            std::array<bool, 2> result{false, false};
            for (size_t k = 0; k < grain; ++k) {
                for (size_t t = 0; t < N; ++t) {
                    size_t i = start + t*grain + k;
                    if (i >= stop) continue;
                    auto v = foo(i);
                    result[0] |= v[0]; result[1] |= v[1];
                }
            }
            return result;
        }
        std::vector<uint8_t> booleans0(N), booleans1(N);
        detail::runChunks(N, [&](size_t t) {
            size_t begin = start+t*grain, end = std::min(start+(t+1)*grain, stop);
            bool res0 = false, res1 = false;
            for (size_t i = begin; i < end; ++i) {
                auto val = foo(i);
                res0 |= val[0];
                res1 |= val[1];
            }
            booleans0[t] = res0;
            booleans1[t] = res1;
        });
        std::array<bool, 2> result{false, false};
        for (size_t i = 0; i < N; ++i) {
            result[0] |= booleans0[i];
            result[1] |= booleans1[i];
        }
        return result;
    }

    template <class F>
    size_t ArgMin(size_t start, size_t stop, F &&foo) {
        struct FS { float f; size_t s; };
        if (stop - start < kSerialBelow) {
            float val = std::numeric_limits<float>::infinity();
            size_t id = 0;
            for (size_t i = start; i < stop; ++i) {
                float newVal = foo(i);
                if (newVal < val) { val = newVal; id = i; }
            }
            return id;
        }
        size_t N = detail::threadCount();
        size_t grain = (stop-start)/N + ((stop-start)%N ? 1 : 0);
        std::vector<FS> values(N);
        detail::runChunks(N, [&](size_t t) {
            size_t begin = start+t*grain, end = std::min(start+(t+1)*grain, stop);
            float val = std::numeric_limits<float>::infinity();
            size_t id = 0;
            for (size_t i = begin; i < end; ++i) {
                float newVal = foo(i);
                if (newVal < val) {
                    val = newVal;
                    id = i;
                }
            }
            values[t] = {val, id};
        });
        FS result{std::numeric_limits<float>::infinity(), 0};
        for (auto v : values) {
            if (v.f < result.f) result = v;
        }
        return result.s;
    }
}
