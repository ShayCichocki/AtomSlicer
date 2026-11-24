// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "parallel.h"
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Toolpath-speedup change: worker threads mark themselves, so a Parallel call made from
// inside a pool task (e.g. per-layer Curve work run by ForEachTask) runs its chunks
// serially, in chunk order, on that worker instead of deadlocking on the shared pool.
// The chunk boundaries are unchanged, so per-chunk reductions give the same results.
static thread_local bool tlInWorker = false;

class ThreadPool {
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) : pendingTask_(0), noPendingTask_(true) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] {
                tlInWorker = true;
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this] {
                            return !tasks_.empty() || stop_;
                        });
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                    { 
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        --pendingTask_;
                        if (pendingTask_ == 0) {
                            noPendingTask_ = true;
                            cvPending_.notify_one();
                        }
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(task));
            ++pendingTask_;
            noPendingTask_ = false;
        }
        cv_.notify_one();
    }

    // Metal-port change: push a whole batch under one lock and wake all workers once,
    // instead of one lock + notify_one per task.
    void enqueueBatch(size_t count, const std::function<void(size_t)> &chunk) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            for (size_t t = 0; t < count; ++t) {
                tasks_.emplace([t, &chunk]() { chunk(t); });
            }
            pendingTask_ += count;
            noPendingTask_ = count == 0 && pendingTask_ == 0;
        }
        cv_.notify_all();
    }

    void sync() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cvPending_.wait(lock, [this]{ return noPendingTask_; });
    }

    size_t threadCount() {
        return threads_.size();
    }

private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()> > tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    size_t pendingTask_;
    bool noPendingTask_ = true;
    std::condition_variable cvPending_;
};

// ATOM_THREADS=n caps the pool (default unchanged: hardware_concurrency).
static size_t poolSize() {
    const char *e = getenv("ATOM_THREADS");
    long n = e ? atol(e) : 0;
    return n > 0 ? size_t(n) : std::thread::hardware_concurrency();
}

static ThreadPool &getThreadPool() {
    static ThreadPool pool(poolSize());
    return pool;
}

size_t Parallel::detail::threadCount() {
    return getThreadPool().threadCount();
}

bool Parallel::inWorker() {
    return tlInWorker;
}

void Parallel::ForEachTask(size_t count, const std::function<void(size_t)> &task) {
    if (tlInWorker || count <= 1) {
        for (size_t i = 0; i < count; ++i) task(i);
        return;
    }
    auto &pool = getThreadPool();
    pool.enqueueBatch(count, task);
    pool.sync();
}

void Parallel::detail::runChunks(size_t N, const std::function<void(size_t)> &chunk) {
    if (tlInWorker) {
        for (size_t t = 0; t < N; ++t) chunk(t);
        return;
    }
    auto &pool = getThreadPool();
    pool.enqueueBatch(N, chunk);
    pool.sync();
}

void Parallel::ForProgress(size_t start, size_t stop, std::function<void(size_t)> foo, bool quiet) {
    if (tlInWorker) {
        for (size_t i = start; i < stop; ++i) foo(i);
        return;
    }
    auto &pool = getThreadPool();
    size_t N = pool.threadCount();
    size_t grain = (stop-start)/N + ((stop-start)%N ? 1 : 0);
    for (size_t t = 0; t < N; ++t) {
        size_t begin = start+t*grain, end = std::min(start+(t+1)*grain, stop);
        auto task = [begin, end, &foo, t, quiet]() {
            for (size_t i = begin; i < end; ++i) {
                foo(i);
                if (!quiet && !t) {
                    static uint8_t last = 0xff;
                    float percentage = float(i-begin+1)/(end-begin);
                    if (uint8_t(percentage*100.0f) != last) {
                        last = uint8_t(percentage*100.0f);
                        char buff[64];
                        const char* bar = "||||||||||||||||||||||||||||||||||||||||";
                        const uint32_t barWidth = strlen(bar);
                        uint32_t val = (uint32_t) (percentage * 100);
                        uint32_t lpad = (uint32_t) (percentage * barWidth);
                        uint32_t rpad = barWidth - lpad;
                        sprintf(buff, "\r%3d%% [%.*s%*s]", val, lpad, bar, rpad, "");
                        std::cout << buff << std::flush;
                    }
                    if (percentage == 1.0f) {
                        std::cout << std::endl;
                        last = 0xff;
                    }
                }
            }
        };
        pool.enqueue(task);
    }
    pool.sync();
}

void Parallel::ForGrainProgress(size_t start, size_t stop, std::function<void(size_t)> foo, size_t grain, bool quiet) {
    if (tlInWorker) {
        for (size_t i = start; i < stop; ++i) foo(i);
        return;
    }
    auto &pool = getThreadPool();
    size_t N = (stop-start)/grain + ((stop-start)%grain ? 1 : 0);
    size_t done = 0;
    std::mutex progressMutex;
    for (size_t t = 0; t < N; ++t) {
        size_t begin = start+t*grain, end = std::min(start+(t+1)*grain, stop);
        auto task = [begin, end, &foo, t, N, &done, &progressMutex, quiet]() {
            for (size_t i = begin; i < end; ++i) {
                foo(i);
            }
            if (!quiet) {
                std::unique_lock<std::mutex> lock(progressMutex);
                ++done;
                static uint8_t last = 0xff;
                float percentage = float(done)/N;
                if (uint8_t(percentage*100.0f) != last) {
                    last = uint8_t(percentage*100.0f);
                    char buff[64];
                    const char* bar = "||||||||||||||||||||||||||||||||||||||||";
                    const uint32_t barWidth = strlen(bar);
                    uint32_t val = (uint32_t) (percentage * 100);
                    uint32_t lpad = (uint32_t) (percentage * barWidth);
                    uint32_t rpad = barWidth - lpad;
                    sprintf(buff, "\r%3d%% [%.*s%*s]", val, lpad, bar, rpad, "");
                    std::cout << buff << std::flush;
                }
                if (percentage == 1.0f) {
                    std::cout << std::endl;
                    last = 0xff;
                }
            }
        };
        pool.enqueue(task);
    }
    pool.sync();
}

