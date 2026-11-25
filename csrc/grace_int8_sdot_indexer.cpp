// APEX Blackwell Next-Gen — Grace Neoverse-V2 1.41 TB/s INT8 sdot Coherent Page Indexer
// Compiles cleanly with: g++ -O3 -mcpu=native -shared -fPIC grace_int8_sdot_indexer.cpp -o libgrace_sdot.so
#include <arm_neon.h>
#include <sched.h>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <stdint.h>

static void pin_socket(int tid, int socket) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((socket == 0 ? 0 : 70) + (tid % 70), &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

struct ScoreIdx { float s; int32_t id; };

struct GraceSdotPool {
    int nt, socket, sub_shift;
    std::vector<std::thread> workers;
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> done_cnt{0};
    std::atomic<bool> stop{false};
    const int8_t* centroids = nullptr;
    const float* scales = nullptr;
    int8x16_t qv[8];
    int n_sub = 0, K_pages = 128, k_loc = 16;
    std::vector<std::vector<ScoreIdx>> per_thr;

    GraceSdotPool(int n_threads, int sock, int sub_per_page_shift = 2)
        : nt(n_threads), socket(sock), sub_shift(sub_per_page_shift), per_thr(n_threads) {
        for (int t = 0; t < nt; t++) {
            per_thr[t].reserve(256);
            workers.emplace_back([this, t]() {
                pin_socket(t, socket);
                uint64_t last_ep = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    uint64_t ep = epoch.load(std::memory_order_acquire);
                    if (ep == last_ep) continue;
                    last_ep = ep;
                    int p0 = (n_sub * t) / nt, p1 = (n_sub * (t + 1)) / nt;
                    auto& heap = per_thr[t];
                    heap.clear();
                    float min_s = -1e30f;
                    int kl = k_loc;
                    for (int p = p0; p < p1; p++) {
                        const int8_t* c = centroids + (size_t)p * 128;
                        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                        a0 = vdotq_s32(a0, vld1q_s8(c +  0), qv[0]);
                        a1 = vdotq_s32(a1, vld1q_s8(c + 16), qv[1]);
                        a0 = vdotq_s32(a0, vld1q_s8(c + 32), qv[2]);
                        a1 = vdotq_s32(a1, vld1q_s8(c + 48), qv[3]);
                        a0 = vdotq_s32(a0, vld1q_s8(c + 64), qv[4]);
                        a1 = vdotq_s32(a1, vld1q_s8(c + 80), qv[5]);
                        a0 = vdotq_s32(a0, vld1q_s8(c + 96), qv[6]);
                        a1 = vdotq_s32(a1, vld1q_s8(c +112), qv[7]);
                        float sc = (float)vaddvq_s32(vaddq_s32(a0, a1)) * scales[p];
                        int32_t parent_page = p >> sub_shift;
                        if ((int)heap.size() < kl) {
                            heap.push_back({sc, parent_page});
                            if ((int)heap.size() == kl) {
                                std::make_heap(heap.begin(), heap.end(),
                                    [](const ScoreIdx& x, const ScoreIdx& y){ return x.s > y.s; });
                                min_s = heap.front().s;
                            }
                        } else if (sc > min_s) {
                            std::pop_heap(heap.begin(), heap.end(),
                                [](const ScoreIdx& x, const ScoreIdx& y){ return x.s > y.s; });
                            heap.back() = {sc, parent_page};
                            std::push_heap(heap.begin(), heap.end(),
                                [](const ScoreIdx& x, const ScoreIdx& y){ return x.s > y.s; });
                            min_s = heap.front().s;
                        }
                    }
                    done_cnt.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }

    void run(const int8_t* c, const float* sc, const int8_t* q, int32_t* out_pids, int ns, int kp) {
        centroids = c; scales = sc; n_sub = ns; K_pages = kp;
        k_loc = std::min(kp, std::max(8, (kp * 2) / std::max(1, nt) + 4));
        for (int i = 0; i < 8; i++) qv[i] = vld1q_s8(q + i * 16);
        done_cnt.store(0, std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_release);
        while (done_cnt.load(std::memory_order_acquire) < nt) {}
        std::vector<ScoreIdx> merged;
        merged.reserve(nt * k_loc);
        for (int t = 0; t < nt; t++) merged.insert(merged.end(), per_thr[t].begin(), per_thr[t].end());
        if ((int)merged.size() > kp)
            std::nth_element(merged.begin(), merged.begin() + kp, merged.end(),
                             [](const ScoreIdx& x, const ScoreIdx& y){ return x.s > y.s; });
        int n_out = std::min(kp, (int)merged.size());
        for (int i = 0; i < n_out; i++) out_pids[i] = merged[i].id;
    }

    ~GraceSdotPool() {
        stop.store(true);
        epoch.fetch_add(1);
        for (auto& th : workers) th.join();
    }
};

extern "C" {
void* grace_sdot_pool_create(int nt, int sock, int sub_shift) { return new GraceSdotPool(nt, sock, sub_shift); }
void  grace_sdot_pool_run(void* p, const int8_t* c, const float* sc, const int8_t* q, int32_t* out_pids, int ns, int kp) {
    ((GraceSdotPool*)p)->run(c, sc, q, out_pids, ns, kp);
}
void  grace_sdot_pool_destroy(void* p) { delete (GraceSdotPool*)p; }
}
