// Benchmark for LeMonHash (monotone minimal perfect hashing).
//
// Generates a random sorted set of k 64-bit integers using the same
// algorithm as the sux-rs bench_lcp_mmphf binary: draw k + 2*(k²/(2*2^64))
// values (accounting for expected collisions), sort, dedup, take k.
// Retry if not enough unique values.
//
// Output format matches the rest of the framework:
//   "Construction completed in X.XXX seconds (N keys, Y.YYY ns/key)"
//   "Size: Z Bytes = B.BBB bits/key"
//   "Min: ... Median: ... Max: ... Average: ..."

#include <LeMonHash.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define DO_NOT_OPTIMIZE(value) asm volatile("" : : "r,m"(value) : "memory")

static std::vector<uint64_t> gen_sorted_keys(size_t k, uint64_t seed) {
    std::mt19937_64 rng(seed);

    // Expected collisions for k draws from [0, 2^64): k²/(2*2^64).
    // We add twice that margin.
    __uint128_t k128 = k;
    size_t extra = static_cast<size_t>((k128 * k128) >> 65);
    size_t draw = k + 2 * extra;

    for (;;) {
        std::vector<uint64_t> keys;
        keys.reserve(draw);
        for (size_t i = 0; i < draw; i++) {
            keys.push_back(rng());
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        if (keys.size() >= k) {
            keys.resize(k);
            return keys;
        }
        // Extremely unlikely retry
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: bench_lemonhash <n> [query_keys] [repeats] [seed]\n";
        return 1;
    }

    size_t n = std::stoull(argv[1]);
    size_t query_keys = (argc > 2) ? std::stoull(argv[2]) : 100000000;
    int repeats = (argc > 3) ? std::stoi(argv[3]) : 5;
    uint64_t seed = (argc > 4) ? std::stoull(argv[4]) : 42;

    // Generate sorted keys
    std::vector<uint64_t> keys = gen_sorted_keys(n, seed);

    // Construction
    auto t0 = std::chrono::steady_clock::now();
    lemonhash::LeMonHash<> mmphf(keys);
    auto t1 = std::chrono::steady_clock::now();

    double constr_secs = std::chrono::duration<double>(t1 - t0).count();
    double constr_ns = constr_secs * 1e9 / n;
    size_t size_bits = mmphf.spaceBits(false);
    size_t size_bytes = (size_bits + 7) / 8;

    std::cerr << "Construction completed in " << constr_secs
              << " seconds (" << n << " keys, " << constr_ns << " ns/key)\n";
    std::cerr << "Size: " << size_bytes << " Bytes = "
              << (double)size_bits / n << " bits/key\n";

    // Correctness check
    for (size_t i = 0; i < std::min<size_t>(100000, n); i++) {
        size_t result = mmphf(keys[i]);
        if (result != i) {
            std::cerr << "CORRECTNESS FAILED at index " << i
                      << ": expected " << i << " got " << result << "\n";
            return 1;
        }
    }

    // Pre-generate random query indices (same approach as bench_lcp_mmphf.rs)
    std::mt19937_64 qrng(1);
    std::vector<size_t> query_indices(query_keys);
    for (size_t i = 0; i < query_keys; i++) {
        query_indices[i] = qrng() % n;
    }

    // Query benchmark: random access into keys
    std::vector<double> timings;
    timings.reserve(repeats);

    for (int r = 0; r < repeats; r++) {
        uint64_t acc = 0;
        auto qstart = std::chrono::steady_clock::now();
        for (size_t i = 0; i < query_keys; i++) {
            acc ^= mmphf(keys[query_indices[i]]);
        }
        auto qend = std::chrono::steady_clock::now();
        DO_NOT_OPTIMIZE(acc);
        double ns_per_key = std::chrono::duration<double>(qend - qstart).count() * 1e9 / query_keys;
        timings.push_back(ns_per_key);
        std::cerr << ns_per_key << " ns/key\n";
    }

    std::sort(timings.begin(), timings.end());
    double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    std::cerr << "Min: " << timings.front()
              << " Median: " << timings[timings.size() / 2]
              << " Max: " << timings.back()
              << " Average: " << avg << "\n";

    return 0;
}
