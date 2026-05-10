// Benchmark for LeMonHashVL (string variant of LeMonHash).
//
// Reads a sorted, newline-separated string file, builds a
// LeMonHashVL, and benchmarks random-access queries.
// Query strings are packed into a contiguous buffer to reduce
// memory-chasing overhead.
//
// Output format matches the rest of the framework:
//   "Construction completed in X.XXX seconds (N keys, Y.YYY ns/key)"
//   "Size: Z Bytes = B.BBB bits/key"
//   "Min: ... Median: ... Max: ... Average: ..."

#include <LeMonHashVL.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static std::vector<std::string> load_sorted_strings(const char *path) {
  std::vector<std::string> keys;
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty())
      keys.push_back(std::move(line));
  }
  return keys;
}

// Reservoir-samples up to n strings from a newline-separated file.
// The result is shuffled so strings don't arrive in file order.
static std::vector<std::string> reservoir_sample(const char *path, size_t n,
                                                  uint64_t seed) {
  std::vector<std::string> reservoir;
  reservoir.reserve(n);
  std::mt19937_64 rng(seed);
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  std::string line;
  size_t i = 0;
  while (std::getline(ifs, line)) {
    if (line.empty())
      continue;
    if (i < n) {
      reservoir.push_back(std::move(line));
    } else {
      size_t j = rng() % (i + 1);
      if (j < n)
        reservoir[j] = std::move(line);
    }
    ++i;
  }
  std::shuffle(reservoir.begin(), reservoir.end(), rng);
  return reservoir;
}

// Packs n queries into a contiguous char buffer by cycling through
// the queries vector.  Guarantees exactly n entries in the output.
static void pack_strings(const std::vector<std::string> &queries, size_t n,
                          std::vector<char> &packed,
                          std::vector<size_t> &offsets) {
  packed.clear();
  offsets.clear();
  offsets.reserve(n + 1);
  offsets.push_back(0);
  for (size_t i = 0; i < n; i++) {
    auto &s = queries[i % queries.size()];
    packed.insert(packed.end(), s.begin(), s.end());
    offsets.push_back(packed.size());
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: bench_lemonhash_str <file> [query_keys] [repeats]\n";
    return 1;
  }

  const char *file = argv[1];
  size_t query_keys = (argc > 2) ? std::stoull(argv[2]) : 10000000;
  int repeats = (argc > 3) ? std::stoi(argv[3]) : 5;

  std::vector<std::string> keys = load_sorted_strings(file);
  size_t n = keys.size();
  if (n < 2) {
    std::cerr << "File does not contain enough strings\n";
    return 1;
  }
  std::cerr << "Loaded " << n << " keys\n";

  // Construction
  auto t0 = std::chrono::steady_clock::now();
  lemonhash::LeMonHashVL<> mmphf(keys);
  auto t1 = std::chrono::steady_clock::now();

  double constr_secs = std::chrono::duration<double>(t1 - t0).count();
  double constr_ns = constr_secs * 1e9 / n;
  size_t size_bits = mmphf.spaceBits();
  size_t size_bytes = (size_bits + 7) / 8;

  std::cerr << "Construction completed in " << constr_secs << " seconds (" << n
            << " keys, " << constr_ns << " ns/key)\n";
  std::cerr << "Size: " << size_bytes << " Bytes = " << (double)size_bits / n
            << " bits/key\n";

  // Correctness check
  for (size_t i = 0; i < std::min<size_t>(100000, n); i++) {
    size_t result = mmphf(keys[i]);
    if (result != i) {
      std::cerr << "CORRECTNESS FAILED at index " << i << ": expected " << i
                << " got " << result << "\n";
      return 1;
    }
  }

  // Free construction keys before loading queries
  { std::vector<std::string>().swap(keys); }

  // Reservoir-sample query strings and pack into a contiguous buffer
  std::vector<char> packed;
  std::vector<size_t> offsets;
  {
    auto queries = reservoir_sample(file, query_keys, 42);
    pack_strings(queries, query_keys, packed, offsets);
  }

  // Query benchmark
  std::vector<double> timings;
  timings.reserve(repeats);

  // LeMonHashVL::operator() takes const std::string&, so we need a
  // std::string for each query.  We reuse a single instance: assign()
  // avoids heap allocation when the new content fits in the existing
  // capacity, reducing the per-query cost to a memcpy.
  std::string tmp;
  tmp.reserve(1024);

  for (int r = 0; r < repeats; r++) {
    uint64_t acc = 0;
    auto qstart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < query_keys; i++) {
      tmp.assign(&packed[offsets[i]], offsets[i + 1] - offsets[i]);
      acc ^= mmphf(tmp);
    }
    const volatile uint64_t _sink = acc;
    auto qend = std::chrono::steady_clock::now();

    double ns_per_key =
        std::chrono::duration<double>(qend - qstart).count() * 1e9 / query_keys;
    timings.push_back(ns_per_key);
    std::cerr << ns_per_key << " ns/key\n";
  }

  std::sort(timings.begin(), timings.end());
  double avg =
      std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
  std::cerr << "Min: " << timings.front()
            << " Median: " << timings[timings.size() / 2]
            << " Max: " << timings.back() << " Average: " << avg << "\n";

  return 0;
}
