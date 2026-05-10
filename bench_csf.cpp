// Benchmark for compressed static functions: BuRR-VL with Shannon-Fano
// or Huffman coding, comparable to sux-rs CompVFunc.
//
// Value distributions:
//   --geometric   trailing zeros of random uint64 (highly skewed)
//   --zipf K      Zipf(1) over [0..K)
//   --uniform K   uniform over [0..K)

#include "ribbon.hpp"
#include "rocksdb/stop_watch.h"

#include <tlx/cmdline_parser.hpp>
#include <tlx/logger.hpp>

#include "lsf/filtered_lsf_storage.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <vector>

// =====================================================================
//  Compressed static function: thin wrapper around FilteredLSFStorage
//  adding stored probabilities and serialization.
// =====================================================================

template <typename Coding> class CompressedStaticFunction {
  lsf::FilteredLSFStorage<Coding> storage;
  std::vector<float> probabilities;

public:
  void build(size_t n, const uint64_t *keys, const uint32_t *values,
             size_t classes_count, const std::vector<float> &freqs) {
    probabilities.assign(freqs.begin(), freqs.end());
    std::span<float> probs(probabilities);
    storage.build(n, classes_count, [&](size_t i) {
      return std::make_tuple(keys[i], static_cast<uint64_t>(values[i]), probs);
    });
  }

  uint64_t query(uint64_t key) {
    std::span<float> probs(probabilities);
    return storage.query(key, probs);
  }

  size_t size_in_bytes() const { return storage.size_in_bytes(); }

  void serialize(const std::string &filename) {
    std::ofstream os(filename, std::ios::binary);
    uint32_t nc = probabilities.size();
    os.write(reinterpret_cast<const char *>(&nc), sizeof(nc));
    os.write(reinterpret_cast<const char *>(probabilities.data()),
             nc * sizeof(float));
    storage.filterVLSF.Serialize(os);
    storage.correctionVLSF.Serialize(os);
  }

  void deserialize(const std::string &filename) {
    std::ifstream is(filename, std::ios::binary);
    uint32_t nc;
    is.read(reinterpret_cast<char *>(&nc), sizeof(nc));
    probabilities.resize(nc);
    is.read(reinterpret_cast<char *>(probabilities.data()), nc * sizeof(float));
    std::span<float> probs(probabilities);
    storage.coder = Coding(nc, probs);
    storage.filterVLSF.Deserialize(is);
    storage.correctionVLSF.Deserialize(is);
  }
};

// =====================================================================
//  Value generation
// =====================================================================

void generate_geometric(size_t n, uint32_t *values, size_t &classes_count) {
  std::mt19937_64 rng(42);
  classes_count = 0;
  for (size_t i = 0; i < n; ++i) {
    uint64_t x = rng();
    uint32_t v = (x == 0) ? 64 : __builtin_ctzll(x);
    values[i] = v;
    if (v + 1 > classes_count)
      classes_count = v + 1;
  }
}

void generate_zipf(size_t n, uint32_t *values, size_t K,
                   size_t &classes_count) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  double H = 0.0;
  for (size_t j = 1; j <= K; ++j)
    H += 1.0 / j;

  std::vector<double> cdf(K);
  double cumsum = 0.0;
  for (size_t j = 0; j < K; ++j) {
    cumsum += (1.0 / (j + 1)) / H;
    cdf[j] = cumsum;
  }

  classes_count = K;
  for (size_t i = 0; i < n; ++i) {
    double u = unif(rng);
    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
    values[i] = static_cast<uint32_t>(it - cdf.begin());
  }
}

void generate_uniform(size_t n, uint32_t *values, size_t K,
                      size_t &classes_count) {
  std::mt19937_64 rng(42);
  classes_count = K;
  for (size_t i = 0; i < n; ++i) {
    values[i] = rng() % K;
  }
}

std::vector<float> compute_frequencies(const uint32_t *values, size_t n,
                                       size_t classes_count) {
  std::vector<size_t> counts(classes_count, 0);
  for (size_t i = 0; i < n; ++i)
    counts[values[i]]++;
  std::vector<float> freqs(classes_count);
  for (size_t i = 0; i < classes_count; ++i)
    freqs[i] = static_cast<float>(counts[i]) / static_cast<float>(n);
  return freqs;
}

// =====================================================================
//  Benchmark driver
// =====================================================================

using FanoCoding = lsf::BitWiseFilterCoding<lsf::FilterFanoCoder>;
using HuffmanCoding = lsf::BitWiseFilterCoding<lsf::FilterHuffmanCoderCSF>;

bool no_queries = false;

template <typename Coding>
void run_bench(size_t n, const uint64_t *keys, const uint32_t *values,
               size_t classes_count, const std::vector<float> &freqs,
               const std::string &ofile, const std::string &ifile) {

  CompressedStaticFunction<Coding> csf;

  if (ifile.length() != 0) {
    rocksdb::StopWatchNano timer(true);
    csf.deserialize(ifile);
    LOG1 << "Deserialization took " << timer.ElapsedNanos(true) / 1e6 << "ms";
    LOG1 << "Size: " << csf.size_in_bytes()
         << " Bytes = " << (csf.size_in_bytes() * 8.0) / n << " bits/key";
  } else {
    rocksdb::StopWatchNano timer(true);
    csf.build(n, keys, values, classes_count, freqs);

    uint64_t constr_time = timer.ElapsedNanos(true);
    size_t bytes = csf.size_in_bytes();
    LOG1 << "Completed in " << constr_time / 1e6 << "ms, " << n << " keys, "
         << (double)constr_time / n << " ns/key";
    LOG1 << "Size: " << bytes << " Bytes = " << (bytes * 8.0) / n
         << " bits/key";
  }

  if (ofile.length() != 0) {
    rocksdb::StopWatchNano timer(true);
    csf.serialize(ofile);
    LOG1 << "Serialization took " << timer.ElapsedNanos(true) / 1e6 << "ms";
  }

  if (keys != nullptr && values != nullptr) {
    unsigned errors = 0;
    for (size_t i = 0; i < std::min<size_t>(100000, n); ++i) {
      uint64_t got = csf.query(keys[i]);
      if (got != values[i]) {
        if (errors < 10)
          LOG1 << "MISMATCH key=" << keys[i] << " expected=" << values[i]
               << " got=" << got;
        errors++;
      }
    }
    if (errors > 0) {
      LOG1 << "CORRECTNESS FAILED: " << errors << " / " << n << " mismatches";
      return;
    }
  }

  if (no_queries)
    return;

  const uint64_t N = 100000000;
  const int REPEATS = 5;

  std::vector<double> timings;
  for (int k = 0; k < REPEATS; k++) {
    rocksdb::StopWatchNano timer_query(true);
    uint64_t key = 0;
    uint64_t acc = 0;
    for (size_t v = 0; v < N; v++) {
      key += 0x9e3779b97f4a7c15;
      acc ^= csf.query(key);
    }
    const volatile uint64_t _sink = acc;
    double timing = (double)timer_query.ElapsedNanos(true) / N;
    LOG1 << timing << " ns/key";
    timings.push_back(timing);
  }

  sort(timings.begin(), timings.end());
  LOG1 << "Min: " << timings[0] << " Median: " << timings[timings.size() / 2]
       << " Max: " << timings[timings.size() - 1] << " Average: "
       << reduce(timings.begin(), timings.end(), 0.0) / timings.size();
}

int main(int argc, char **argv) {
  tlx::CmdlineParser cmd;
  size_t num_items = 1024 * 1024;
  bool geometric = false;
  size_t zipf_k = 0, uniform_k = 0;
  std::string coder_name = "fano";
  std::string ifile, ofile;

  cmd.add_size_t('n', "items", num_items, "number of items");
  cmd.add_bool('g', "geometric", geometric,
               "geometric distribution (trailing zeros)");
  cmd.add_size_t('z', "zipf", zipf_k, "Zipf(1) distribution over [0..K)");
  cmd.add_size_t('u', "uniform", uniform_k, "uniform distribution over [0..K)");
  cmd.add_string('c', "coder", coder_name,
                 "coder: fano or huffman (default: fano)");
  cmd.add_bool('Q', "noqueries", no_queries, "don't run any queries");
  cmd.add_string('r', "read", ifile, "file to read serialized data");
  cmd.add_string('w', "write", ofile, "file to write serialized data");

  if (!cmd.process(argc, argv)) {
    return 1;
  }

  // Exactly one distribution must be selected (unless reading from file)
  int dist_count =
      (geometric ? 1 : 0) + (zipf_k > 0 ? 1 : 0) + (uniform_k > 0 ? 1 : 0);
  if (ifile.length() == 0 && dist_count != 1) {
    std::cerr << "Exactly one of --geometric, --zipf K, --uniform K must be "
                 "specified.\n";
    return 1;
  }

  cmd.print_result();

  // When deserializing, we still need n for the query benchmark but
  // don't need to generate keys/values.
  if (ifile.length() != 0) {
    if (coder_name == "fano") {
      LOG1 << "Coder: Shannon-Fano";
      run_bench<FanoCoding>(num_items, nullptr, nullptr, 0, {}, ofile, ifile);
    } else if (coder_name == "huffman") {
      LOG1 << "Coder: Huffman (CSF)";
      run_bench<HuffmanCoding>(num_items, nullptr, nullptr, 0, {}, ofile,
                               ifile);
    } else {
      std::cerr << "Unknown coder: " << coder_name
                << " (use 'fano' or 'huffman')\n";
      return 1;
    }
    return 0;
  }

  // Generate keys [0..n)
  auto keys = std::make_unique<uint64_t[]>(num_items);
  std::iota(keys.get(), keys.get() + num_items, uint64_t(0));

  // Generate values
  auto values = std::make_unique<uint32_t[]>(num_items);
  size_t classes_count = 0;

  if (geometric) {
    generate_geometric(num_items, values.get(), classes_count);
    LOG1 << "Distribution: geometric (trailing zeros), " << classes_count
         << " classes";
  } else if (zipf_k > 0) {
    generate_zipf(num_items, values.get(), zipf_k, classes_count);
    LOG1 << "Distribution: Zipf(1) over [0.." << zipf_k << "), "
         << classes_count << " classes";
  } else {
    generate_uniform(num_items, values.get(), uniform_k, classes_count);
    LOG1 << "Distribution: uniform over [0.." << uniform_k << "), "
         << classes_count << " classes";
  }

  auto freqs = compute_frequencies(values.get(), num_items, classes_count);

  if (coder_name == "fano") {
    LOG1 << "Coder: Shannon-Fano";
    run_bench<FanoCoding>(num_items, keys.get(), values.get(), classes_count,
                          freqs, ofile, ifile);
  } else if (coder_name == "huffman") {
    LOG1 << "Coder: Huffman (CSF)";
    run_bench<HuffmanCoding>(num_items, keys.get(), values.get(), classes_count,
                             freqs, ofile, ifile);
  } else {
    std::cerr << "Unknown coder: " << coder_name
              << " (use 'fano' or 'huffman')\n";
    return 1;
  }
}
