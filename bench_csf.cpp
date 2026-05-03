// Benchmark for compressed static functions: BuRR-VL with Shannon-Fano
// or Huffman coding, comparable to sux-rs CompVFunc.
//
// Uses the FilteredLSFStorage architecture from the LearnedStaticFunction
// paper: two BuRR-VL ribbon structures (filter + correction) with a
// prefix-free code (Shannon-Fano or Huffman) driving the encoding.
//
// Value distributions:
//   --geometric   trailing zeros of random uint64 (highly skewed)
//   --zipf K      Zipf(1) over [0..K)
//   --uniform K   uniform over [0..K)
//
// BuRR-VL configuration matches the LSF paper:
//   ribbon width 128, result bits 1, two-bit threshold, interleaved,
//   depth 2, slotsPerItem 0.96, bucket size 128, uint64_t keys.

#include "ribbon.hpp"
#include "rocksdb/stop_watch.h"

#include <tlx/cmdline_parser.hpp>
#include <tlx/logger.hpp>

// Coding infrastructure from LearnedStaticFunction
#include "lsf/filter_coding.hpp"

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

using namespace ribbon;

// BuRR-VL configuration matching LSF paper exactly.
struct BuRRConfig
        : public RConfig<128, 1, ThreshMode::twobit, false, true, false, 0, uint64_t> {
    static constexpr bool kUseVLR = true;
    static constexpr Index kBucketSize = 128;
};

static constexpr size_t recDepth = 2;
static constexpr float slotsPerItem = 0.96;


// =====================================================================
//  Compressed static function: two BuRR-VL ribbons + prefix-free code
// =====================================================================

template<typename Coding>
class CompressedStaticFunction {
    ribbon_filter<recDepth, BuRRConfig> correctionVLSF;
    ribbon_filter<recDepth, BuRRConfig> filterVLSF;
    Coding coder;
    std::vector<float> probabilities;

public:

    CompressedStaticFunction() {}

    // Build from (key, value) pairs with a known global frequency table.
    void build(size_t n, const uint64_t *keys, const uint32_t *values,
               size_t classes_count, const std::vector<float> &freqs) {
        IMPORT_RIBBON_CONFIG(BuRRConfig);

        probabilities.assign(freqs.begin(), freqs.end());
        std::span<float> probs(probabilities);
        coder = Coding(classes_count, probs);

        // Phase 1: encode filter bits
        auto inputFilter = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
        size_t maxlenfilter = 0;
        for (size_t i = 0; i < n; ++i) {
            auto [code, filterLength, bitsSet] = coder.encode_once_filter(probs, values[i]);
            inputFilter[i].first = keys[i];
            if (filterLength > maxlenfilter)
                maxlenfilter = filterLength;
            inputFilter[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << filterLength);
        }

        filterVLSF = ribbon_filter<recDepth, BuRRConfig>(slotsPerItem, 42, maxlenfilter);
        filterVLSF.AddRange(inputFilter.get(), inputFilter.get() + n, true);
        filterVLSF.BackSubst();

        // Phase 2: encode correction bits using the filter output
        auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
        size_t maxlen = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t filterVal = filterVLSF.QueryRetrieval(keys[i]);
            auto [code, length] = coder.encode_once_corrected_code(probs, values[i], filterVal);
            input[i].first = keys[i];
            if (length > maxlen)
                maxlen = length;
            input[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << length);
        }

        correctionVLSF = ribbon_filter<recDepth, BuRRConfig>(slotsPerItem, 42, maxlen);
        correctionVLSF.AddRange(input.get(), input.get() + n);
        correctionVLSF.BackSubst();
    }

    uint64_t query(uint64_t key) {
        uint64_t corrected_code = correctionVLSF.QueryRetrieval(key);
        uint64_t filterCode = filterVLSF.QueryRetrieval(key);
        std::span<float> probs(probabilities);
        return coder.decode_once(probs, corrected_code, filterCode);
    }

    size_t size_in_bytes() const {
        return filterVLSF.Size() + correctionVLSF.Size();
    }

    void serialize(const std::string &filename) {
        std::ofstream os(filename, std::ios::binary);
        uint32_t nc = probabilities.size();
        os.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
        os.write(reinterpret_cast<const char*>(probabilities.data()), nc * sizeof(float));
        filterVLSF.Serialize(os);
        correctionVLSF.Serialize(os);
    }

    void deserialize(const std::string &filename) {
        std::ifstream is(filename, std::ios::binary);
        uint32_t nc;
        is.read(reinterpret_cast<char*>(&nc), sizeof(nc));
        probabilities.resize(nc);
        is.read(reinterpret_cast<char*>(probabilities.data()), nc * sizeof(float));
        std::span<float> probs(probabilities);
        coder = Coding(nc, probs);
        filterVLSF.Deserialize(is);
        correctionVLSF.Deserialize(is);
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
        uint32_t v = (x == 0) ? 63 : __builtin_ctzll(x);
        values[i] = v;
        if (v + 1 > classes_count) classes_count = v + 1;
    }
}

void generate_zipf(size_t n, uint32_t *values, size_t K, size_t &classes_count) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double H = 0.0;
    for (size_t j = 1; j <= K; ++j) H += 1.0 / j;

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

void generate_uniform(size_t n, uint32_t *values, size_t K, size_t &classes_count) {
    std::mt19937_64 rng(42);
    classes_count = K;
    for (size_t i = 0; i < n; ++i) {
        values[i] = rng() % K;
    }
}

std::vector<float> compute_frequencies(const uint32_t *values, size_t n, size_t classes_count) {
    std::vector<size_t> counts(classes_count, 0);
    for (size_t i = 0; i < n; ++i) counts[values[i]]++;
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

template<typename Coding>
void run_bench(size_t n, const uint64_t *keys, const uint32_t *values,
               size_t classes_count, const std::vector<float> &freqs,
               const std::string &ofile, const std::string &ifile) {

    CompressedStaticFunction<Coding> csf;

    if (ifile.length() != 0) {
        rocksdb::StopWatchNano timer(true);
        csf.deserialize(ifile);
        LOG1 << "Deserialization took " << timer.ElapsedNanos(true) / 1e6 << "ms";
        LOG1 << "Size: " << csf.size_in_bytes() << " Bytes = "
             << (csf.size_in_bytes() * 8.0) / n << " bits/key";
    } else {
        rocksdb::StopWatchNano timer(true);
        csf.build(n, keys, values, classes_count, freqs);

        uint64_t constr_time = timer.ElapsedNanos(true);
        size_t bytes = csf.size_in_bytes();
        LOG1 << "Completed in " << constr_time / 1e6 << "ms, "
             << n << " keys, " << (double)constr_time / n << " ns/key";
        LOG1 << "Size: " << bytes << " Bytes = " << (bytes * 8.0) / n << " bits/key";
    }

    if (ofile.length() != 0) {
        rocksdb::StopWatchNano timer(true);
        csf.serialize(ofile);
        LOG1 << "Serialization took " << timer.ElapsedNanos(true) / 1e6 << "ms";
    }

    // Correctness check: verify ALL keys return the correct value.
    if (keys != nullptr && values != nullptr) {
        unsigned errors = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t got = csf.query(keys[i]);
            if (got != values[i]) {
                if (errors < 10)
                    LOG1 << "MISMATCH key=" << keys[i]
                         << " expected=" << values[i] << " got=" << got;
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

    const uint64_t N = 10000000;
    const int REPEATS = 5;
    bool found = false;

    std::vector<double> timings;
    for (int k = 0; k < REPEATS; k++) {
        rocksdb::StopWatchNano timer_query(true);
        uint64_t key = 0;
        uint64_t acc = 0;
        for (size_t v = 0; v < N; v++) {
            key += 0x9e3779b97f4a7c15;
            acc ^= csf.query(key);
        }
        found ^= (acc != 0);
        double timing = (double)timer_query.ElapsedNanos(true) / N;
        LOG1 << timing << " ns/key";
        timings.push_back(timing);
    }

    sort(timings.begin(), timings.end());
    LOG1 << "Min: " << timings[0] << " Median: " << timings[timings.size() / 2]
         << " Max: " << timings[timings.size() - 1] << " Average: "
         << reduce(timings.begin(), timings.end(), 0.0) / timings.size()
         << " (" << found << ")";
}


int main(int argc, char **argv) {
    tlx::CmdlineParser cmd;
    size_t num_items = 1024 * 1024;
    bool geometric = false;
    size_t zipf_k = 0, uniform_k = 0;
    std::string coder_name = "fano";
    std::string ifile, ofile;

    cmd.add_size_t('n', "items", num_items, "number of items");
    cmd.add_bool('g', "geometric", geometric, "geometric distribution (trailing zeros)");
    cmd.add_size_t('z', "zipf", zipf_k, "Zipf(1) distribution over [0..K)");
    cmd.add_size_t('u', "uniform", uniform_k, "uniform distribution over [0..K)");
    cmd.add_string('c', "coder", coder_name, "coder: fano or huffman (default: fano)");
    cmd.add_bool('Q', "noqueries", no_queries, "don't run any queries");
    cmd.add_string('r', "read", ifile, "file to read serialized data");
    cmd.add_string('w', "write", ofile, "file to write serialized data");

    if (!cmd.process(argc, argv)) {
        return 1;
    }

    // Exactly one distribution must be selected (unless reading from file)
    int dist_count = (geometric ? 1 : 0) + (zipf_k > 0 ? 1 : 0) + (uniform_k > 0 ? 1 : 0);
    if (ifile.length() == 0 && dist_count != 1) {
        std::cerr << "Exactly one of --geometric, --zipf K, --uniform K must be specified.\n";
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
            run_bench<HuffmanCoding>(num_items, nullptr, nullptr, 0, {}, ofile, ifile);
        } else {
            std::cerr << "Unknown coder: " << coder_name << " (use 'fano' or 'huffman')\n";
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
        LOG1 << "Distribution: geometric (trailing zeros), " << classes_count << " classes";
    } else if (zipf_k > 0) {
        generate_zipf(num_items, values.get(), zipf_k, classes_count);
        LOG1 << "Distribution: Zipf(1) over [0.." << zipf_k << "), " << classes_count << " classes";
    } else {
        generate_uniform(num_items, values.get(), uniform_k, classes_count);
        LOG1 << "Distribution: uniform over [0.." << uniform_k << "), " << classes_count << " classes";
    }

    auto freqs = compute_frequencies(values.get(), num_items, classes_count);

    if (coder_name == "fano") {
        LOG1 << "Coder: Shannon-Fano";
        run_bench<FanoCoding>(num_items, keys.get(), values.get(), classes_count, freqs, ofile, ifile);
    } else if (coder_name == "huffman") {
        LOG1 << "Coder: Huffman (CSF)";
        run_bench<HuffmanCoding>(num_items, keys.get(), values.get(), classes_count, freqs, ofile, ifile);
    } else {
        std::cerr << "Unknown coder: " << coder_name << " (use 'fano' or 'huffman')\n";
        return 1;
    }
}
