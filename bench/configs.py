"""
Benchmark configurations for the ε-cost sharding paper.

This module defines the benchmark matrix: which methods to test, with what
parameters, and how they map to paper figures.

Each entry in METHODS specifies:
  - tool: "sux" (VFilter/VFunc), "burr" (BuRR ribbon), "csf"
    (BuRR-VL compressed static functions), "comp-vfunc" (sux-rs
    CompVFunc), "phastr" (PHastR + array), "mmphf" (monotone MMPHF, integer
    keys), "mmphf-str" (monotone MMPHF, string keys), "lemonhash"
    (LeMonHash, integer keys), "lemonhash-str" (LeMonHash, string
    keys), "java" (Sux4J GOV3)

  - How to build the structure and how to query it - Default size
    ranges and special flags

Bit widths for filters come in two families:
  - "even" (8, 16, 32): power-of-2 widths, byte-aligned storage
  - "odd" (9, 17, 33): non-power-of-2

To add a new method add an entry to METHODS following the existing pattern.
"""

# --- Bit-width families ---

EVEN_BITS = [8, 16, 32]
ODD_BITS = [9, 17, 33]
ALL_BITS = sorted(EVEN_BITS + ODD_BITS)

# --- Global defaults ---

# Starting size exponent: 2^20 ≈ 1M keys.
MIN_LOG2 = 20

# Default number of query operations per benchmark run.  Monotone MMPHF
# and LeMonHash methods override this via per-method "query_keys" configs.
QUERY_KEYS = 100_000_000

import os

# Default thread count for parallel construction.
DEFAULT_THREADS = max(1, os.cpu_count() // 2)

# Marker for the build_threads/burr_threads of methods whose construction is
# parallel: it resolves to --threads if given, and to DEFAULT_THREADS
# otherwise. Methods with a fixed count (e.g., the -seq variants, which measure
# single-threaded construction) are not affected by --threads.
PARALLEL = "parallel"

# Default number of parallel query instances for par-bench.
DEFAULT_PAR_JOBS = DEFAULT_THREADS

SUX_FEATURES = "cli,mwhc,rayon,epserde,deko"

# Cargo profile used for benchmark builds.
SUX_PROFILE = "bench"

# Tool types whose results are bits-agnostic (bits/key is a result
# metric, not a build parameter).  Used as the default when a method
# does not set "bits_agnostic" explicitly.
BITS_AGNOSTIC_TOOLS = frozenset({
    "csf", "comp-vfunc", "phastr", "mmphf", "lemonhash",
})

# --- Method definitions ---

METHODS = {
    # -----------------------------------------------------------------
    # sux-rs VFilter methods
    # -----------------------------------------------------------------
    #
    # All sux-rs methods:
    #   build: vfilter -n <n> --shard-edge <E> --bits <b> [--unaligned] <outfile>
    #   query: bench_vfilter <query_keys> <file> --shard-edge <E> --bits <b> [--unaligned]
    #
    # --unaligned is added only for odd (non-power-of-2) bit widths.
    # For even bit widths (8, 16, 32) the backend is a Box<[u8/u16/u32]>
    # which is inherently aligned, so --unaligned does not apply.
    # --shard-edge selects the hypergraph construction variant.
    #
    # The shard_edge values use clap's kebab-case convention.
    "Fuse3NoShards64": {
        "tool": "sux",
        "label": "Fuse(64)",
        # Fuse 3-hypergraphs, no ε-cost sharding, 64-bit signatures.
        # Fastest queries but limited scalability.
        "shard_edge": "fuse3-no-shards64",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": None,  # no sharding ⇒ single-threaded build
        "max_log2": 31,  # 64-bit sig space limits max size
        "build_extra": ["--seed", "1"],
    },
    "Fuse3NoShards128": {
        "tool": "sux",
        "label": "Fuse(128)",
        # Fuse 3-hypergraphs, no sharding, 128-bit signatures.
        "shard_edge": "fuse3-no-shards128",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": None,
        "max_log2": 32,
        # For n ≥ 2^32 the high-memory peeling algorithm exceeds RAM;
        # --low-mem switches to a slower but leaner variant.
        "low_mem_threshold": 32,
    },
    "FuseLge3Shards": {
        "tool": "sux",
        "label": "VFilter(Fuse+LGE)",
        # Fuse + lazy Gaussian elimination + ε-cost sharding.
        # The paper's main contribution.
        "shard_edge": "fuse-lge3-shards",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": PARALLEL,
        "max_log2": 33,
    },
    "FuseLge3Shards-seq": {
        "tool": "sux",
        "label": "VFilter(Fuse+LGE,seq)",
        # Same structure, single-threaded build.
        # Used only for construction-time comparison (no query benchmark).
        "shard_edge": "fuse-lge3-shards",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": 1,
        "no_queries": True,
        "max_log2": 33,
    },
    "Mwhc3": {
        "tool": "sux",
        "label": "VFilter(MWHC)",
        # Classic MWHC 3-hypergraphs with ε-cost sharding.
        "shard_edge": "mwhc3",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": PARALLEL,
        "max_log2": 33,
        # MWHC uses a slightly larger epsilon to shard finely
        "build_extra": ["--eps", "0.01"],
    },
    # -----------------------------------------------------------------
    # HEM-sharded VFilter (the sharding ablation)
    # -----------------------------------------------------------------
    #
    # Identical to FuseLge3Shards in every respect but the layout: same
    # signatures, same shard selection, same fuse-graph edges, same expansion
    # factor and segment-size law, same value backends, and by default the
    # same number of shards. What differs is that shards keep their natural,
    # unequal sizes, so their offsets have to be stored and loaded, and each
    # shard carries a local seed. The gap against VFilter is therefore the
    # cost of the offset/seed indirection alone. See hem-vfilter/src/lib.rs.
    #
    # Build:  hem_vfilter -n <n> --bits <b> <outfile>
    # Query:  bench_hem_vfilter <query_keys> <outfile>
    # The builder keeps all signatures in core (VFilter streams them through a
    # disk-backed store), so it needs about 32 bytes per key while sharding.
    # That fits 2^32 keys on a 384 GiB machine, and no more.
    #
    # This method appears in the query figures and in the filter table, but not
    # in the construction figures: it has no lazy Gaussian elimination and a
    # plain peeling loop, so its build time would differ from VFilter's for
    # reasons that have nothing to do with sharding.
    "HemVFilter": {
        "tool": "hem",
        "label": "HEM(Fuse)",
        "max_log2": 32,
    },
    # -----------------------------------------------------------------
    # BuRR (Bumped Ribbon Retrieval) methods
    # -----------------------------------------------------------------
    #
    # BuRR is a C++ project. The bit width is compiled into the binary
    # (e.g. ribbon8, ribbon16).  All runs use:
    #   -L 64   ribbon width 64
    #   -2      two-bit threshold compression
    #   -I      interleaved solution storage
    #
    # We use the 'bench' Makefile target (-O3 -DNDEBUG), not 'ribbon'
    # (-O2, assertions enabled).
    #
    # Build:  bench<bits> -t <threads> -L 64 -2 -I -n <n>
    # Query:  bench<bits> -t 1         -L 64 -2 -I -n <n> -r <file>
    "BuRR": {
        "tool": "burr",
        "label": "BuRR",
        "burr_binary": "bench",
        "burr_flags": ["-L", "64", "-2", "-I"],
        "burr_threads": PARALLEL,
        "max_log2": 33,
    },
    "BuRR-seq": {
        "tool": "burr",
        "label": "BuRR (seq)",
        "burr_binary": "bench",
        "burr_flags": ["-L", "64", "-2", "-I"],
        "burr_threads": 1,
        "no_queries": True,
        "max_log2": 33,
    },
    # -----------------------------------------------------------------
    # CSF (Compressed Static Functions via BuRR-VL)
    # -----------------------------------------------------------------
    #
    # bench_csf builds two BuRR-VL ribbons (filter + correction) with a
    # prefix-free code.  Bits-agnostic (bits/key is a result metric).
    #
    # Build:  bench_csf -n <n> --geometric --coder huffman -Q -w <file>
    # Query:  bench_csf -n <n> --coder huffman -r <file>
    "CSF-Geo": {
        "tool": "csf",
        "label": "CSF(geo)",
        "distribution": ["--geometric"],
        "coder": "huffman",
        "max_log2": 30,
        "query_keys": 20_000_000,
    },
    "CSF-Zipf": {
        "tool": "csf",
        "label": "CSF(zipf)",
        "distribution": ["--zipf", "2000"],
        "coder": "huffman",
        "max_log2": 30,
        "query_keys": 20_000_000,
    },
    "CSF-Uni256": {
        "tool": "csf",
        "label": "CSF(uni256)",
        "distribution": ["--uniform", "256"],
        "coder": "huffman",
        "max_log2": 30,
        "query_keys": 20_000_000,
    },
    # -----------------------------------------------------------------
    # sux-rs CompVFunc (compressed static functions)
    # -----------------------------------------------------------------
    #
    # Build with comp_vfunc, query with
    # bench_comp_vfunc.  Bits-agnostic (bits/key is a result metric).
    # Uses bits=0 sentinel.
    #
    # Build:  comp_vfunc -n <n> --geometric --shard-edge fuse3-shards <outfile>
    # Query:  bench_comp_vfunc <n> <outfile> --shard-edge fuse3-shards
    "CompVFunc-Geo": {
        "tool": "comp-vfunc",
        "label": "CompVFunc(geo)",
        "distribution": ["--geometric"],
        "shard_edge": "fuse3-shards",
        "build_threads": 1,
        "max_log2": 30,
    },
    "CompVFunc-Zipf": {
        "tool": "comp-vfunc",
        "label": "CompVFunc(zipf)",
        "distribution": ["--zipf", "2000"],
        "shard_edge": "fuse3-shards",
        "build_threads": 1,
        "max_log2": 30,
    },
    "CompVFunc-Uni256": {
        "tool": "comp-vfunc",
        "label": "CompVFunc(uni256)",
        "distribution": ["--uniform", "256"],
        "shard_edge": "fuse3-shards",
        "build_threads": 1,
        "max_log2": 30,
    },
    # -----------------------------------------------------------------
    # sux-rs VFunc (list index functions / minimal perfect hash)
    # -----------------------------------------------------------------
    #
    # Same two-step pattern as VFilter:
    #   Build: vfunc -n <n> --shard-edge <E> [--threads T] <outfile>
    #   Query: bench_vfunc <query_keys> <outfile> --shard-edge <E>
    #
    # VFunc maps keys to their rank [0..n), so bits/key is a result
    # metric (bits-agnostic).
    "VFunc": {
        "tool": "sux",
        "label": "VFunc(Fuse+LGE)",
        "shard_edge": "fuse-lge3-shards",
        "build_bin": "vfunc",
        "bench_bin": "bench_vfunc",
        "bits_agnostic": True,
        "build_threads": PARALLEL,
        "max_log2": 32,
        # Compared against PHastR + array, whose build+query run in one process
        # and therefore cannot be pinned.  Run VFunc queries unpinned
        # too so the comparison is apples-to-apples.
        "pin_query": False,
    },
    # -----------------------------------------------------------------
    # PHastR (minimal perfect hash + array)
    # -----------------------------------------------------------------
    #
    # bench_phastr_array (phastr-array crate) builds a PHastR + array
    # function and queries it in-process.
    # Bits-agnostic (bits/key is a result metric).
    # No separate build/query phases — construction and queries run in
    # a single invocation.
    #
    # Run:  bench_phastr_array <n> <query_keys> <threads>
    "PHastR": {
        "tool": "phastr",
        "label": "PHastR",
        "build_threads": PARALLEL,
        "max_log2": 32,
    },
    # -----------------------------------------------------------------
    # Monotone MMPHF methods (integer keys)
    # -----------------------------------------------------------------
    #
    # All use sorted random 64-bit keys (same generation algorithm).
    # Bits-agnostic (bits/key is a result metric).
    #
    # LcpMmphf/Lcp2Mmphf: use bench_lcp_mmphf from sux-rs.
    #   Run:  bench_lcp_mmphf <n> <query_keys> <threads>
    #
    # LeMonHash: uses bench_lemonhash (C++, cmake-built).
    #   Run:  bench_lemonhash <n> <query_keys> <repeats>
    "LcpMmphfInt": {
        "tool": "mmphf",
        "label": "LcpMmphf(int)",
        "variant": "lcp",
        "build_threads": 1,
        "max_log2": 32,
    },
    "Lcp2MmphfInt": {
        "tool": "mmphf",
        "label": "Lcp2Mmphf(int)",
        "variant": "lcp2",
        "build_threads": 1,
        "max_log2": 32,
    },
    "LeMonHashInt": {
        "tool": "lemonhash",
        "label": "LeMonHash(int)",
        "max_log2": 32,
    },
    # -----------------------------------------------------------------
    # Monotone MMPHF methods (string keys from data/ directory)
    # -----------------------------------------------------------------
    #
    # These iterate over sorted string datasets in the data/ directory
    # instead of using generated integer keys.  Skipped if data/ is
    # absent or empty.
    #
    # LcpMmphfStr/Lcp2MmphfStr: build with lcp_mmphf -f, query with
    #   bench_lcp_mmphf -f.
    #
    # LeMonHashStr: uses bench_lemonhash_str (C++, cmake-built).
    #   Run:  bench_lemonhash_str <file> <query_keys> <repeats>
    "LcpMmphfStr": {
        "tool": "mmphf-str",
        "label": "LcpMmphf(str)",
        "variant": "lcp",
        "build_threads": 1,
    },
    "Lcp2MmphfStr": {
        "tool": "mmphf-str",
        "label": "Lcp2Mmphf(str)",
        "variant": "lcp2",
        "build_threads": 1,
    },
    "LeMonHashStr": {
        "tool": "lemonhash-str",
        "label": "LeMonHash(str)",
        "query_keys": 20_000_000,
    },
    # -----------------------------------------------------------------
    # Sux4J GOV3 methods (Java build + C query)
    # -----------------------------------------------------------------
    #
    # Build: Java BenchBuilder creates GOV3Function/GV3CompressedFunction
    #        with LE-encoded uint64 keys, dumps C-binary.
    # Query: C test harness (sf3/csf3) loads dump, benchmarks queries.
    # Also the HEM baseline for the paper: GOV3 shards with the Heuristic
    # External Memory scheme of Botelho, Pagh and Ziviani, so a query must
    # load an offset and a local seed before it can generate its edge. That
    # is the indirection epsilon-cost sharding removes. Queries run through
    # the C code distributed with Sux4J, so the comparison with the Rust
    # implementations is not affected by the language.
    "GOV3Function": {
        "tool": "java",
        "label": "HEM/GOV3 (Sux4J)",
        "java_type": "GOV3Function",
        "sizes": [1 << 20, 1 << 30],
    },
    "GV3CompressedFunction-Geo": {
        "tool": "java",
        "label": "GOV3Comp(geo)",
        "java_type": "GV3CompressedFunction",
        "distribution": ["--geometric"],
        "bits_agnostic": True,
        "sizes": [1 << 20, 1 << 28],
    },
    "GV3CompressedFunction-Zipf": {
        "tool": "java",
        "label": "GOV3Comp(zipf)",
        "java_type": "GV3CompressedFunction",
        "distribution": ["--zipf", "2000"],
        "bits_agnostic": True,
        "sizes": [1 << 20, 1 << 28],
    },
    "GV3CompressedFunction-Uni256": {
        "tool": "java",
        "label": "GOV3Comp(uni256)",
        "java_type": "GV3CompressedFunction",
        "distribution": ["--uniform", "256"],
        "bits_agnostic": True,
        "sizes": [1 << 20, 1 << 28],
    },
}


# -----------------------------------------------------------------
# Figure presets
# -----------------------------------------------------------------
#
# Each entry maps a figure name to the methods and data column it
# should plot.  These match the figures generated by figs/genfigs.sh
# in the paper.
#
# The "bits" axis is handled at plot time: one PDF per bit width.
# For "even" figures use bits in EVEN_BITS; for "odd" use ODD_BITS.

FIGURES = {
    # --- Filters: per bit width ---
    # One PDF per bit width: filter-build-8.pdf, filter-query-32.pdf, etc.
    "filter-build": {
        "title": "Filter Construction ({bits} bits)",
        "methods": [
            "Fuse3NoShards64",
            "Fuse3NoShards128",
            "FuseLge3Shards",
            "Mwhc3",
            "BuRR",
            "FuseLge3Shards-seq",
            "BuRR-seq",
        ],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "filter-query": {
        "title": "Filter Queries ({bits} bits)",
        "methods": [
            "Fuse3NoShards64",
            "Fuse3NoShards128",
            "FuseLge3Shards",
            "HemVFilter",
            "Mwhc3",
            "BuRR",
        ],
        "column": "query_min",
        "ylabel": "ns/key",
    },
    "filter-parquery": {
        "title": "Filter Parallel Queries ({bits} bits)",
        "methods": [
            "Fuse3NoShards64",
            "Fuse3NoShards128",
            "FuseLge3Shards",
            "HemVFilter",
            "Mwhc3",
            "BuRR",
        ],
        "column": "parbench_med",
        "ylabel": "ns/key",
    },
    # --- Compressed static functions: all distributions on one plot ---
    # Bits-agnostic (auto-detected): comp-build.pdf, comp-query.pdf.
    "comp-build": {
        "title": "Compressed Function Construction",
        "methods": [
            "CSF-Geo", "CSF-Zipf", "CSF-Uni256",
            "CompVFunc-Geo", "CompVFunc-Zipf", "CompVFunc-Uni256",
        ],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "comp-query": {
        "title": "Compressed Function Queries",
        "methods": [
            "CSF-Geo", "CSF-Zipf", "CSF-Uni256",
            "CompVFunc-Geo", "CompVFunc-Zipf", "CompVFunc-Uni256",
        ],
        "column": "query_min",
        "ylabel": "ns/key",
    },
    # --- PHastR comparison: no bit width suffix ---
    # Bits-agnostic: phast-build.pdf, phast-query.pdf.
    "phast-build": {
        "title": "PHastR vs VFunc Construction",
        "methods": ["PHastR", "VFunc"],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "phast-query": {
        "title": "PHastR vs VFunc Queries",
        "methods": ["PHastR", "VFunc"],
        "column": "query_min",
        "ylabel": "ns/key",
    },
    # --- Monotone MMPHF comparison (integer keys) ---
    # Bits-agnostic: mmphf-build.pdf, mmphf-query.pdf.
    "mmphf-build": {
        "title": "Monotone MMPHF Construction",
        "methods": ["LcpMmphfInt", "Lcp2MmphfInt", "LeMonHashInt"],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "mmphf-query": {
        "title": "Monotone MMPHF Queries",
        "methods": ["LcpMmphfInt", "Lcp2MmphfInt", "LeMonHashInt"],
        "column": "query_min",
        "ylabel": "ns/key",
    },
}
