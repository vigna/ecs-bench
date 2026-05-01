"""
Benchmark configurations for the ε-cost sharding paper.

This module defines the benchmark matrix: which methods to test, with what
parameters, and how they map to paper figures.

Each entry in METHODS specifies:
  - tool: "sux", "burr", "burr-vl", "csf", or "comp-vfunc"
  - How to build the structure and how to query it
  - Default size ranges and special flags

Bit widths come in two families:
  - "even" (8, 16, 32): power-of-2 widths, byte-aligned storage
  - "odd" (9, 17, 33): non-power-of-2, needs unaligned access

To add a new method (e.g., MMPHF, compressed functions), add an entry
to METHODS following the existing pattern.
"""

# --- Bit-width families ---

EVEN_BITS = [8, 16, 32]
ODD_BITS = [9, 17, 33]
ALL_BITS = sorted(EVEN_BITS + ODD_BITS)

# --- Global defaults ---

# Starting size exponent: 2^20 ≈ 1M keys.
MIN_LOG2 = 20

# Number of query operations per benchmark run. All methods use the same
# count so that per-query timing is directly comparable.
QUERY_KEYS = 100_000_000

# Default parallel instances for par-bench.  Defaults to half the
# available CPUs — using all cores causes problems on architectures
# with heterogeneous cores (P/E cores) or heavy hyperthreading.
import os
DEFAULT_THREADS = max(1, os.cpu_count() // 2)

# Default number of parallel query instances for par-bench.
DEFAULT_PAR_JOBS = DEFAULT_THREADS

# BuRR-VL flags matching the LSF paper: L=128, depth 2, two-bit
# threshold compression, interleaved solution storage.
BURR_VL_FLAGS = ["-L", "128", "-d", "2", "-2", "-I"]

# Cargo features required for sux-rs binaries.
#   cli      – enables the CLI binaries (vfilter, vfunc, …)
#   mwhc     – enables the Mwhc3 shard/edge type
#   rayon    – parallel construction via Rayon
#   epserde  – serialization support (ε-serde)
#   deko     – decompression support for compressed input
SUX_FEATURES = "cli,mwhc,rayon,epserde,deko"

# Cargo profile used for benchmark builds.
SUX_PROFILE = "bench"

# --- Method definitions ---

METHODS = {
    # -----------------------------------------------------------------
    # sux-rs VFilter methods
    # -----------------------------------------------------------------
    #
    # All sux-rs methods:
    #   build: vfilter -n <n> --shard-edge <E> --bits <b> [--unaligned] <outfile>
    #   query: bench_vfilter <query_keys> <file> --edge <E> --bits <b> [--unaligned]
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
        "build_threads": None,     # no sharding ⇒ single-threaded build
        "max_log2": 31,            # 64-bit sig space limits max size
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
        "label": "VFilter(Fuse+LGE,par)",
        # Fuse + lazy Gaussian elimination + ε-cost sharding.
        # The paper's main contribution.
        "shard_edge": "fuse-lge3-shards",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": DEFAULT_THREADS,
        "max_log2": 33,
    },

    "FuseLge3Shards-seq": {
        "tool": "sux",
        "label": "VFilter(Fuse+LGE,seq)",
        # Same structure, single-threaded build.
        # Used only for construction-time comparison (no query benchmark).
        "shard_edge": "fuse-lge3-shards",
        "build_bin": "vfilter",
        "bench_bin": None,         # no queries
        "build_threads": 1,
        "max_log2": 33,
    },

    "Mwhc3": {
        "tool": "sux",
        "label": "VFilter(MWHC,par)",
        # Classic MWHC 3-hypergraphs with ε-cost sharding.
        "shard_edge": "mwhc3",
        "build_bin": "vfilter",
        "bench_bin": "bench_vfilter",
        "build_threads": DEFAULT_THREADS,
        "max_log2": 33,
        # MWHC needs a slightly larger epsilon for solvability.
        "build_extra": ["--eps", "0.01"],
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

    "BuRR-par": {
        "tool": "burr",
        "label": "BuRR (par)",
        "burr_binary": "bench",
        "burr_flags": ["-L", "64", "-2", "-I"],
        "burr_threads": DEFAULT_THREADS,
        "max_log2": 32,
    },

    "BuRR-seq": {
        "tool": "burr",
        "label": "BuRR (seq)",
        "burr_binary": "bench",
        "burr_flags": ["-L", "64", "-2", "-I"],
        "burr_threads": 1,
        "max_log2": 32,
    },

    # -----------------------------------------------------------------
    # BuRR-VL (Variable-Length Ribbon retrieval)
    # -----------------------------------------------------------------
    #
    # Plain VLR ribbon matching the LSF paper config: L=128, depth 2,
    # two-bit threshold, interleaved, uint64_t keys.
    # Takes --geometric / --zipf K / --uniform K for value distribution.
    #
    # Build:  ribbon_vlr -L 128 -d 2 -2 -I -n <n> --geometric -Q -w <file>
    # Query:  ribbon_vlr -L 128 -d 2 -2 -I -n <n> -r <file>

    "BuRR-VL-Geo": {
        "tool": "burr-vl",
        "label": "BuRR-VL(geo)",
        "burr_flags": BURR_VL_FLAGS,
        "distribution": ["--geometric"],
        "max_log2": 32,
    },

    "BuRR-VL-Zipf": {
        "tool": "burr-vl",
        "label": "BuRR-VL(zipf)",
        "burr_flags": BURR_VL_FLAGS,
        "distribution": ["--zipf", "2000"],
        "max_log2": 32,
    },

    "BuRR-VL-Uni256": {
        "tool": "burr-vl",
        "label": "BuRR-VL(uni256)",
        "burr_flags": BURR_VL_FLAGS,
        "distribution": ["--uniform", "256"],
        "max_log2": 32,
    },

    # -----------------------------------------------------------------
    # CSF (Compressed Static Functions via BuRR-VL)
    # -----------------------------------------------------------------
    #
    # bench_csf builds two BuRR-VL ribbons (filter + correction) with a
    # prefix-free code.  Bits-agnostic (bits/key is a result metric).
    #
    # Build:  bench_csf -n <n> --geometric --coder fano -Q -w <file>
    # Query:  bench_csf -n <n> --coder fano -r <file>

    "CSF-Fano-Geo": {
        "tool": "csf",
        "label": "CSF(Fano,geo)",
        "distribution": ["--geometric"],
        "coder": "fano",
        "max_log2": 32,
    },

    "CSF-Fano-Zipf": {
        "tool": "csf",
        "label": "CSF(Fano,zipf)",
        "distribution": ["--zipf", "2000"],
        "coder": "fano",
        "max_log2": 32,
    },

    "CSF-Fano-Uni256": {
        "tool": "csf",
        "label": "CSF(Fano,uni256)",
        "distribution": ["--uniform", "256"],
        "coder": "fano",
        "max_log2": 32,
    },

    # -----------------------------------------------------------------
    # sux-rs CompVFunc (compressed static functions)
    # -----------------------------------------------------------------
    #
    # Two-step like VFilter: build with comp_vfunc, query with
    # bench_comp_vfunc.  Bits-agnostic (bits/key is a result metric).
    # Uses bits=0 sentinel.
    #
    # Build:  comp_vfunc -n <n> --geometric --shard-edge fuse3-shards <outfile>
    # Query:  bench_comp_vfunc <n> <outfile> --edge fuse3-shards

    "CompVFunc-Geo": {
        "tool": "comp-vfunc",
        "label": "CompVFunc(geo)",
        "distribution": ["--geometric"],
        "shard_edge": "fuse3-shards",
        "build_threads": DEFAULT_THREADS,
        "max_log2": 33,
    },

    "CompVFunc-Zipf": {
        "tool": "comp-vfunc",
        "label": "CompVFunc(zipf)",
        "distribution": ["--zipf", "2000"],
        "shard_edge": "fuse3-shards",
        "build_threads": DEFAULT_THREADS,
        "max_log2": 33,
    },

    "CompVFunc-Uni256": {
        "tool": "comp-vfunc",
        "label": "CompVFunc(uni256)",
        "distribution": ["--uniform", "256"],
        "shard_edge": "fuse3-shards",
        "build_threads": DEFAULT_THREADS,
        "max_log2": 33,
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
    "construction": {
        "title": "Construction ({bits} bits)",
        "methods": [
            "Fuse3NoShards64", "Fuse3NoShards128",
            "FuseLge3Shards", "FuseLge3Shards-seq",
            "Mwhc3", "BuRR-par", "BuRR-seq",
        ],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "queries": {
        "title": "Queries ({bits} bits)",
        "methods": [
            "Fuse3NoShards64", "Fuse3NoShards128",
            "FuseLge3Shards", "Mwhc3", "BuRR-par",
        ],
        "column": "query_min",
        "ylabel": "ns/key",
    },
    "parqueries": {
        "title": "Parallel Queries ({bits} bits)",
        "methods": [
            "Fuse3NoShards64", "Fuse3NoShards128",
            "FuseLge3Shards", "Mwhc3", "BuRR-par",
        ],
        "column": "parbench_max",
        "ylabel": "ns/key",
    },
    "comp": {
        "title": "Construction",
        # Cross-bit-width comparison: FuseLge3Shards-seq vs Fuse3NoShards128.
        # All bit widths appear on the same plot (label includes bit width).
        "methods": ["FuseLge3Shards-seq", "Fuse3NoShards128"],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },

    # --- Compressed / variable-length figures ---
    # These are bits-agnostic (auto-detected by the plot code).

    "csf-construction-geo": {
        "title": "CSF Construction (geometric)",
        "methods": ["BuRR-VL-Geo", "CSF-Fano-Geo", "CompVFunc-Geo"],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "csf-construction-zipf": {
        "title": "CSF Construction (zipf 2k)",
        "methods": ["BuRR-VL-Zipf", "CSF-Fano-Zipf", "CompVFunc-Zipf"],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "csf-construction-uni": {
        "title": "CSF Construction (uniform 256)",
        "methods": ["BuRR-VL-Uni256", "CSF-Fano-Uni256", "CompVFunc-Uni256"],
        "column": "constr_ns",
        "ylabel": "ns/key",
    },
    "csf-queries-geo": {
        "title": "CSF Queries (geometric)",
        "methods": ["BuRR-VL-Geo", "CSF-Fano-Geo", "CompVFunc-Geo"],
        "column": "query_min",
        "ylabel": "ns/key",
    },
    "csf-queries-zipf": {
        "title": "CSF Queries (zipf 2k)",
        "methods": ["BuRR-VL-Zipf", "CSF-Fano-Zipf", "CompVFunc-Zipf"],
        "column": "query_min",
        "ylabel": "ns/key",
    },
    "csf-queries-uni": {
        "title": "CSF Queries (uniform 256)",
        "methods": ["BuRR-VL-Uni256", "CSF-Fano-Uni256", "CompVFunc-Uni256"],
        "column": "query_min",
        "ylabel": "ns/key",
    },
}
