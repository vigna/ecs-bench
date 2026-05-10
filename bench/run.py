#!/usr/bin/env python3
"""
Benchmark runner for the ε-cost sharding paper.

Orchestrates build, query, and reporting for sux-rs (VFilter/VFunc/CompVFunc),
BuRR (bumped ribbon), and CSF (compressed static functions via BuRR-VL)
benchmarks.  Results are stored as individual files under
results/<method>/, one per (n, bits) data point, enabling interrupt/resume
without losing partial progress.

Usage examples:

    # Smoke test: one method, 8 bits, small sizes
    python bench/run.py bench --method FuseLge3Shards --bits 8 --max-log2 22

    # Full run with CPU pinning
    python bench/run.py bench --method all --bits all \\
        --pin "chrt -f 99 taskset -c 2"

    # Report to stdout
    python bench/run.py report --method FuseLge3Shards --bits 8

    # Parallel query benchmark (96 instances via GNU parallel)
    python bench/run.py par-bench --method FuseLge3Shards --bits 8 \\
        --max-log2 30 --jobs 96

    # Generate paper figures
    python bench/run.py plot --figure construction --bits even

    # Show what has been completed
    python bench/run.py status

Environment variables:
    BENCH_PIN     Default CPU-pinning prefix, e.g. "chrt -f 99 taskset -c 2"
    SUX_DIR       Path to sux-rs repo   (default: sux-rs/ submodule)
    BURR_DIR      Path to BuRR repo     (default: BuRR/ submodule)
    RESULTS_DIR   Path to results dir   (default: ./results)
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sibling modules importable when invoked as "python bench/run.py".
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import (  # noqa: E402
    ALL_BITS,
    DEFAULT_PAR_JOBS,
    DEFAULT_THREADS,
    EVEN_BITS,
    FIGURES,
    METHODS,
    MIN_LOG2,
    ODD_BITS,
    QUERY_KEYS,
    SUX_FEATURES,
    SUX_PROFILE,
)
from parse import (compute_overhead, distribution_entropy, parse_build,
                    parse_burr_size, parse_query)

# Repo root: bench/ is one level below the repository root, which
# contains the sux-rs/ and BuRR/ submodules.
REPO_ROOT = str(Path(__file__).resolve().parent.parent)


# =====================================================================
#  Helpers
# =====================================================================


def sux_target_dir(sux_dir):
    """Return the cargo target directory for the sux-rs crate.

    Respects CARGO_TARGET_DIR if set (resolved relative to sux_dir,
    matching cargo's behavior when invoked from that directory).
    Falls back to <sux_dir>/target.
    """
    v = os.environ.get("CARGO_TARGET_DIR")
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = Path(sux_dir) / p
        return p
    return Path(sux_dir) / "target"


_PARSE_TOOL_MAP = {"csf": "burr", "comp-vfunc": "sux", "phast": "sux",
                   "mmphf": "sux", "lemonhash": "sux", "java": "sux",
                   "mmphf-str": "sux", "lemonhash-str": "sux"}


def _parse_tool(tool):
    """Map a method's tool type to the parser tool name."""
    return _PARSE_TOOL_MAP.get(tool, tool)


def result_complete(path, tool, no_queries=False):
    """Check whether a result file exists and contains valid timing data.

    A .err file can exist but be incomplete if the benchmark was killed
    mid-run (e.g. after the build phase wrote its output but before the
    query phase appended its results).  We require both construction
    *and* query timing to be present, unless ``no_queries`` is True.
    """
    if not path.exists():
        return False
    text = path.read_text()
    bd = parse_build(text, tool)
    if "constr_ns" not in bd:
        return False
    if no_queries:
        return True
    qd = parse_query(text, tool)
    return "query_min" in qd


def resolve_methods(arg):
    """Turn a --method argument into a list of method names.

    Accepts "all", a comma-separated list, or a single name.
    """
    if arg == "all":
        return list(METHODS.keys())
    names = [s.strip() for s in arg.split(",")]
    for n in names:
        if n not in METHODS:
            sys.exit(f"Unknown method: {n}\nAvailable: {', '.join(METHODS)}")
    return names


def resolve_bits(arg):
    """Turn a --bits argument into a list of integer bit widths.

    Accepts "all", "even", "odd", or a comma-separated list of ints.
    """
    if arg == "all":
        return ALL_BITS
    if arg == "even":
        return EVEN_BITS
    if arg == "odd":
        return ODD_BITS
    return [int(b) for b in arg.split(",")]


def result_dir(base, method):
    """Return (and create) the results directory for a method."""
    d = Path(base) / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(base, method):
    """Return (and create) the directory for ephemeral serialized files."""
    d = Path(base) / method / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def par_query_limit(bits):
    """Maximum n for which parallel queries are feasible.

    With 96 instances each mapping the structure, the total memory is
    roughly 96 × file_size.  The old scripts capped this at
    (2^30 × 8) / effective_bits, where effective_bits rounds down to
    the nearest multiple of 8 (so 9→8, 17→16, 33→32).  This keeps
    total memory under ~96 GB.
    """
    effective = (bits // 8) * 8
    if effective == 0:
        return 0
    return (1 << 30) * 8 // effective


# =====================================================================
#  Binary builders
# =====================================================================


def ensure_sux_built(sux_dir, need_comp_vfunc=False):
    """Compile the sux-rs binaries if they are not already present.

    Runs cargo build with the bench profile and the feature set needed
    for all current benchmarks.
    """
    sux = Path(sux_dir)
    print(f"Building sux-rs binaries in {sux}…")
    cmd = [
        "cargo", "build",
        "--profile", SUX_PROFILE,
        "--features", SUX_FEATURES,
        "--bin", "vfilter",
        "--bin", "vfunc",
        "--example", "bench_vfilter",
        "--example", "bench_vfunc",
        "--bin", "comp_vfunc",
        "--example", "bench_comp_vfunc",
        "--example", "bench_phast",
    ]
    r = subprocess.run(cmd, cwd=str(sux), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"cargo build failed (exit {r.returncode})")


def ensure_csf_built():
    """Ensure bench_csf is built via the top-level Makefile."""
    print("Building bench_csf…")
    r = subprocess.run(
        ["make", "bench_csf"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("WARNING: bench_csf build failed, skipping CSF methods")
        return False
    return True


def ensure_burr_built(burr_dir, bits_values):
    """Compile BuRR binaries for each requested bit width.

    BuRR compiles the bit width into the binary via -DRIBBON_BITS=X,
    producing e.g. bench8, bench16, bench32.  We use the 'bench'
    Makefile target (-O3 -DNDEBUG), not 'ribbon' (-O2, assertions on).
    """
    burr = Path(burr_dir)
    if not burr.exists():
        print(f"WARNING: BuRR directory {burr} not found, skipping BuRR methods")
        return False

    # GCC 15+ miscompiles LeMonHash/SimpleRibbon at -O3 without -fno-strict-aliasing
    #
    # The BuRR Makefile's bench target uses wildcard header deps that
    # cause unconditional rebuilds.  We check source timestamps ourselves
    # and skip if the binary is newer than all sources.
    src_mtime = max(
        (f.stat().st_mtime for f in burr.iterdir()
         if f.suffix in (".cpp", ".hpp") or f.name == "Makefile"),
        default=0.0,
    )
    for bits in bits_values:
        binary = burr / f"bench{bits}"
        if binary.exists() and binary.stat().st_mtime >= src_mtime:
            continue
        print(f"Building BuRR bench{bits} in {burr}…")
        r = subprocess.run(
            ["make", f"RIBBON_BITS={bits}",
             "OPTFLAGS=-O3 -DNDEBUG -march=native -fPIE -fno-strict-aliasing", "bench"],
            cwd=str(burr), capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            sys.exit(f"BuRR make failed for RIBBON_BITS={bits}")
    return True


def ensure_lemonhash_built():
    """Ensure bench_lemonhash is built via the top-level Makefile."""
    print("Building bench_lemonhash…")
    r = subprocess.run(
        ["make", "bench_lemonhash"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("WARNING: bench_lemonhash build failed, skipping LeMonHash")
        return False
    return True


def ensure_lemonhash_str_built():
    """Ensure bench_lemonhash_str is built via the top-level Makefile."""
    print("Building bench_lemonhash_str…")
    r = subprocess.run(
        ["make", "bench_lemonhash_str"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("WARNING: bench_lemonhash_str build failed, skipping LeMonHashStr")
        return False
    return True


# --- String dataset helpers ---

# Directory containing sorted string datasets (one string per line).
DATA_DIR = os.path.join(REPO_ROOT, "data")


def list_datasets():
    """Return sorted list of dataset filenames in the data/ directory.

    Returns an empty list if the directory does not exist.
    """
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        f for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
        and not f.startswith(".")
    )


def ensure_mmphf_built(sux_dir):
    """Ensure lcp_mmphf and bench_lcp_mmphf are built in sux-rs."""
    sux = Path(sux_dir)
    print(f"Building lcp_mmphf + bench_lcp_mmphf in {sux}…")
    cmd = [
        "cargo", "build",
        "--profile", SUX_PROFILE,
        "--features", SUX_FEATURES,
        "--bin", "lcp_mmphf",
        "--example", "bench_lcp_mmphf",
    ]
    r = subprocess.run(cmd, cwd=str(sux), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"cargo build failed for lcp_mmphf (exit {r.returncode})")


def ensure_java_built():
    """Compile the Java benchmark builder."""
    java_dir = Path(REPO_ROOT) / "java"
    print("Building Java benchmark builder…")
    mvnw = java_dir / "mvnw"
    r = subprocess.run(
        [str(mvnw), "-q", "package", "-DskipTests"],
        cwd=str(java_dir), capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("WARNING: Java build failed, skipping GOV3 methods")
        return False
    return True


_SUX4J_C_VARIANTS = {}
for _b in EVEN_BITS:
    _SUX4J_C_VARIANTS[(_b, False)] = (f"test_sf3_filter{_b}", [f"-DSF_{_b}", "-DFILTER"])
for _b in ODD_BITS:
    _SUX4J_C_VARIANTS[(_b, False)] = (f"test_sf3_filter{_b}", ["-DFILTER"])
_SUX4J_C_VARIANTS[("csf", True)] = ("test_csf3_uint64", [])


def ensure_sux4j_c_built(bits_values):
    """Compile the Sux4J C test harnesses for the requested bit widths."""
    sux4j_c = Path(REPO_ROOT) / "Sux4J" / "c"
    if not sux4j_c.exists():
        print("WARNING: Sux4J/c/ not found, skipping GOV3 methods")
        return False

    cc = os.environ.get("CC", "cc")
    cflags = ["-O3", "-march=native"]

    for bits in bits_values:
        key = (bits, False)
        if key not in _SUX4J_C_VARIANTS:
            continue
        binary_name, extra_flags = _SUX4J_C_VARIANTS[key]
        binary_path = sux4j_c / binary_name
        print(f"  Building {binary_name}…")
        cmd = [cc] + cflags + extra_flags + [
            "-o", str(binary_path),
            str(sux4j_c / "test_sf3_uint64_t.c"),
            str(sux4j_c / "sf3.c"),
            str(sux4j_c / "sf.c"),
            str(sux4j_c / "spooky.c"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            print(f"WARNING: failed to compile {binary_name}")
            return False

    key = ("csf", True)
    binary_name, extra_flags = _SUX4J_C_VARIANTS[key]
    binary_path = sux4j_c / binary_name
    print(f"  Building {binary_name}…")
    cmd = [cc] + cflags + extra_flags + [
        "-o", str(binary_path),
        str(sux4j_c / "test_csf3_uint64_t.c"),
        str(sux4j_c / "csf3.c"),
        str(sux4j_c / "csf.c"),
        str(sux4j_c / "spooky.c"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print(f"WARNING: failed to compile {binary_name}")
        return False

    return True


# =====================================================================
#  Command constructors
# =====================================================================


def sux_build_cmd(cfg, n, bits, outfile, args):
    """Assemble the vfilter build command line.

    Example (odd bits):
        vfilter -n 1048576 --shard-edge fuse-lge3-shards --bits 9
                --unaligned --threads 32 output.bin

    --unaligned is only valid for non-power-of-2 bit widths (9, 17, 33).
    For power-of-2 widths (8, 16, 32) the backend is a boxed slice of
    the matching primitive, which is inherently aligned.
    """
    target = sux_target_dir(args.sux_dir)
    cmd = [str(target / "release" / cfg["build_bin"])]
    cmd += ["-n", str(n)]
    cmd += ["--shard-edge", cfg["shard_edge"]]
    if bits > 0:
        cmd += ["--bits", str(bits)]
        if bits not in EVEN_BITS:
            cmd += ["--unaligned"]

    # Per-method extras: --eps, --seed, etc.
    cmd += cfg.get("build_extra", [])

    # Thread count (omit if None = default / single-threaded)
    threads = cfg.get("build_threads")
    if args.threads is not None:
        threads = args.threads
    if threads is not None:
        cmd += ["--threads", str(threads)]

    # Dynamic flag: --low-mem for very large instances
    log2_n = n.bit_length() - 1
    threshold = cfg.get("low_mem_threshold")
    if threshold is not None and log2_n >= threshold:
        cmd += ["--low-mem"]

    # Positional: output file (ε-serde serialized structure)
    cmd += [str(outfile)]
    return cmd


def sux_query_cmd(cfg, n, bits, infile, args):
    """Assemble the bench_vfilter query command line.

    Example (odd bits):
        bench_vfilter 100000000 output.bin --edge fuse-lge3-shards
                      --bits 9 --unaligned
    """
    target = sux_target_dir(args.sux_dir)
    cmd = [str(target / "release" / "examples" / cfg["bench_bin"])]
    cmd += [str(QUERY_KEYS)]             # number of query operations
    cmd += [str(infile)]                 # serialized structure
    cmd += ["--edge", cfg["shard_edge"]]
    if bits > 0:
        cmd += ["--bits", str(bits)]
    if bits > 0 and bits not in EVEN_BITS:
        cmd += ["--unaligned"]
    return cmd


def burr_bench_cmd(cfg, n, bits, args, serialize_to=None):
    """Assemble the BuRR ribbon build command line.

    Pass serialize_to to write the structure for later query / par-query.
    Caller should append -Q to suppress queries during the build step.

    Example output:
        bench8 -t 32 -L 64 -2 -I -n 1048576 -w output.bin
    """
    burr = Path(args.burr_dir)
    binary = f"{cfg['burr_binary']}{bits}"
    cmd = [str(burr / binary)]

    threads = cfg.get("burr_threads", 1)
    if args.threads is not None:
        threads = args.threads
    cmd += ["-t", str(threads)]

    cmd += cfg["burr_flags"]
    cmd += ["-n", str(n)]

    if serialize_to is not None:
        cmd += ["-w", str(serialize_to)]

    return cmd


def burr_query_cmd(cfg, n, bits, infile, args):
    """Assemble the BuRR query-only command line.

    Uses -r to deserialize a previously built structure, then runs
    the built-in query benchmark (100M keys, 5 repeats).

    Example output:
        bench8 -t 1 -L 64 -2 -I -n 1048576 -r input.bin
    """
    burr = Path(args.burr_dir)
    binary = f"{cfg['burr_binary']}{bits}"
    cmd = [str(burr / binary)]
    cmd += ["-t", "1"]
    cmd += cfg["burr_flags"]
    cmd += ["-n", str(n)]
    cmd += ["-r", str(infile)]
    return cmd


def comp_vfunc_build_cmd(cfg, n, outfile, args):
    """Assemble the comp_vfunc build command line.

    Example:
        comp_vfunc -n 1048576 --geometric --shard-edge fuse3-shards
                   --threads 32 output.bin
    """
    target = sux_target_dir(args.sux_dir)
    cmd = [str(target / "release" / "comp_vfunc")]
    cmd += ["-n", str(n)]
    cmd += cfg["distribution"]
    cmd += ["--shard-edge", cfg["shard_edge"]]
    cmd += ["--huffman-entropy-threshold", "1"]

    threads = cfg.get("build_threads")
    if args.threads is not None:
        threads = args.threads
    if threads is not None:
        cmd += ["--threads", str(threads)]

    cmd += [str(outfile)]
    return cmd


def comp_vfunc_query_cmd(cfg, n, infile, args):
    """Assemble the bench_comp_vfunc query command line.

    Example:
        bench_comp_vfunc 100000000 output.bin --edge fuse3-shards
    """
    target = sux_target_dir(args.sux_dir)
    cmd = [str(target / "release" / "examples" / "bench_comp_vfunc")]
    cmd += [str(QUERY_KEYS)]
    cmd += [str(infile)]
    cmd += ["--edge", cfg["shard_edge"]]
    return cmd


def csf_build_cmd(cfg, n, outfile):
    """Assemble the bench_csf build command (no queries, serialize).

    Example:
        bench_csf -n 1048576 --geometric --coder huffman -Q -w output.bin
    """
    cmd = [str(Path(REPO_ROOT) / "bench_csf")]
    cmd += ["-n", str(n)]
    cmd += cfg["distribution"]
    cmd += ["--coder", cfg["coder"]]
    cmd += ["-Q", "-w", str(outfile)]
    return cmd


def csf_query_cmd(cfg, n, infile):
    """Assemble the bench_csf query command (deserialize, run queries).

    Example:
        bench_csf -n 1048576 --coder huffman -r input.bin
    """
    cmd = [str(Path(REPO_ROOT) / "bench_csf")]
    cmd += ["-n", str(n)]
    cmd += ["--coder", cfg["coder"]]
    cmd += ["-r", str(infile)]
    return cmd


def java_build_cmd(cfg, n, bits, outfile):
    """Assemble the Java BenchBuilder command line."""
    java_dir = Path(REPO_ROOT) / "java"
    jar = java_dir / "target" / "ecs-bench-java-1.0-SNAPSHOT.jar"
    classpath = f"{jar}:{java_dir / 'target' / 'lib' / '*'}"
    mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    xmx = int(mem * 0.9) // (1 << 20)
    cmd = ["java", f"-Xmx{xmx}m", "-cp", classpath, "it.unimi.dsi.ecs.BenchBuilder"]

    if cfg["java_type"] == "GOV3Function":
        cmd += ["GOV3Function", str(n), str(bits), str(outfile)]
    elif cfg["java_type"] == "GV3CompressedFunction":
        cmd += ["GV3CompressedFunction", str(n)]
        cmd += cfg["distribution"]
        cmd += [str(outfile)]
    return cmd


def java_query_cmd(cfg, bits, dumpfile):
    """Assemble the Sux4J C query harness command line."""
    sux4j_c = Path(REPO_ROOT) / "Sux4J" / "c"
    if cfg["java_type"] == "GV3CompressedFunction":
        binary = "test_csf3_uint64"
    else:
        binary = f"test_sf3_filter{bits}"
    return [str(sux4j_c / binary), str(dumpfile)]


def run_cmd(cmd, pin=None, dry_run=False):
    """Run a benchmark command, optionally prefixed by a pinning command.

    Args:
        cmd:     list of command-line tokens.
        pin:     optional CPU-pinning prefix, e.g. "chrt -f 99 taskset -c 2".
                 Parsed with shlex.split and prepended to cmd.
        dry_run: if True, print the command but don't execute.

    Returns:
        subprocess.CompletedProcess, or None if dry_run.
    """
    if pin:
        cmd = shlex.split(pin) + cmd

    if dry_run:
        print(f"  [dry-run] {shlex.join(cmd)}")
        return None

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode}): {shlex.join(cmd)}", file=sys.stderr)
        for stream in (r.stderr, r.stdout):
            if stream:
                for line in stream.strip().splitlines():
                    print(f"    {line}", file=sys.stderr)
    return r


# =====================================================================
#  Subcommand: bench
# =====================================================================


def cmd_bench(args):
    """Build structures, query them, and delete the serialized file.

    For each (n, method, bits) triple, ordered by size first:
      1. Skip if bench.<n>.<bits>.err already exists (resume support).
      2. Build the structure (unpinned), serialize to disk.
      3. Run the query benchmark (pinned).
      4. Delete the serialized file (unless --keep-data).
    """
    methods = resolve_methods(args.method)
    bits_list = resolve_bits(args.bits)

    # Ensure binaries are compiled.
    sux_needed = any(METHODS[m]["tool"] == "sux" for m in methods)
    comp_vfunc_needed = any(METHODS[m]["tool"] == "comp-vfunc" for m in methods)
    phast_needed = any(METHODS[m]["tool"] == "phast" for m in methods)
    burr_needed = any(METHODS[m]["tool"] == "burr" for m in methods)
    csf_needed = any(METHODS[m]["tool"] == "csf" for m in methods)
    mmphf_needed = any(METHODS[m]["tool"] == "mmphf" for m in methods)
    mmphf_str_needed = any(METHODS[m]["tool"] == "mmphf-str" for m in methods)
    lemonhash_needed = any(METHODS[m]["tool"] == "lemonhash" for m in methods)
    lemonhash_str_needed = any(METHODS[m]["tool"] == "lemonhash-str" for m in methods)
    java_needed = any(METHODS[m]["tool"] == "java" for m in methods)
    burr_ok = True
    csf_ok = True
    lemonhash_ok = True
    lemonhash_str_ok = True
    java_ok = True

    if sux_needed or comp_vfunc_needed or phast_needed:
        ensure_sux_built(args.sux_dir, need_comp_vfunc=comp_vfunc_needed)
    if mmphf_needed or mmphf_str_needed:
        ensure_mmphf_built(args.sux_dir)
    if burr_needed:
        burr_ok = ensure_burr_built(args.burr_dir, bits_list)
    if csf_needed:
        csf_ok = ensure_csf_built()
    if lemonhash_needed:
        lemonhash_ok = ensure_lemonhash_built()
    if lemonhash_str_needed:
        lemonhash_str_ok = ensure_lemonhash_str_built()
    if java_needed:
        java_ok = ensure_java_built()
        if java_ok:
            filter_bits = bits_list if any(
                METHODS[m].get("java_type") == "GOV3Function" for m in methods
                if METHODS[m]["tool"] == "java") else []
            java_ok = ensure_sux4j_c_built(filter_bits)

    # --- String-dataset methods (iterate over files in data/) ---
    datasets = list_datasets()
    str_methods = [m for m in methods
                   if METHODS[m]["tool"] in ("mmphf-str", "lemonhash-str")]
    if str_methods and not datasets:
        print("Skipping string methods (no datasets in data/)")
        str_methods = []

    for method_name in str_methods:
        cfg = METHODS[method_name]
        if cfg["tool"] == "lemonhash-str" and not lemonhash_str_ok:
            print(f"Skipping {method_name} (bench_lemonhash_str not available)")
            continue
        rdir = result_dir(args.results_dir, method_name)
        tool_for_parse = _parse_tool(cfg["tool"])
        for dataset in datasets:
            bench_err = rdir / f"bench.{dataset}.0.err"
            if not args.force and result_complete(bench_err, tool_for_parse):
                continue
            if cfg["tool"] == "mmphf-str":
                _bench_mmphf_str(cfg, method_name, dataset, rdir, args)
            elif cfg["tool"] == "lemonhash-str":
                _bench_lemonhash_str(cfg, method_name, dataset, rdir, args)

    # --- Size-based methods (the standard log2 iteration) ---
    # Build the work list: (log2_n, method_name, bits) triples,
    # ordered by size first so that if a method OOMs we still have
    # complete data at every smaller size.
    work = []
    for method_name in methods:
        cfg = METHODS[method_name]

        if cfg["tool"] in ("mmphf-str", "lemonhash-str"):
            continue
        if cfg["tool"] == "burr" and not burr_ok:
            print(f"Skipping {method_name} (BuRR not available)")
            continue
        if cfg["tool"] == "csf" and not csf_ok:
            print(f"Skipping {method_name} (bench_csf not available)")
            continue
        if cfg["tool"] == "lemonhash" and not lemonhash_ok:
            print(f"Skipping {method_name} (bench_lemonhash not available)")
            continue
        if cfg["tool"] == "java" and not java_ok:
            print(f"Skipping {method_name} (Java/Sux4J not available)")
            continue

        max_log2 = min(args.max_log2, cfg.get("max_log2", 33))
        bits_agnostic = cfg.get("bits_agnostic", cfg["tool"] in (
            "csf", "comp-vfunc", "phast", "mmphf", "lemonhash"))
        method_bits = [0] if bits_agnostic else bits_list

        explicit_sizes = cfg.get("sizes")
        for bits in method_bits:
            if explicit_sizes is not None:
                for n in explicit_sizes:
                    work.append((n.bit_length() - 1, method_name, bits))
            else:
                for log2_n in range(MIN_LOG2, max_log2 + 1):
                    work.append((log2_n, method_name, bits))

    work.sort(key=lambda t: t[0])

    for log2_n, method_name, bits in work:
        cfg = METHODS[method_name]
        n = 1 << log2_n
        rdir = result_dir(args.results_dir, method_name)

        bench_err = rdir / f"bench.{n}.{bits}.err"
        tool_for_parse = _parse_tool(cfg["tool"])
        if not args.force and result_complete(
                bench_err, tool_for_parse,
                no_queries=cfg.get("no_queries", False)):
            continue

        if cfg["tool"] == "comp-vfunc":
            _bench_comp_vfunc(cfg, method_name, n, log2_n,
                             rdir, args)
        elif cfg["tool"] == "csf":
            _bench_csf(cfg, method_name, n, log2_n,
                           rdir, args)
        elif cfg["tool"] == "burr":
            _bench_burr(cfg, method_name, n, bits, log2_n,
                        rdir, args)
        elif cfg["tool"] == "phast":
            _bench_phast(cfg, method_name, n, log2_n,
                         rdir, args)
        elif cfg["tool"] == "mmphf":
            _bench_mmphf(cfg, method_name, n, log2_n,
                         rdir, args)
        elif cfg["tool"] == "lemonhash":
            _bench_lemonhash(cfg, method_name, n, log2_n,
                             rdir, args)
        elif cfg["tool"] == "java":
            _bench_java(cfg, method_name, n, bits, log2_n,
                        rdir, args)
        else:
            _bench_sux(cfg, method_name, n, bits, log2_n,
                       rdir, args)

    print("Done.")


def _bench_burr(cfg, method_name, n, bits, log2_n, rdir, args):
    """Run a single BuRR benchmark: build, serialize, query, delete.

    Separating build from query is essential for reliable query timings:
    the build can use many threads, but queries must run on a pinned
    core (taskset -c 2) without interference from the parallel builder.

    Steps:
      1. Build with -Q (no queries) and -w (serialize).
      2. Query with -r (deserialize) on a single thread, pinned.
      3. Delete the serialized file (unless --keep-data).
    """
    ddir = data_dir(args.results_dir, method_name)
    outfile = ddir / f"filter.{n}.{bits}.bin"

    # --- Build (no queries, serialize, unpinned) ---
    cmd = burr_bench_cmd(cfg, n, bits, args, serialize_to=outfile)
    cmd += ["-Q"]

    bits_str = f" bits={bits}" if bits > 0 else ""
    print(f"  bench {method_name} n=2^{log2_n}{bits_str}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    # --- Query (deserialize, single-threaded, pinned) ---
    qr = None
    if not cfg.get("no_queries"):
        qcmd = burr_query_cmd(cfg, n, bits, outfile, args)
        if args.dry_run or result.returncode == 0:
            qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    if result.returncode != 0:
        (rdir / f"bench.{n}.{bits}.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    build_output = result.stdout
    (rdir / f"bench.{n}.{bits}.err").write_text(build_output)

    mem_size = parse_burr_size(build_output)
    if mem_size is not None:
        (rdir / f"bench.{n}.{bits}.size").write_text(str(mem_size))

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{n}.{bits}.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stdout)

    # --- Clean up ---
    if not args.keep_data and outfile.exists():
        outfile.unlink()


def _bench_comp_vfunc(cfg, method_name, n, log2_n, rdir, args):
    """Run a single CompVFunc benchmark: build, query, delete.

    Two-step like VFilter: build with comp_vfunc, query with
    bench_comp_vfunc.  Results stored with bits=0.
    """
    ddir = data_dir(args.results_dir, method_name)
    outfile = ddir / f"func.{n}.bin"

    # --- Build (unpinned) ---
    cmd = comp_vfunc_build_cmd(cfg, n, outfile, args)
    print(f"  bench {method_name} n=2^{log2_n}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    # --- Query (pinned, only if build succeeded) ---
    qr = None
    if args.dry_run or result.returncode == 0:
        qcmd = comp_vfunc_query_cmd(cfg, n, outfile, args)
        qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    build_output = result.stderr
    if result.returncode != 0:
        (rdir / f"bench.{n}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    (rdir / f"bench.{n}.0.err").write_text(build_output)

    if outfile.exists():
        (rdir / f"bench.{n}.0.size").write_text(
            str(outfile.stat().st_size))

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{n}.0.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stderr)

    # --- Clean up ---
    if not args.keep_data and outfile.exists():
        outfile.unlink()


def _bench_phast(cfg, method_name, n, log2_n, rdir, args):
    """Run a PHast+ benchmark: build and query in one invocation.

    bench_phast <n> <query_keys> <threads> --repeats 5
    Output (construction + queries) goes to stderr.
    """
    target = sux_target_dir(args.sux_dir)
    cmd = [str(target / "release" / "examples" / "bench_phast")]
    cmd += [str(n), str(QUERY_KEYS)]

    threads = cfg.get("build_threads")
    if args.threads is not None:
        threads = args.threads
    cmd += [str(threads or 1)]

    print(f"  bench {method_name} n=2^{log2_n}")
    result = run_cmd(cmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    if result.returncode != 0:
        (rdir / f"bench.{n}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    (rdir / f"bench.{n}.0.err").write_text(result.stderr)


def _bench_mmphf(cfg, method_name, n, log2_n, rdir, args):
    """Run a LcpMmphf/Lcp2Mmphf benchmark: build, then query.

    Build:  lcp_mmphf -n <N> --unaligned [--threads 1] [-2] <outfile>
    Query:  bench_lcp_mmphf <n> <outfile> --unaligned [-2] --repeats 5
    """
    target = sux_target_dir(args.sux_dir)
    ddir = data_dir(args.results_dir, method_name)
    outfile = ddir / f"func.{n}.bin"

    # --- Build (unpinned) ---
    cmd = [str(target / "release" / "lcp_mmphf")]
    cmd += ["-n", str(n)]
    cmd += ["--unaligned"]
    cmd += ["--threads", "1"]
    if cfg.get("variant") == "lcp2":
        cmd += ["-2"]
    cmd += [str(outfile)]

    print(f"  bench {method_name} n=2^{log2_n}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    # --- Query (pinned, only if build succeeded) ---
    qkeys = cfg.get("query_keys", QUERY_KEYS)
    qcmd = [str(target / "release" / "examples" / "bench_lcp_mmphf")]
    qcmd += [str(qkeys), str(outfile)]
    qcmd += ["--unaligned"]
    if cfg.get("variant") == "lcp2":
        qcmd += ["-2"]
    qcmd += ["--repeats", "5"]
    qr = None
    if args.dry_run or result.returncode == 0:
        qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    build_output = result.stderr
    if result.returncode != 0:
        (rdir / f"bench.{n}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    (rdir / f"bench.{n}.0.err").write_text(build_output)

    if outfile.exists():
        (rdir / f"bench.{n}.0.size").write_text(
            str(outfile.stat().st_size))

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{n}.0.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stderr)

    # --- Clean up ---
    if not args.keep_data and outfile.exists():
        outfile.unlink()


def _bench_lemonhash(cfg, method_name, n, log2_n, rdir, args):
    """Run a LeMonHash benchmark: build and query in one invocation.

    bench_lemonhash <n> <query_keys> <repeats>
    Output goes to stderr.
    """
    cmd = [str(Path(REPO_ROOT) / "bench_lemonhash")]
    qkeys = cfg.get("query_keys", QUERY_KEYS)
    cmd += [str(n), str(qkeys), "5"]

    print(f"  bench {method_name} n=2^{log2_n}")
    result = run_cmd(cmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    if result.returncode != 0:
        (rdir / f"bench.{n}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    output = result.stderr
    (rdir / f"bench.{n}.0.err").write_text(output)

    mem_size = parse_burr_size(output)
    if mem_size is not None:
        (rdir / f"bench.{n}.0.size").write_text(str(mem_size))


def _bench_mmphf_str(cfg, method_name, dataset, rdir, args):
    """Run a LcpMmphfStr/Lcp2MmphfStr benchmark on a string dataset.

    Build:  lcp_mmphf -f <datafile> --unaligned --threads 1 [-2] <outfile>
    Query:  bench_lcp_mmphf <query_keys> <outfile> -f <datafile>
            --unaligned [-2] --repeats 5
    """
    target = sux_target_dir(args.sux_dir)
    ddir = data_dir(args.results_dir, method_name)
    outfile = ddir / f"func.{dataset}.bin"
    datafile = str(Path(DATA_DIR) / dataset)

    # --- Build (unpinned) ---
    cmd = [str(target / "release" / "lcp_mmphf")]
    cmd += ["-f", datafile]
    cmd += ["--unaligned"]
    cmd += ["--threads", "1"]
    if cfg.get("variant") == "lcp2":
        cmd += ["-2"]
    cmd += [str(outfile)]

    print(f"  bench {method_name} dataset={dataset}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    # --- Query (pinned, only if build succeeded) ---
    qkeys = cfg.get("query_keys", QUERY_KEYS)
    qcmd = [str(target / "release" / "examples" / "bench_lcp_mmphf")]
    qcmd += [str(qkeys), str(outfile)]
    qcmd += ["-f", datafile]
    qcmd += ["--unaligned"]
    if cfg.get("variant") == "lcp2":
        qcmd += ["-2"]
    qcmd += ["--repeats", "5"]
    qr = None
    if args.dry_run or result.returncode == 0:
        qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    build_output = result.stderr
    if result.returncode != 0:
        (rdir / f"bench.{dataset}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    (rdir / f"bench.{dataset}.0.err").write_text(build_output)

    if outfile.exists():
        (rdir / f"bench.{dataset}.0.size").write_text(
            str(outfile.stat().st_size))

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{dataset}.0.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stderr)

    # --- Clean up ---
    if not args.keep_data and outfile.exists():
        outfile.unlink()


def _bench_lemonhash_str(cfg, method_name, dataset, rdir, args):
    """Run a LeMonHashVL benchmark on a string dataset.

    bench_lemonhash_str <file> <query_keys> <repeats>
    Output goes to stderr.
    """
    cmd = [str(Path(REPO_ROOT) / "bench_lemonhash_str")]
    qkeys = cfg.get("query_keys", QUERY_KEYS)
    datafile = str(Path(DATA_DIR) / dataset)
    cmd += [datafile, str(qkeys), "5"]

    print(f"  bench {method_name} dataset={dataset}")
    result = run_cmd(cmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    if result.returncode != 0:
        (rdir / f"bench.{dataset}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    output = result.stderr
    (rdir / f"bench.{dataset}.0.err").write_text(output)

    mem_size = parse_burr_size(output)
    if mem_size is not None:
        (rdir / f"bench.{dataset}.0.size").write_text(str(mem_size))


def _bench_csf(cfg, method_name, n, log2_n, rdir, args):
    """Run a single CSF benchmark: build, serialize, query, delete.

    Same separation as BuRR: build unpinned with -Q -w, query pinned
    with -r.  Results stored with bits=0.
    """
    ddir = data_dir(args.results_dir, method_name)
    outfile = ddir / f"csf.{n}.bin"

    # --- Build (unpinned, no queries, serialize) ---
    cmd = csf_build_cmd(cfg, n, outfile)
    print(f"  bench {method_name} n=2^{log2_n}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    # --- Query (pinned, only if build succeeded) ---
    qr = None
    if args.dry_run or result.returncode == 0:
        qcmd = csf_query_cmd(cfg, n, outfile)
        qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    build_output = result.stdout
    if result.returncode != 0:
        (rdir / f"bench.{n}.0.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    (rdir / f"bench.{n}.0.err").write_text(build_output)

    mem_size = parse_burr_size(build_output)
    if mem_size is not None:
        (rdir / f"bench.{n}.0.size").write_text(str(mem_size))

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{n}.0.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stdout)

    # --- Clean up ---
    if not args.keep_data and outfile.exists():
        outfile.unlink()


def _bench_sux(cfg, method_name, n, bits, log2_n, rdir, args):
    """Run a single sux-rs benchmark: build, query, delete.

    Build with vfilter (unpinned, may use multiple threads), then
    query with bench_vfilter (pinned), then delete.
    """
    ddir = data_dir(args.results_dir, method_name)
    outfile = ddir / f"filter.{n}.{bits}.bin"

    # --- Build (unpinned) ---
    cmd = sux_build_cmd(cfg, n, bits, outfile, args)
    bits_str = f" bits={bits}" if bits > 0 else ""
    print(f"  bench {method_name} n=2^{log2_n}{bits_str}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    # --- Query (pinned, only if build succeeded) ---
    qr = None
    if not cfg.get("no_queries"):
        qcmd = sux_query_cmd(cfg, n, bits, outfile, args)
        if args.dry_run or result.returncode == 0:
            qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if args.dry_run:
        return

    build_output = result.stderr
    if result.returncode != 0:
        (rdir / f"bench.{n}.{bits}.err").write_text(
            (result.stdout or "") + (result.stderr or ""))
        return

    (rdir / f"bench.{n}.{bits}.err").write_text(build_output)

    if outfile.exists():
        (rdir / f"bench.{n}.{bits}.size").write_text(
            str(outfile.stat().st_size))

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{n}.{bits}.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stderr)

    # --- Clean up ---
    if not args.keep_data and outfile.exists():
        outfile.unlink()


def _bench_java(cfg, method_name, n, bits, log2_n, rdir, args):
    """Run a single Sux4J benchmark: Java build+dump, C query, clean up."""
    ddir = data_dir(args.results_dir, method_name)
    is_compressed = cfg["java_type"] == "GV3CompressedFunction"
    suffix = "func" if is_compressed else "filter"
    outfile = ddir / f"{suffix}.{n}.{bits}.bin"

    # --- Java build + dump (unpinned) ---
    cmd = java_build_cmd(cfg, n, bits, outfile)
    bits_str = f" bits={bits}" if bits > 0 else ""
    print(f"  bench {method_name} n=2^{log2_n}{bits_str}")
    result = run_cmd(cmd, dry_run=args.dry_run)

    if args.dry_run:
        return

    if result is None or result.returncode != 0:
        err_text = ""
        if result:
            err_text = (result.stdout or "") + (result.stderr or "")
        (rdir / f"bench.{n}.{bits}.err").write_text(err_text)
        return

    (rdir / f"bench.{n}.{bits}.err").write_text(result.stderr or "")

    if outfile.exists():
        (rdir / f"bench.{n}.{bits}.size").write_text(
            str(outfile.stat().st_size))

    # --- C query (pinned) ---
    qcmd = java_query_cmd(cfg, bits, outfile)
    qr = run_cmd(qcmd, pin=args.pin, dry_run=args.dry_run)

    if qr is not None and qr.returncode == 0:
        with open(rdir / f"bench.{n}.{bits}.err", "a") as f:
            f.write("\n--- query ---\n")
            f.write(qr.stdout)

    # --- Clean up ---
    if not getattr(args, 'keep_data', False) and outfile.exists():
        outfile.unlink()


# =====================================================================
#  Subcommand: par-bench
# =====================================================================


def cmd_par_bench(args):
    """Build a structure, run parallel query instances, and clean up.

    For each (method, bits, n) triple:
      1. Build the structure and serialize it to disk.
      2. Launch --jobs query instances via GNU parallel, all reading
         the same serialized file.
      3. Extract the worst-case (maximum) query Max across all
         instances — this measures memory-contention impact.
      4. Delete the serialized file and per-instance outputs.

    For sux-rs: build with vfilter, query with bench_vfilter × jobs.
    For BuRR: build with ribbon -w, query with ribbon -r × jobs.
    """
    methods = resolve_methods(args.method)
    bits_list = resolve_bits(args.bits)

    sux_needed = any(METHODS[m]["tool"] == "sux" for m in methods)
    burr_needed = any(METHODS[m]["tool"] == "burr" for m in methods)
    burr_ok = True

    if sux_needed:
        ensure_sux_built(args.sux_dir)
    if burr_needed:
        burr_ok = ensure_burr_built(args.burr_dir, bits_list)

    # Build work list sorted by size, like cmd_bench.
    # Only filter methods (non-bits-agnostic) make sense for par-bench.
    work = []
    for method_name in methods:
        cfg = METHODS[method_name]

        if cfg.get("no_queries"):
            continue
        if cfg["tool"] == "burr" and not burr_ok:
            print(f"Skipping {method_name} (BuRR not available)")
            continue
        if cfg["tool"] in ("csf", "comp-vfunc", "phast", "mmphf", "lemonhash",
                          "mmphf-str", "lemonhash-str", "java"):
            continue
        if cfg.get("bits_agnostic"):
            continue

        max_log2 = min(args.max_log2, cfg.get("max_log2", 33))

        for bits in bits_list:
            max_n = par_query_limit(bits)
            for log2_n in range(MIN_LOG2, max_log2 + 1):
                n = 1 << log2_n
                if n > max_n:
                    continue
                work.append((log2_n, method_name, bits))

    work.sort(key=lambda t: t[0])

    for log2_n, method_name, bits in work:
        cfg = METHODS[method_name]
        n = 1 << log2_n
        rdir = result_dir(args.results_dir, method_name)
        ddir = data_dir(args.results_dir, method_name)

        agg_err = rdir / f"parbench.{n}.{bits}.err"
        tool_for_parse = _parse_tool(cfg["tool"])
        if not args.force and agg_err.exists() and \
                parse_query(agg_err.read_text(), tool_for_parse):
            continue

        # --- Build and serialize ---
        outfile = ddir / f"filter.{n}.{bits}.bin"

        if cfg["tool"] == "sux":
            build_cmd = sux_build_cmd(cfg, n, bits, outfile, args)
        else:
            build_cmd = burr_bench_cmd(cfg, n, bits, args,
                                       serialize_to=outfile)
            build_cmd += ["-Q"]

        print(f"  par-bench {method_name} n=2^{log2_n} bits={bits}"
              f" (build)")
        br = run_cmd(build_cmd, dry_run=args.dry_run)
        if not args.dry_run and br.returncode != 0:
            continue

        if args.dry_run:
            print(f"  [dry-run] echo <{args.jobs} commands> | "
                  f"parallel -j {args.jobs}")
            continue

        if not outfile.exists():
            print(f"  ERROR: serialized file not created",
                  file=sys.stderr)
            continue

        # --- Parallel queries ---
        if cfg["tool"] == "sux":
            qcmd = sux_query_cmd(cfg, n, bits, outfile, args)
        else:
            qcmd = burr_query_cmd(cfg, n, bits, outfile, args)

        instance_dir = rdir / f"parbench.{n}.{bits}.d"
        instance_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        for i in range(args.jobs):
            out_file = instance_dir / f"{i}.err"
            if cfg["tool"] == "sux":
                job = shlex.join(qcmd) + f" 2>{out_file}"
            else:
                job = shlex.join(qcmd) + f" >{out_file}"
            jobs.append(job)

        print(f"  par-bench {method_name} n=2^{log2_n} bits={bits}"
              f" ({args.jobs} queries)")

        par_input = "\n".join(jobs)
        r = subprocess.run(
            ["parallel", "-j", str(args.jobs)],
            input=par_input, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  WARNING: parallel returned {r.returncode}",
                  file=sys.stderr)

        # Aggregate: median of per-instance median times.
        medians = []
        for i in range(args.jobs):
            ifile = instance_dir / f"{i}.err"
            if not ifile.exists():
                continue
            qdata = parse_query(ifile.read_text(), tool_for_parse)
            if "query_median" in qdata:
                medians.append(qdata["query_median"])
        medians.sort()
        if medians:
            med = medians[len(medians) // 2]
        else:
            med = 0.0

        agg_err.write_text(
            f"Min: {med} Median: {med} "
            f"Max: {med} Average: {med}\n"
        )

        # --- Clean up ---
        shutil.rmtree(instance_dir, ignore_errors=True)
        if outfile.exists():
            outfile.unlink()

    print("Done.")


# =====================================================================
#  Subcommand: report
# =====================================================================


def _parse_n_from_build(text):
    """Extract the key count from sux-rs build output.

    Looks for lines like "Found N keys" or "(N keys," in the
    construction summary.
    """
    import re
    m = re.search(r'(?:Found|Loaded)\s+(\d+)\s+keys', text)
    if m:
        return int(m.group(1))
    m = re.search(r'\((\d+)\s+keys', text)
    if m:
        return int(m.group(1))
    return None


def cmd_report(args):
    """Print a space-aligned report of available results to stdout.

    Columns:
        n  dataset  bits  method  constr_ns  query_min  query_median
        query_max  query_avg  parbench_med  bits_per_key  overhead_pct
        file_bytes

    For filters (sux, burr), overhead_pct is relative to the bit width.
    For compressed functions (csf, comp-vfunc), it is relative to the
    distribution entropy.

    Columns are padded with spaces for readability; spaces are
    non-significant so the output can still be split on whitespace.
    """
    methods = resolve_methods(args.method)
    bits_list = resolve_bits(args.bits)

    cols = ["n", "dataset", "bits", "method", "constr_ns", "query_min",
            "query_median", "query_max", "query_avg", "parbench_med",
            "bits_per_key", "overhead_pct", "file_bytes"]

    rows = []
    for method_name in methods:
        cfg = METHODS[method_name]
        rdir = Path(args.results_dir) / method_name
        if not rdir.exists():
            continue

        tool_for_parse = _parse_tool(cfg["tool"])

        # String-dataset methods: scan the result directory.
        if cfg["tool"] in ("mmphf-str", "lemonhash-str"):
            import re
            for f in sorted(rdir.iterdir()):
                m = re.match(r'bench\.(.+)\.0\.err$', f.name)
                if not m:
                    continue
                dataset = m.group(1)
                text = f.read_text()
                bd = parse_build(text, tool_for_parse)
                qd = parse_query(text, tool_for_parse)

                n = _parse_n_from_build(text)
                n_str = str(n) if n else ""

                size_file = rdir / f"bench.{dataset}.0.size"
                file_bytes = ""
                bpk = ""
                if size_file.exists() and n:
                    file_bytes = int(size_file.read_text().strip())
                    bpk = f"{file_bytes * 8.0 / n:.2f}"

                rows.append([
                    n_str, dataset, "0", method_name,
                    _fmt(bd.get("constr_ns")),
                    _fmt(qd.get("query_min")),
                    _fmt(qd.get("query_median")),
                    _fmt(qd.get("query_max")),
                    _fmt(qd.get("query_avg")),
                    "",
                    str(bpk),
                    "",
                    str(file_bytes),
                ])
            continue

        # Standard size-based methods.
        max_log2 = min(args.max_log2, cfg.get("max_log2", 33))
        bits_agnostic = cfg.get("bits_agnostic", cfg["tool"] in (
            "csf", "comp-vfunc", "phast", "mmphf", "lemonhash"))
        method_bits = [0] if bits_agnostic else bits_list

        entropy = distribution_entropy(cfg.get("distribution"))

        for bits in method_bits:
            for log2_n in range(MIN_LOG2, max_log2 + 1):
                n = 1 << log2_n

                bench_err = rdir / f"bench.{n}.{bits}.err"
                if not bench_err.exists():
                    continue
                text = bench_err.read_text()
                bd = parse_build(text, tool_for_parse)
                qd = parse_query(text, tool_for_parse)

                pq_err = rdir / f"parbench.{n}.{bits}.err"
                pq_med = ""
                if pq_err.exists():
                    pqd = parse_query(pq_err.read_text(), tool_for_parse)
                    pq_med = pqd.get("query_median", "")

                size_file = rdir / f"bench.{n}.{bits}.size"
                file_bytes = ""
                bpk = ""
                overhead = ""
                if size_file.exists():
                    file_bytes = int(size_file.read_text().strip())
                    bpk = f"{file_bytes * 8.0 / n:.2f}"
                    if bits > 0:
                        overhead = f"{compute_overhead(file_bytes, n, bits):.2f}"
                    elif entropy is not None:
                        overhead = f"{compute_overhead(file_bytes, n, entropy):.2f}"

                rows.append([
                    str(n), "", str(bits), method_name,
                    _fmt(bd.get("constr_ns")),
                    _fmt(qd.get("query_min")),
                    _fmt(qd.get("query_median")),
                    _fmt(qd.get("query_max")),
                    _fmt(qd.get("query_avg")),
                    _fmt(pq_med),
                    str(bpk),
                    str(overhead),
                    str(file_bytes),
                ])

    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_row(cols))
    for row in rows:
        print(fmt_row(row))


def _fmt(v):
    """Format a numeric value for TSV, leaving empty string for None."""
    if v is None or v == "":
        return ""
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# =====================================================================
#  Subcommand: plot
# =====================================================================


def cmd_plot(args):
    """Generate paper figures from benchmark results.

    Reads the same result files as 'report' and produces one PDF per
    (figure, bits) combination in the output directory.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for plotting: pip install matplotlib")

    if args.figure == "all":
        figure_names = list(FIGURES.keys())
    else:
        if args.figure not in FIGURES:
            sys.exit(f"Unknown figure: {args.figure}\n"
                     f"Available: {', '.join(FIGURES)}")
        figure_names = [args.figure]

    for figure_name in figure_names:
        _plot_one(figure_name, args, plt)


def _plot_one(figure_name, args, plt):
    """Generate PDFs for a single figure preset."""
    fig_cfg = FIGURES[figure_name]
    bits_list = resolve_bits(args.bits)
    markers = ["o", "s", "^", "v", "p", "*", "h", "D"]
    col = fig_cfg["column"]

    all_agnostic = all(
        METHODS[m].get("bits_agnostic", METHODS[m]["tool"] in ("csf", "comp-vfunc", "phast", "mmphf", "lemonhash"))
        for m in fig_cfg["methods"] if m in METHODS
    )
    if all_agnostic:
        bits_list = [0]

    global_ymax = 0.0
    for bits in bits_list:
        for method_name in fig_cfg["methods"]:
            if method_name not in METHODS:
                continue
            cfg = METHODS[method_name]
            rdir = Path(args.results_dir) / method_name
            if not rdir.exists():
                continue
            max_log2 = min(args.max_log2, cfg.get("max_log2", 33))
            tool_for_parse = _parse_tool(cfg["tool"])
            for log2_n in range(MIN_LOG2, max_log2 + 1):
                n = 1 << log2_n
                bench_err = rdir / f"bench.{n}.{bits}.err"
                pq_err = rdir / f"parbench.{n}.{bits}.err"
                val = None
                if col in ("constr_ns",) and bench_err.exists():
                    val = parse_build(bench_err.read_text(), tool_for_parse).get("constr_ns")
                elif col.startswith("query_") and bench_err.exists():
                    val = parse_query(bench_err.read_text(), tool_for_parse).get(col)
                elif col == "parbench_med" and pq_err.exists():
                    val = parse_query(pq_err.read_text(), tool_for_parse).get("query_median")
                if val is not None and val > global_ymax:
                    global_ymax = val
    global_ymax *= 1.15

    for bits in bits_list:
        plt.figure(figsize=(8, 5.5))
        mi = 0

        for method_name in fig_cfg["methods"]:
            if method_name not in METHODS:
                continue
            cfg = METHODS[method_name]
            rdir = Path(args.results_dir) / method_name
            if not rdir.exists():
                continue

            xs, ys = [], []
            max_log2 = min(args.max_log2, cfg.get("max_log2", 33))
            tool_for_parse = _parse_tool(cfg["tool"])

            for log2_n in range(MIN_LOG2, max_log2 + 1):
                n = 1 << log2_n
                bench_err = rdir / f"bench.{n}.{bits}.err"
                pq_err = rdir / f"parbench.{n}.{bits}.err"

                val = None
                if col in ("constr_ns",) and bench_err.exists():
                    d = parse_build(bench_err.read_text(), tool_for_parse)
                    val = d.get("constr_ns")
                elif col.startswith("query_") and bench_err.exists():
                    d = parse_query(bench_err.read_text(), tool_for_parse)
                    val = d.get(col)
                elif col == "parbench_med" and pq_err.exists():
                    d = parse_query(pq_err.read_text(), tool_for_parse)
                    val = d.get("query_median")

                if val is not None:
                    xs.append(n)
                    ys.append(val)

            if xs:
                plt.plot(xs, ys,
                         marker=markers[mi % len(markers)],
                         linestyle="-",
                         label=cfg["label"])
                mi += 1

        title = fig_cfg["title"].format(bits=bits) if "{bits}" in fig_cfg["title"] else fig_cfg["title"]
        plt.title(title)
        plt.xlabel("Number of keys")
        plt.ylabel(fig_cfg["ylabel"])
        plt.xscale("log")
        if mi > 0:
            plt.legend()
        if global_ymax > 0:
            plt.ylim(0, global_ymax)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if all_agnostic else f"-{bits}"
        out_path = out_dir / f"{figure_name}{suffix}.pdf"
        plt.savefig(str(out_path), format="pdf")
        plt.close()
        print(f"  saved {out_path}")


# =====================================================================
#  Subcommand: status
# =====================================================================


def cmd_status(args):
    """Show a summary of completed and pending benchmark runs."""
    methods = resolve_methods(args.method)
    bits_list = resolve_bits(args.bits)
    datasets = list_datasets()

    for method_name in methods:
        cfg = METHODS[method_name]
        rdir = Path(args.results_dir) / method_name
        tool_for_parse = _parse_tool(cfg["tool"])

        # String-dataset methods: count by dataset files.
        if cfg["tool"] in ("mmphf-str", "lemonhash-str"):
            total = len(datasets)
            done = sum(
                1 for d in datasets
                if result_complete(rdir / f"bench.{d}.0.err", tool_for_parse)
            )
            print(f"{method_name:24s}  bench {done}/{total}")
            continue

        max_log2 = min(args.max_log2, cfg.get("max_log2", 33))

        done_bench, done_pq = 0, 0
        total = 0
        bits_agnostic = cfg.get("bits_agnostic", cfg["tool"] in (
            "csf", "comp-vfunc", "phast", "mmphf", "lemonhash"))
        method_bits = [0] if bits_agnostic else bits_list

        for bits in method_bits:
            for log2_n in range(MIN_LOG2, max_log2 + 1):
                n = 1 << log2_n
                total += 1
                if result_complete(rdir / f"bench.{n}.{bits}.err", tool_for_parse):
                    done_bench += 1
                if (rdir / f"parbench.{n}.{bits}.err").exists():
                    done_pq += 1

        print(f"{method_name:24s}  bench {done_bench}/{total}"
              f"  par-bench {done_pq}/{total}")


# =====================================================================
#  Subcommand: clean
# =====================================================================


def cmd_clean(args):
    """Delete result files and/or serialized data."""
    methods = resolve_methods(args.method)

    for method_name in methods:
        rdir = Path(args.results_dir) / method_name
        if not rdir.exists():
            continue

        if args.data_only:
            # Only delete the serialized structure files.
            ddir = rdir / "data"
            if ddir.exists():
                shutil.rmtree(ddir)
                print(f"  deleted {ddir}")
        else:
            # Delete everything.
            shutil.rmtree(rdir)
            print(f"  deleted {rdir}")


# =====================================================================
#  Subcommand: table
# =====================================================================


# Table presets.  Each maps a table name to:
#   - "methods": ordered list of method names (rows)
#   - "bits_order": ordered list of bit widths (column groups)
#   - "sizes": list of n values (one table per size)
#   - "columns": list of (column_key, header) for each half

TABLES = {
    "filter": {
        "rows": [
            ("BuRR", {"constr_seq": "BuRR-seq", "constr_par": "BuRR",
                      "query_seq": "BuRR", "query_par": "BuRR"}),
            ("VFunc", {"constr_seq": "FuseLge3Shards-seq", "constr_par": "FuseLge3Shards",
                       "query_seq": "FuseLge3Shards", "query_par": "FuseLge3Shards"}),
            ("MWHC", {"constr_seq": None, "constr_par": "Mwhc3",
                      "query_seq": "Mwhc3", "query_par": "Mwhc3"}),
            ("Fuse(64)", {"constr_seq": "Fuse3NoShards64", "constr_par": None,
                          "query_seq": "Fuse3NoShards64", "query_par": "Fuse3NoShards64"}),
            ("Fuse(128)", {"constr_seq": "Fuse3NoShards128", "constr_par": None,
                           "query_seq": "Fuse3NoShards128", "query_par": "Fuse3NoShards128"}),
        ],
        "bits_order": [8, 16, 32, 9, 17, 33],
        "sizes": [1 << 20, 1 << 30],
    },
    "mmphf": {
        "methods": [
            ("LcpMmphfInt", "LcpMmphfInt"),
            ("Lcp2MmphfInt", "Lcp2MmphfInt"),
            ("LeMonHashInt", "LeMonHashInt"),
        ],
        "sizes": [1 << 20, 1 << 31],
    },
    "mmphf-str": {
        "methods": [
            ("LcpMmphfStr", "LcpMmphf"),
            ("Lcp2MmphfStr", "Lcp2Mmphf"),
            ("LeMonHashStr", "LeMonHash"),
        ],
    },
    "comp": {
        "rows": [
            ("CompVFunc", ["CompVFunc-Geo", "CompVFunc-Zipf", "CompVFunc-Uni256"]),
            ("BuRR-CSF", ["CSF-Geo", "CSF-Zipf", "CSF-Uni256"]),
        ],
        "dist_labels": ["Geometric", "Zipf", "Uniform"],
        "sizes": [1 << 20, 1 << 29],
    },
    "java": {
        "rows": [
            ("VFilter", {"query": "FuseLge3Shards"}),
            ("GOV3", {"query": "GOV3Function"}),
        ],
        "bits_order": [8, 16, 32, 9, 17, 33],
        "sizes": [1 << 20, 1 << 30],
    },
    "java-comp": {
        "rows": [
            ("CompVFunc", ["CompVFunc-Geo", "CompVFunc-Zipf", "CompVFunc-Uni256"]),
            ("GOV3Comp", ["GV3CompressedFunction-Geo", "GV3CompressedFunction-Zipf", "GV3CompressedFunction-Uni256"]),
        ],
        "dist_labels": ["Geometric", "Zipf", "Uniform"],
        "sizes": [1 << 20, 1 << 28],
    },
}


def _filter_constr_row(label, variants, bits_order, n, args):
    """Build one construction row (sequential | parallel, integer ns/key)."""
    row = label
    for mode in ("constr_seq", "constr_par"):
        method_name = variants[mode]
        if method_name is None:
            row += " & ---" * len(bits_order)
            continue
        cfg = METHODS.get(method_name)
        if cfg is None:
            row += " &" * len(bits_order)
            continue
        rdir = Path(args.results_dir) / method_name
        tool_for_parse = _parse_tool(cfg["tool"])
        for bits in bits_order:
            bench_err = rdir / f"bench.{n}.{bits}.err"
            val = ""
            if bench_err.exists():
                d = parse_build(bench_err.read_text(), tool_for_parse)
                v = d.get("constr_ns")
                if v is not None:
                    val = f"{v:.0f}"
            row += f" & {val}"
    return row + r" \\"


def _filter_query_row(label, variants, bits_order, n, args):
    """Build one query row (sequential from bench.err, parallel from parbench.err)."""
    row = label
    for mode in ("query_seq", "query_par"):
        method_name = variants[mode]
        if method_name is None:
            row += " & ---" * len(bits_order)
            continue
        cfg = METHODS.get(method_name)
        if cfg is None:
            row += " &" * len(bits_order)
            continue
        rdir = Path(args.results_dir) / method_name
        tool_for_parse = _parse_tool(cfg["tool"])
        prefix = "parbench" if mode == "query_par" else "bench"
        for bits in bits_order:
            bench_err = rdir / f"{prefix}.{n}.{bits}.err"
            val = ""
            if bench_err.exists():
                d = parse_query(bench_err.read_text(), tool_for_parse)
                v = d.get("query_min")
                if v is not None:
                    val = f"{v:.1f}"
            row += f" & {val}"
    return row + r" \\"


def _table_filter(tbl, args):
    """Render filter tables: one construction table and one query table per size."""
    bits_order = tbl["bits_order"]
    ncols = len(bits_order)
    colspec = "l" + "rrr|rrr" + "||" + "rrr|rrr"

    for n in tbl["sizes"]:
        log2_n = n.bit_length() - 1

        for kind, row_fn in (("Construction", _filter_constr_row),
                             ("Queries", _filter_query_row)):
            print(f"% {kind}, n = 2^{{{log2_n}}}")
            print(r"\begin{tabular}{" + colspec + "}")
            print(r"\toprule")

            print(f"& \\multicolumn{{{ncols}}}{{c}}{{Sequential}}"
                  f" & \\multicolumn{{{ncols}}}{{c}}{{Parallel}} \\\\")

            hdr = ""
            for bits in bits_order:
                hdr += f" & {bits}"
            hdr += hdr
            print(hdr + r" \\")
            print(r"\midrule")

            for label, variants in tbl["rows"]:
                print(row_fn(label, variants, bits_order, n, args))

            print(r"\bottomrule")
            print(r"\end{tabular}")
            print()


def _table_mmphf(tbl, args):
    """Render the 'mmphf' table: sizes as column groups, each with constr/bpk/query."""
    sizes = tbl["sizes"]
    nsizes = len(sizes)
    cols_per_size = 3  # construction, bits/key, query

    # Column spec: label + 3 columns per size, separated by ||
    colspec = "l" + "||".join(["rrr"] * nsizes)
    print(r"\begin{tabular}{" + colspec + "}")
    print(r"\toprule")

    # Header row 1: size groups
    hdr1 = ""
    for n in sizes:
        log2_n = n.bit_length() - 1
        hdr1 += f" & \\multicolumn{{{cols_per_size}}}{{c}}{{$n = 2^{{{log2_n}}}$}}"
    print(hdr1 + r" \\")

    # Header row 2: sub-column labels
    sub_hdr = " & Constr. & Bits/key & Query" * nsizes
    print(sub_hdr + r" \\")
    print(r"\midrule")

    # Data rows
    for method_name, label in tbl["methods"]:
        if method_name not in METHODS:
            continue
        cfg = METHODS[method_name]
        rdir = Path(args.results_dir) / method_name
        tool_for_parse = _parse_tool(cfg["tool"])

        row = label
        for n in sizes:
            bench_err = rdir / f"bench.{n}.0.err"
            size_file = rdir / f"bench.{n}.0.size"

            constr_val = ""
            bpk_val = ""
            query_val = ""

            if bench_err.exists():
                text = bench_err.read_text()
                bd = parse_build(text, tool_for_parse)
                qd = parse_query(text, tool_for_parse)
                v = bd.get("constr_ns")
                if v is not None:
                    constr_val = f"{v:.1f}"
                v = qd.get("query_min")
                if v is not None:
                    query_val = f"{v:.1f}"

            if size_file.exists():
                file_bytes = int(size_file.read_text().strip())
                bpk_val = f"{file_bytes * 8.0 / n:.2f}"

            row += f" & {constr_val} & {bpk_val} & {query_val}"

        row += r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")


def _table_mmphf_str(tbl, args):
    """Render the 'mmphf-str' table: datasets as column groups, constr/bpk/query each."""
    import re
    # Discover datasets from result directories.
    datasets = set()
    for method_name, _ in tbl["methods"]:
        rdir = Path(args.results_dir) / method_name
        if not rdir.exists():
            continue
        for f in rdir.iterdir():
            m = re.match(r'bench\.(.+)\.0\.err$', f.name)
            if m:
                datasets.add(m.group(1))
    datasets = sorted(datasets)
    if not datasets:
        print("No string dataset results found.")
        return

    ndatasets = len(datasets)
    cols_per_dataset = 3  # construction, bits/key, query

    colspec = "l" + "||".join(["rrr"] * ndatasets)
    print(r"\begin{tabular}{" + colspec + "}")
    print(r"\toprule")

    hdr1 = ""
    for d in datasets:
        safe = d.replace("_", r"\_")
        hdr1 += f" & \\multicolumn{{{cols_per_dataset}}}{{c}}{{\\texttt{{{safe}}}}}"
    print(hdr1 + r" \\")

    sub_hdr = " & Constr. & Bits/key & Query" * ndatasets
    print(sub_hdr + r" \\")
    print(r"\midrule")

    for method_name, label in tbl["methods"]:
        if method_name not in METHODS:
            continue
        cfg = METHODS[method_name]
        rdir = Path(args.results_dir) / method_name
        tool_for_parse = _parse_tool(cfg["tool"])

        row = label
        for dataset in datasets:
            bench_err = rdir / f"bench.{dataset}.0.err"
            size_file = rdir / f"bench.{dataset}.0.size"

            constr_val = ""
            bpk_val = ""
            query_val = ""

            n = None
            if bench_err.exists():
                text = bench_err.read_text()
                bd = parse_build(text, tool_for_parse)
                qd = parse_query(text, tool_for_parse)
                n = _parse_n_from_build(text)
                v = bd.get("constr_ns")
                if v is not None:
                    constr_val = f"{v:.1f}"
                v = qd.get("query_min")
                if v is not None:
                    query_val = f"{v:.1f}"

            if size_file.exists() and n:
                file_bytes = int(size_file.read_text().strip())
                bpk_val = f"{file_bytes * 8.0 / n:.2f}"

            row += f" & {constr_val} & {bpk_val} & {query_val}"

        row += r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")


def _table_comp(tbl, args):
    """Render the 'comp' table: distributions as column groups, constr/query each."""
    dist_labels = tbl["dist_labels"]
    ndists = len(dist_labels)

    for n in tbl["sizes"]:
        log2_n = n.bit_length() - 1

        colspec = "l" + "||rrr" * ndists
        print(f"% n = 2^{{{log2_n}}}")
        print(r"\begin{tabular}{" + colspec + "}")
        print(r"\toprule")

        hdr1 = ""
        for i, dl in enumerate(dist_labels):
            sep = "||" if i < ndists - 1 else ""
            hdr1 += f" & \\multicolumn{{3}}{{c{sep}}}{{{dl}}}"
        print(hdr1 + r" \\")

        sub_hdr = (" & Constr. & Query & b/key") * ndists
        print(sub_hdr + r" \\")
        print(r"\midrule")

        for label, method_names in tbl["rows"]:
            row = label
            for method_name in method_names:
                if method_name not in METHODS:
                    row += " & & &"
                    continue
                cfg = METHODS[method_name]
                rdir = Path(args.results_dir) / method_name
                tool_for_parse = _parse_tool(cfg["tool"])

                bench_err = rdir / f"bench.{n}.0.err"
                size_file = rdir / f"bench.{n}.0.size"
                constr_val = ""
                bpk_val = ""
                query_val = ""

                if bench_err.exists():
                    text = bench_err.read_text()
                    bd = parse_build(text, tool_for_parse)
                    qd = parse_query(text, tool_for_parse)
                    v = bd.get("constr_ns")
                    if v is not None:
                        constr_val = f"{v:.1f}"
                    v = qd.get("query_min")
                    if v is not None:
                        query_val = f"{v:.1f}"

                if size_file.exists():
                    file_bytes = int(size_file.read_text().strip())
                    bpk_val = f"{file_bytes * 8.0 / n:.2f}"

                row += f" & {constr_val} & {query_val} & {bpk_val}"

            row += r" \\"
            print(row)

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()


def _table_java(tbl, args):
    """Render the 'java' table: filter query + overhead comparison."""
    bits_order = tbl["bits_order"]
    even = [b for b in bits_order if b % 2 == 0]
    odd = [b for b in bits_order if b % 2 != 0]
    ncols = len(bits_order)
    n_even = len(even)
    n_odd = len(odd)
    colspec = "l" + "r" * n_even + "|" + "r" * n_odd + "r"

    for n in tbl["sizes"]:
        log2_n = n.bit_length() - 1

        print(f"% Java comparison, n = 2^{{{log2_n}}}")
        print(r"\begin{tabular}{" + colspec + "}")
        print(r"\toprule")

        print(f"& \\multicolumn{{{ncols}}}{{c}}{{Query (ns/key)}}"
              r" & {Overhead} \\")

        hdr = ""
        for bits in bits_order:
            hdr += f" & {bits}"
        hdr += " &"
        print(hdr + r" \\")
        print(r"\midrule")

        for label, variants in tbl["rows"]:
            method_name = variants["query"]
            cfg = METHODS.get(method_name)
            if cfg is None:
                continue
            rdir = Path(args.results_dir) / method_name
            tool_for_parse = _parse_tool(cfg["tool"])

            query_cells = ""
            overheads = []
            for bits in bits_order:
                bench_err = rdir / f"bench.{n}.{bits}.err"
                size_file = rdir / f"bench.{n}.{bits}.size"

                qval = ""
                if bench_err.exists():
                    text = bench_err.read_text()
                    qd = parse_query(text, tool_for_parse)
                    v = qd.get("query_min")
                    if v is not None:
                        qval = f"{v:.1f}"

                if size_file.exists():
                    file_bytes = int(size_file.read_text().strip())
                    overheads.append(compute_overhead(file_bytes, n, bits))

                query_cells += f" & {qval}"

            if overheads:
                avg_ovhd = sum(overheads) / len(overheads)
                overhead_cell = f" & $\\approx {avg_ovhd:.1f}\\%$"
            else:
                overhead_cell = " &"

            print(label + query_cells + overhead_cell + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()


def _table_java_comp(tbl, args):
    """Render the 'java-comp' table: compressed function comparison."""
    dist_labels = tbl["dist_labels"]
    ndists = len(dist_labels)

    for n in tbl["sizes"]:
        log2_n = n.bit_length() - 1

        colspec = "l" + "||rrr" * ndists
        print(f"% Java compressed comparison, n = 2^{{{log2_n}}}")
        print(r"\begin{tabular}{" + colspec + "}")
        print(r"\toprule")

        hdr1 = ""
        for i, dl in enumerate(dist_labels):
            sep = "||" if i < ndists - 1 else ""
            hdr1 += f" & \\multicolumn{{3}}{{c{sep}}}{{{dl}}}"
        print(hdr1 + r" \\")

        sub_hdr = (" & Query & b/key & Ovhd\\%") * ndists
        print(sub_hdr + r" \\")
        print(r"\midrule")

        for label, method_names in tbl["rows"]:
            row = label
            for method_name in method_names:
                if method_name not in METHODS:
                    row += " & & &"
                    continue
                cfg = METHODS[method_name]
                rdir = Path(args.results_dir) / method_name
                tool_for_parse = _parse_tool(cfg["tool"])

                bench_err = rdir / f"bench.{n}.0.err"
                size_file = rdir / f"bench.{n}.0.size"
                query_val = ""
                bpk_val = ""
                ovhd_val = ""

                if bench_err.exists():
                    text = bench_err.read_text()
                    qd = parse_query(text, tool_for_parse)
                    v = qd.get("query_min")
                    if v is not None:
                        query_val = f"{v:.1f}"

                if size_file.exists():
                    file_bytes = int(size_file.read_text().strip())
                    bpk_val = f"{file_bytes * 8.0 / n:.2f}"
                    dist = cfg.get("distribution")
                    entropy = distribution_entropy(dist)
                    if entropy is not None:
                        ovhd = compute_overhead(file_bytes, n, entropy)
                        ovhd_val = f"{ovhd:.1f}"

                row += f" & {query_val} & {bpk_val} & {ovhd_val}"

            row += r" \\"
            print(row)

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()


def cmd_table(args):
    """Print LaTeX tables for the paper."""
    if args.table not in TABLES:
        sys.exit(f"Unknown table: {args.table}\n"
                 f"Available: {', '.join(TABLES)}")

    tbl = TABLES[args.table]

    if args.table == "mmphf":
        _table_mmphf(tbl, args)
    elif args.table == "mmphf-str":
        _table_mmphf_str(tbl, args)
    elif args.table == "comp":
        _table_comp(tbl, args)
    elif args.table == "java":
        _table_java(tbl, args)
    elif args.table == "java-comp":
        _table_java_comp(tbl, args)
    else:
        _table_filter(tbl, args)


# =====================================================================
#  CLI
# =====================================================================


def make_parser():
    """Build the argument parser with all subcommands."""
    p = argparse.ArgumentParser(
        description="Benchmark runner for the ε-cost sharding paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Global options ---
    p.add_argument("--sux-dir", default=os.environ.get("SUX_DIR", os.path.join(REPO_ROOT, "sux-rs")),
                   help="Path to sux-rs repository (default: sux-rs/ submodule)")
    p.add_argument("--burr-dir", default=os.environ.get("BURR_DIR", os.path.join(REPO_ROOT, "BuRR")),
                   help="Path to BuRR repository (default: BuRR/ submodule)")
    p.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"),
                   help="Path to results directory")
    p.add_argument("--pin", default=os.environ.get("BENCH_PIN"),
                   help="CPU-pinning prefix, e.g. 'chrt -f 99 taskset -c 2'")
    p.add_argument("--threads", type=int, default=None,
                   help="Override thread count for parallel builds")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if result files already exist")

    sub = p.add_subparsers(dest="command", required=True)

    # Common arguments shared by multiple subcommands.
    def add_common(sp):
        sp.add_argument("--method", default="all",
                        help="Method name(s), comma-separated, or 'all'")
        sp.add_argument("--bits", default="all",
                        help="Bit widths: 'all', 'even', 'odd', or comma-separated ints")
        sp.add_argument("--max-log2", type=int, default=33,
                        help=f"Maximum log2(n) (default: 33, min: {MIN_LOG2})")
        # Allow global flags after the subcommand too.
        sp.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
        sp.add_argument("--force", action="store_true", default=argparse.SUPPRESS)
        sp.add_argument("--pin", default=argparse.SUPPRESS)
        sp.add_argument("--threads", type=int, default=argparse.SUPPRESS)

    # --- bench ---
    b = sub.add_parser("bench", help="Build structures, query, and clean up")
    add_common(b)
    b.add_argument("--keep-data", action="store_true", default=False,
                   help="Keep serialized files after query (for debugging)")
    b.add_argument("--no-keep-data", action="store_false", dest="keep_data",
                   help="Delete serialized files after query (default)")

    # --- par-bench ---
    pq = sub.add_parser("par-bench", help="Build, run parallel queries, clean up")
    add_common(pq)
    pq.add_argument("--jobs", type=int, default=DEFAULT_PAR_JOBS,
                    help=f"Number of parallel instances (default: {DEFAULT_PAR_JOBS})")

    # --- report ---
    r = sub.add_parser("report", help="Print TSV report of results")
    add_common(r)

    # --- plot ---
    pl = sub.add_parser("plot", help="Generate paper figures")
    pl.add_argument("--figure", default="all",
                    help=f"Figure preset, or 'all' (default): {', '.join(FIGURES)}")
    pl.add_argument("--bits", default="all",
                    help="Bit widths for figures")
    pl.add_argument("--max-log2", type=int, default=33)
    pl.add_argument("--output-dir", default="figs",
                    help="Output directory for PDFs (default: figs)")

    # --- table ---
    tb = sub.add_parser("table", help="Print LaTeX tables")
    tb.add_argument("--table", default="filter",
                    help=f"Table preset (default: filter). Available: {', '.join(TABLES)}")
    tb.add_argument("--results-dir", default=argparse.SUPPRESS)

    # --- status ---
    st = sub.add_parser("status", help="Show completed/pending benchmark runs")
    add_common(st)

    # --- clean ---
    cl = sub.add_parser("clean", help="Delete results and/or serialized data")
    cl.add_argument("--method", default="all",
                    help="Method name(s), comma-separated, or 'all'")
    cl.add_argument("--data-only", action="store_true",
                    help="Only delete serialized structure files, keep .err results")

    return p


def main():
    p = make_parser()
    args = p.parse_args()

    dispatch = {
        "bench": cmd_bench,
        "par-bench": cmd_par_bench,
        "report": cmd_report,
        "plot": cmd_plot,
        "table": cmd_table,
        "status": cmd_status,
        "clean": cmd_clean,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
