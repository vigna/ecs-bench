# ecs-bench

This is the companion repository for the paper

> Sebastiano Vigna.
> [_ε-Cost Sharding: Scaling Hypergraph-Based Static Functions and Filters to Trillions of Keys_](https://arxiv.org/abs/2503.18397), 2025.

The paper introduces a partitioning scheme that lets MWHC and fuse-graph
constructions of static functions and filters scale to trillions of keys with
only a small (hence, ε), controllable relative space overhead. The technique is
then extended to monotone minimal perfect hash functions based on prefixes, and
to compressed static functions.

This repository contains the Python framework
that drives all benchmarks, the figure/table generators, and pinned submodules
of every competing implementation we compare against. The actual implementations
live in the [`sux`](https://github.com/vigna/sux-rs) Rust crate.

## Contents

- `bench/`: Python framework. `run.py` is the single entry point; `configs.py`
  defines the method matrix; `parse.py` extracts numbers from build/query
  output.
- `Sux4J/`, `BuRR/`, `BuRR-VL/`, `LearnedStaticFunction/`, `MMPHF-Experiments/`:
  pinned submodules of competing implementations.
- `sux-rs/`: pinned submodule of `sux`.
- `java/`: Maven project for the Sux4J GOV3 build path (one of the two HEM
  baselines; see below).
- `hem-vfilter/`: a HEM-sharded clone of VFilter, the paper's sharding
  ablation (see below).
- `Makefile`: builds the C++ benches (BuRR, `bench_csf`, `bench_lemonhash`,
  `bench_lemonhash_str`). Its `init` target pulls submodules and applies the
  fairness patches in `lemonhash-vl-string-view.patch` and
  `murmurhash-string-view.patch`.

## Dependencies

The build was developed and tested on several different Linux distributions. The
dependencies are:

- `g++`, `boost-devel`, `tbb-devel`, `xxhash-devel`, `clang-devel`: C++
  toolchain plus headers needed by the BuRR / BuRR-VL / sdsl-lite build paths.
- `cmake`, `ninja-build`, `make`, `git`, `parallel` (GNU parallel; needed by
  `par-bench`).
- A JDK and Maven, for the Sux4J GOV3 build path.
- Python 3.9+, with `pip install matplotlib` (only needed for `plot`).
- `zstd`, to decompress the string data.
- `patch`, to apply the fairness patches to LeMonHash.

The C++ benches require a C++20 toolchain; we use `-fno-strict-aliasing` to
[keep GCC 15+ from miscompiling BuRR under
`-O3`](https://github.com/lorenzhs/BuRR/issues/4).

Rust (stable) must be installed. `make init` then pulls submodules and applies
the LeMonHashVL string-view fairness patches (they are necessary to speed up the
LeMonHashVL benchmark by avoiding cache misses or copies when accessing the
strings).

## Datasets

The integer benchmarks generate keys deterministically and need no external
data. The string benchmarks (`LcpMmphfStr`, `LeMonHashStr`) read sorted-string
files from `data/`. Pre-built sorted datasets used in the paper can be
downloaded with

```sh
curl https://vigna.di.unimi.it/data.tar.zst | zstd -d | tar xv
```

The framework picks them up automatically (one structure built and timed per
file, so you can delete files that are too big for your hardware, or add your
own).

## Installation

First, clone the repository and prepare the data:

```sh
git clone https://github.com/vigna/ecs-bench.git
cd ecs-bench
curl https://vigna.di.unimi.it/data.tar.zst | zstd -d | tar xv
make init
```

Benchmarks are run by `bench/run.py`, which has several subcommands:

- `bench` — build each (method, n, bits) triple, run queries, write a result
  file under `results/<method>/` (by default).
- `par-bench` — build once, run `--jobs` parallel query instances via GNU
  parallel, write the aggregate.
- `report` — print a TSV report of all completed runs.
- `plot` — produce paper figures (PDF) under `figs/` (by default).
- `table` — print LaTeX tables.
- `status` — list completed and pending runs.
- `clean` — delete results and/or serialized data.

Results are stored **per data point** as individual files under
`results/<method>/bench.<n>.<bits>.err`. The framework treats any existing,
parseable result file as authoritative: **interrupted runs resume from where
they stopped**, and successive `bench` invocations only execute the data
points that aren't already complete. Pass `--force` to override.

All binaries are built on demand by `run.py`: the sux-rs binaries (compiled
with `cargo build --profile bench`), the Java `BenchBuilder` (via Maven), and
the C++ benches (BuRR, `bench_csf`, `bench_lemonhash`, `bench_lemonhash_str`,
and the Sux4J/C harness).

## Recommended first-time workflow

The full campaign takes hours to days. Treat it as additive: the framework
never recomputes a data point that already has a result file, so it is safe
(and cheap) to extend the matrix gradually.

1. **Dry-run first**, to see exactly what commands and parameter combinations
   would be executed without running anything:

   ```sh
   python bench/run.py bench --dry-run
   ```

2. **A small smoke run** at small sizes, to confirm that every binary compiles,
   runs, and produces parseable output:

   ```sh
   python bench/run.py bench --max-log2 22
   ```

3. **Extend gradually.** Each successive call only executes benchmarks whose
   result files don't already exist.

   ```sh
   python bench/run.py bench --max-log2 25
   ```

4. For the parallel-query column of the paper:

   ```sh
   python bench/run.py par-bench --threads 32
   ```

   `--jobs` sets the number of instances, and defaults to 32 (the number
   used in the paper), or to the number of CPUs if smaller. `par-bench` will
   also automatically limit the maximum size based on the physical RAM.

5. Render the figures and tables once the matrix is populated:

   ```sh
   python bench/run.py plot --figure all
   python bench/run.py table --table java
   ```

Explore the help of `run.py` for more fine-grained control, e.g. to run only a
subset of methods or sizes, or to keep the transient serialized data (which is
deleted by default to save space).

### The HEM baselines

The paper claims that ε-cost sharding removes an offset and a local seed from
the query path, and with them a memory dependency. Two methods measure that
claim.

**`HemVFilter`** is the ablation. The `hem-vfilter/` crate builds a filter that
matches `FuseAug3Shards` in every part but the layout: same signatures, same
shard selection, same fuse-graph edges, same value backends, same number of
shards, and beyond 2·10⁷ keys the same expansion factor and segment-size law.
Only the layout differs. Shards keep their natural, unequal sizes, so the builder stores their
start positions in an array, and gives each shard a local seed. A query must
load that array before it can build its edge. When a shard peels at the first
try its seed is zero, and its edges and hashes are those of `VFilter`, bit for
bit.

The crate lives here and not in `sux`, because HEM is the scheme the paper
replaces. Its query benchmark follows the protocol of `bench_vfilter`, and the
harness drives it like any sux-rs method:

```sh
python bench/run.py bench --method HemVFilter,FuseAug3Shards --bits 8
```

The crate does not implement augmented peeling, which `VFilter` uses up to
2·10⁷ keys. There it peels the same shards with the expansion factor and
segment-size law of plain peeling, so its graphs are 1.8–4.2% larger between
2²⁰ and 2²⁴ keys, and it refuses fewer than a million keys. It also keeps all
signatures in core, where `VFilter` streams them through a disk-backed store,
so it stops at 2³² keys. It also remixes the local seed with one multiplication,
where Sux4J re-hashes the whole signature. The measured gap is therefore a
lower bound. `hem-vfilter/src/lib.rs` explains these points, and why a cheaper
remix fails.

To check a build instead of timing it:

```sh
hem-vfilter/target/release/hem_vfilter -n 4000000 --bits 8 /tmp/f.bin
hem-vfilter/target/release/bench_hem_vfilter 4000000 /tmp/f.bin --check
```

The filter must find every key, and the false-positive rate must match 2⁻ᵇ.

**`GOV3Function`** is HEM as shipped. `java/` builds it, and the Sux4J C code
queries it. GOV3 solves each shard by lazy Gaussian elimination, which is cubic
and forces buckets of about 1500 keys (`GOV3Function.BUCKET_SIZE`). At 2³⁰ keys
that gives some 716 000 buckets and an offset array of several megabytes,
against the few dozen shards `VFilter` picks. The solver and the space also
differ, so this row is not an ablation: it shows what HEM costs in practice.

`HemVFilter` appears in the query figures and in the filter table, and answers
to `par-bench` like any sux-rs method. It does not appear in the construction
figures: its plain peeling loop and missing augmented peeling would make its
build time differ from `VFilter`'s for reasons that have nothing to do with
sharding. GOV3 reports no construction time either, because Java builds
it.

### CPU pinning

For more stable measurements, pin single-threaded phases to an isolated core:

```sh
python bench/run.py bench --pin "taskset -c 2"
```

The `--pin` prefix is applied to query commands and to single-threaded
builders; multi-threaded builds run unpinned so that parallel construction is
not artificially serialised.

## Results layout

```
results/
  <method>/
    bench.<n>.<bits>.err        stdout/stderr of one build+query
    parbench.<n>.<bits>.err     aggregate of one par-bench run
    data/                       transient serialized structures
                                (deleted unless --keep-data)
```

Each `.err` contains a self-reported `Bits/key: X.XXX` line, which is the
canonical space number used by the paper. The framework cross-checks this
against the on-disk file size and appends a `WARNING:` line if the two disagree
by more than 1%.
