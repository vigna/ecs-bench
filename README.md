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
- `java/`: Maven project for the Sux4J GOV3 build path.
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

The C++ benches require a C++20 toolchain; we use `-fno-strict-aliasing`
to keep GCC 15+ from miscompiling BuRR under `-O3`.

Rust (stable) must be installed. `make init` then pulls submodules and
applies the LeMonHashVL string-view fairness patches (they are necessary to
speed up the LeMonHashVL benchmark by avoiding cache misses when retrieving
the strings).

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

## The workflow

First, clone the repository and prepare data:

```sh
git clone https://github.com/vigna/ecs-bench.git
cd ecs-bench
make init
curl https://vigna.di.unimi.it/data.tar.zst | zstd -d | tar xv
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

2. **A small smoke run** at one bit width and small sizes, to confirm that
   every binary compiles, runs, and produces parseable output:

   ```sh
   python bench/run.py bench --max-log2 22
   ```

3. **Extend gradually.** Each successive call only executes the (method, n,
   bits) triples whose result files don't already exist.

   ```sh
   python bench/run.py bench --max-log2 25
   ```

4. For the parallel-query column of the paper:

   ```sh
   python bench/run.py par-bench --max-log2 30 --jobs 96
   ```

   `--jobs` defaults to `cpu_count()//2` if omitted. `par-bench` will
   also automatically limit the maximum size based on the physical RAM.

5. Render the figures and tables once the matrix is populated:

   ```sh
   python bench/run.py plot --figure all
   python bench/run.py table --table java
   ```

### CPU pinning

For more stable measurements, pin single-threaded phases to an isolated core:

```sh
python bench/run.py bench --pin "taskset -c 2"
```

The `--pin` prefix is applied to query commands and to single-threaded
builders; multi-threaded builds run unpinned so that parallel construction is
not artificially serialised.

## Result layout

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
