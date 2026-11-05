# ecs-bench

This is the companion repository for the paper

> Sebastiano Vigna.
> [_ε-Cost Sharding: Scaling Hypergraph-Based Data Structures to Trillions of Keys_](https://arxiv.org/abs/2503.18397), 2026.

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
  defines the method matrix and the figure presets; `parse.py` extracts numbers
  from build/query output.
- `Sux4J/`, `BuRR/`, `BuRR-VL/`, `LearnedStaticFunction/`, `MMPHF-Experiments/`:
  pinned submodules of competing implementations.
- `sux-rs/`: pinned submodule of `sux`.
- `java/`: Maven project (with the Maven wrapper `mvnw`) for the Sux4J GOV3
  build path (one of the two HEM baselines; see below). Sux4J itself comes from
  Maven Central (see `pom.xml`): only the C query code, `Sux4J/c/`, comes from
  the submodule.
- `hem-vfilter/`: a HEM-sharded clone of VFilter, the paper's sharding
  ablation (see below).
- `multi-vfilter/`: BuRR's parallel query benchmark (`parbench.cpp`) ported to
  Rust for VFilter, used by `multi-bench`.
- `phastr-array/`: a static function made of a PHastR minimal perfect hash
  function and an array of values, which the paper compares with VFunc.
- `bench_csf.cpp`, `bench_lemonhash.cpp`, `bench_lemonhash_str.cpp`: C++
  benches for BuRR-CSF (compressed static functions based on BuRR-VL and on the
  codes of `LearnedStaticFunction`) and for LeMonHash on integers and on
  strings.
- `Makefile`: builds `bench_csf`, `bench_lemonhash`, and `bench_lemonhash_str`.
  Its `init` target pulls submodules and applies the fairness patches in
  `lemonhash-vl-string-view.patch` and `murmurhash-string-view.patch`. The BuRR
  binaries (`bench`, `benchpos`, and `parbench`, one for each bit width) are
  built by `run.py` through BuRR's own Makefile.
- `CMakeLists.txt`: the CMake project the `Makefile` uses to build the
  LeMonHash benches against the LeMonHash submodule of `MMPHF-Experiments`.
- `matplotlibrc`: the style of the figures, which `run.py plot` loads whatever
  the current directory.

## Dependencies

The build was developed and tested on several different Linux distributions. The
dependencies are:

- `g++`, `boost-devel`, `tbb-devel`, `xxhash-devel`, `clang-devel`: C++
  toolchain plus headers needed by the BuRR / BuRR-VL / sdsl-lite build paths.
- `cmake`, `ninja-build`, `make`, `git`.
- A JDK (Java 11 or later), for the Sux4J GOV3 build path. Maven need not be
  installed: `java/mvnw` downloads it on first use, which requires `curl` and
  `unzip`.
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
data. The string benchmarks (`LcpMmphfStr`, `Lcp2MmphfStr`, `LeMonHashStr`)
read sorted-string files from `data/` (in the repository root, whatever the
current directory). Pre-built sorted datasets used in the paper can be
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

The submodules are fetched over HTTPS, so no SSH key is needed. If you push to
these repositories, you can make git use SSH for them with

```sh
git config --global url.git@github.com:.insteadOf https://github.com/
```

Benchmarks are run by `bench/run.py`, which has several subcommands:

- `bench` — build each (method, n, bits) triple, run queries, write a result
  file under `results/<method>/` (by default).
- `multi-bench` — the highly parallel query benchmark of the BuRR paper: for
  each size *n*, build 2³³/*n* structures of *n* keys and query them at random
  with all hardware threads, on keys and on non-keys.
- `report` — print a space-aligned report of all completed `bench` runs.
- `plot` — produce paper figures (PDF) under `figs/` (by default).
- `table` — print LaTeX tables.
- `status` — list completed and pending `bench` runs.
- `clean` — delete results and/or serialized data.

`report` and `status` cover only the results of `bench`; `multi-bench
--report` prints a TSV table of the results of `multi-bench`.

Results are stored **per data point** as individual files under
`results/<method>/bench.<n>.<bits>.err`. The framework treats any existing,
parseable result file as authoritative: **interrupted runs resume from where
they stopped**, and successive `bench` invocations only execute the data
points that aren't already complete. Pass `--force` to override.

All binaries are built on demand by `run.py`: the sux-rs binaries and the
`hem-vfilter`, `multi-vfilter`, and `phastr-array` crates (compiled with `cargo
build --profile bench`), the Java `BenchBuilder` (via the Maven wrapper), the
BuRR binaries (via BuRR's Makefile), `bench_csf`, `bench_lemonhash`, and
`bench_lemonhash_str` (via the top-level `Makefile`), and the Sux4J C query
code (with `cc`, or `$CC`).

### Paths and environment variables

By default, `run.py` uses the `sux-rs/` and `BuRR/` submodules; `--sux-dir`
and `--burr-dir` select other checkouts. `--sux-dir` does not affect
`hem-vfilter`, `multi-vfilter`, and `phastr-array`, which depend on `sux`
through `path = "../sux-rs"` in their `Cargo.toml`, that is, always on the
submodule.

`--results-dir` (default: `results`) and the output directory of `plot`
(`--output-dir`, default: `figs`) are relative to the current directory, not
to the repository root.

The defaults of some global options can be set with environment variables:

- `BENCH_PIN`: default of `--pin`;
- `SUX_DIR`: default of `--sux-dir`;
- `BURR_DIR`: default of `--burr-dir`;
- `RESULTS_DIR`: default of `--results-dir`.

## Recommended first-time workflow

The full campaign takes hours to days. Treat it as additive: the framework
never recomputes a data point that already has a result file, so it is safe
(and cheap) to extend the matrix gradually.

1. **Dry-run first**, to see exactly what commands and parameter combinations
   would be executed without running (or building) anything; data points that
   already have a result are skipped, unless you add `--force`:

   ```sh
   python3 bench/run.py bench --dry-run
   ```

2. **A small smoke run** of a single method at small sizes, to confirm that the
   binaries compile, run, and produce parseable output:

   ```sh
   python3 bench/run.py bench --method FuseAug3Shards --bits 8 --max-log2 22
   ```

   `--method` accepts a comma-separated list of methods. Note that `--max-log2`
   does not apply to the string methods (`LcpMmphfStr`, `Lcp2MmphfStr`,
   `LeMonHashStr`), which always run on every file in `data/`: the largest
   dataset of the paper contains 1.07·10⁹ URLs, so a run with no `--method`
   is not a smoke run, whatever `--max-log2`.

3. **Extend gradually.** Each successive call only executes benchmarks whose
   result files don't already exist (again, the string methods run on all of
   `data/`):

   ```sh
   python3 bench/run.py bench --max-log2 25
   ```

4. For the parallel-query figure and the parallel query columns of the filter
   table:

   ```sh
   python3 bench/run.py multi-bench
   ```

   By default it runs `FuseAug3Shards` (with BuRR's own benchmark ported to
   Rust in `multi-vfilter/`) and `BuRR` (with BuRR's `parbench.cpp`) at all
   output widths, from 2²⁰ to 2³³ keys, building the structures and querying
   them with as many threads as there are CPUs (the global `--pin` and
   `--threads` are ignored; use `--build-threads` and `--query-threads`). All
   structures are in memory at the same time, so `--total-log2` makes the total
   number of keys smaller on smaller machines. However, figures and tables use
   only the results of the default sweep (`--total-log2 33`), only BuRR
   results with the default `--burr-eps bench`, and, if a configuration has
   been run several times, the run with the most query threads: with other
   sweeps the parallel-query figure and columns come out empty.

5. Render the figures and the five tables of the paper once the matrix is
   populated:

   ```sh
   python3 bench/run.py plot
   for t in filter mmphf mmphf-str comp java-comp; do
       python3 bench/run.py table --table $t
   done
   ```

   The script `regen.sh` of the ecs repository (see below) runs these commands
   on the results of the paper. A further table, `java`, is not used by the
   paper.

Explore the help of `run.py` for more fine-grained control, e.g. to run only a
subset of methods or sizes, or to keep the transient serialized data (which is
deleted by default to save space).

## Reproducing the paper

The results of the paper were computed on an AWS `m8a.metal-24xl` instance (96
cores) by

```sh
python3 bench/run.py bench --pin "taskset -c 2" --threads 32
python3 bench/run.py multi-bench
```

`--threads 32` is necessary because the default number of construction
threads is half the number of CPUs, which would be 48 on that machine.
`multi-bench` runs with its defaults: `FuseAug3Shards` and `BuRR`, all six bit
widths, 2³³/*n* structures of *n* keys for *n* from 2²⁰ to 2³³, 10⁸ queries per
thread, and as many construction and query threads as there are CPUs.

The LaTeX source of the paper lives in a separate repository, `ecs`, which
contains also:

- `aws.sh`, which sets up the machine: it installs the system dependencies,
  Java, Maven, and Rust, builds GCC 16.2.0, clones this repository and runs
  `make init`, downloads the datasets, and fixes the CPU frequency with
  `cpupower`; its last (commented) line is the campaign above;
- `results-epyc`, the results used in the paper;
- `regen.sh`, which regenerates all figures and tables of the paper from
  `results-epyc` using `run.py plot` and `run.py table`.

To inspect the results, use `report` (or `status`) for the results of
`bench`, and `multi-bench --report` for those of `multi-bench`. The latter
shows only the runs matching its options, whose thread counts default to the
number of CPUs of the current machine, so on another machine you must pass
those of the paper; for example, from the root of this repository, with the
ecs repository next to it,

```sh
python3 bench/run.py --results-dir ../ecs/results-epyc report
python3 bench/run.py --results-dir ../ecs/results-epyc multi-bench --report \
    --query-threads 96 --build-threads 96
```

### The offline-construction table

The table of the paper on offline construction (Table 2, `tab:offline`), with
2-bit VFilter instances of 10¹⁰, 10¹¹, and 10¹² keys, is not produced by the
harness: it was produced by hand with the `vfilter` command-line utility and
the `bench_vfilter` example of `sux` 0.15.3 (equivalent for this purpose to the
0.15.4 described by the paper; rustc 1.99.1), in November 2026, on an AWS
`m8a.metal-24xl` instance with a 24 000 GiB EBS gp3 volume (2 000 MiB/s,
16 000 IOPS). For each number of keys `n` we ran

```sh
/usr/bin/time -v vfilter -n $n -b 2 -u -s -o --temp-dir ~/tk -t 32 ~/tk/f$n.bin
taskset -c 2 bench_vfilter 100000000 ~/tk/f$n.bin -b 2 -u
```

The first command hashes the keys sequentially (`-s`) into disk-based buckets
(`-o`) in `--temp-dir`, solves the shards with 32 threads, and saves the filter
in unaligned form (`-u`); the second one performs five repetitions of 10⁸
queries on a pinned core.

The logs are in the directory `results-offline` of the ecs repository:
`build.<n>.2.err` is the standard error of the first command (including the
output of `time`), and `query.<n>.2.err` that of the second one. The columns of
the table are derived from them as follows, dividing times by the number of
keys:

- shards: the line `Fuse3 (augmented peeling, shards); shards: …`;
- hashing: the time elapsed until the line `Max shard / average shard`;
- construction: the time of the line `Construction completed` minus the
  hashing time;
- query: the minimum of the five repetitions (`Min:`).

The peak memory in the caption is the maximum resident set size reported by
`time`.

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
python3 bench/run.py bench --method HemVFilter,FuseAug3Shards --bits 8
```

The crate does not implement augmented peeling, which `VFilter` uses up to
2·10⁷ keys. There it peels the same shards with the expansion factor and
segment-size law of plain peeling, so its graphs are 1.8–4.2% larger between
2²⁰ and 2²⁴ keys, and it refuses fewer than a million keys. It also keeps all
signatures in core, where `VFilter` streams them through a disk-backed store,
so it needs about 32 bytes per key while sharding: 2³³ keys still fit on a
384 GiB machine. Finally, it remixes the local seed with one multiplication,
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

`HemVFilter` appears in the single-threaded query figures and in the filter
table. It does not appear in the construction figures: its plain peeling loop
and missing augmented peeling would make its build time differ from `VFilter`'s
for reasons that have nothing to do with sharding. GOV3 reports no construction
time either, because Java builds it.

### CPU pinning

For more stable measurements, pin single-threaded phases to an isolated core:

```sh
python3 bench/run.py bench --pin "taskset -c 2"
```

The `--pin` prefix is applied to:

- the query commands of all methods, except `VFunc` and `VFuncFullSigs`,
  which are compared with `PHastR`, whose construction and queries run in a
  single multi-threaded process, and are therefore queried unpinned too;
- the builds of the LCP-based monotone minimal perfect hash functions
  (`lcp_mmphf`, which runs single-threaded);
- the whole runs of `LeMonHashInt` and `LeMonHashStr`, which build and query
  in a single process.

All other builds run unpinned, including single-threaded ones such as those of
`FuseAug3Shards-seq`, `BuRR-seq`, or `CompVFunc`, so that parallel
construction is never artificially serialised. `PHastR` and `multi-bench`
always run unpinned.

## Results layout

```
results/
  <method>/
    bench.<n>.<bits>.err        stdout/stderr of one build+query
    bench.<dataset>.0.err       the same, for a string dataset of data/
    multibench.<n>.<bits>.<k>.<t>.<queries>.<build threads>[.eps-<bench|parbench>].out
                                one multi-bench run (k structures of n keys,
                                t query threads; the suffix, only for BuRR,
                                records --burr-eps)
    multibench.<...>.failed     output of a failed multi-bench run (the name
                                of its .out file, with suffix .failed; a
                                stale .out file is deleted)
    data/                       transient serialized structures
                                (deleted unless --keep-data)
```

`<bits>` is 0 for bits-agnostic methods (compressed functions, `VFunc`,
`VFuncFullSigs`, `PHastR`, and monotone minimal perfect hash functions), for
which bits/key is a result rather than a parameter.

When building and querying are separate steps, a `.err` file contains the
build output followed, after a `--- query ---` line, by the query output. The
builder self-reports the space used by the structure, which is the canonical
space number used by the paper: sux-rs, `hem-vfilter`, `phastr-array`, and
Sux4J log a `Bits/key: X.XXX` line; `bench_csf`, `bench_lemonhash`, and
`bench_lemonhash_str` log a `Size: N Bytes = X bits/key` line; and BuRR logs a
`Ribbon size: N Bytes = …` line, from which the framework computes bits/key
using the number of keys. When the structure is serialized, the framework
cross-checks this number against the on-disk file size and appends a `WARNING:`
line if the two disagree by more than 1%.

## License

The files of this repository are dedicated to the public domain (see
`LICENSE`); the submodules are distributed under their own licenses.
