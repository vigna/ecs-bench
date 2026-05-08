# Sux4J Java Benchmark Comparison

**Date:** 2026-05-08
**Status:** Approved

## Goal

Add a `java` target to `bench/run.py` that compares query performance and
space overhead of the old Sux4J (Java/C) implementation against the current
sux-rs (Rust) implementation.  No construction-time comparison (the
implementations are too different).  Results feed into a LaTeX comparison
table at 2^20 and 2^30 keys.

## Pipeline Overview

Three-phase pipeline for each (n, bits) data point:

1. **Java build + dump:** A Maven project in `ecs-bench/java/` pulls Sux4J
   from Maven Central.  A main class creates a GOV3Function (filter mode)
   or GOV3CompressedFunction over n keys, then calls `dump()` to write the
   C-friendly binary.

2. **C query benchmark:** Compiled from the Sux4J submodule's `c/`
   directory.  Loads the dump, runs timed queries, prints results in
   sux-rs-compatible format (`Min: X Median: X Max: X Average: X` in
   ns/key).

3. **Result storage:** Same convention as all other methods:
   `results/<Method>/bench.<n>.<bits>.err` and `.size`.

## Benchmark Families

### A. Filters (GOV3Function in dictionary mode)

- **Java side:** GOV3Function with `dictionary(bits)` over n keys.
  Keys are uint64 integers (0..n-1) encoded as 8-byte **little-endian**
  byte arrays (to match the C `sf3_get_uint64_t` path, which hashes the
  native LE representation via SpookyHash).
- **C side:** `sf3_get_uint64_t` with `#ifdef FILTER` returning
  `((result ^ signature[0]) & mask) == 0`.
- **Bit widths:** 8, 16, 32, 9, 17, 33 (same as main filter benchmarks).
- **Sizes:** 2^20, 2^30.
- **Method names:** `GOV3-8`, `GOV3-16`, etc. (or a single `GOV3Function`
  method with bits parameter, matching the existing convention).

### B. Compressed Functions (GOV3CompressedFunction)

- **Java side:** GOV3CompressedFunction over n LE-encoded uint64 keys with
  values drawn from:
  - Geometric (trailing zeros of random u64)
  - Zipf(2000) (exponent 1, inverse CDF sampling)
  - Uniform(256)
- **C side:** `csf3_get_uint64_t` (no modification needed).
- **Sizes:** 2^20, 2^28 (matching CompVFunc benchmarks).
- **Method names:** `GOV3Compressed-Geo`, `GOV3Compressed-Zipf`,
  `GOV3Compressed-Uni256`.

## Components

### 1. Sux4J Submodule (C code changes only)

Added as a git submodule.  Only the `c/` directory is used.  Changes to
the C code (maintained upstream in Sux4J):

**`sf3.c` — `#ifdef FILTER` extension:**

In `sf3_get_uint64_t` (and `sf3_get_byte_array`, `sf3_get_signature`),
wrap the return with:

```c
#ifdef FILTER
    const uint64_t mask = (UINT64_C(1) << sf->width) - 1;
    return ((result ^ signature[0]) & mask) == 0;
#else
    return result;
#endif
```

Where `result` is the existing XOR of the three looked-up values, and
`signature` is the SpookyHash output (already computed in the function).

**`sf3.c` — `SF_16` and `SF_32` optimizations:**

Extend the existing `#ifdef SF_8` pattern for power-of-2 widths:

```c
#ifdef SF_8
    const uint8_t *p = (uint8_t *)sf->array + bucket_offset;
    return p[e[0]] ^ p[e[1]] ^ p[e[2]];
#elif defined(SF_16)
    const uint16_t *p = (uint16_t *)sf->array + bucket_offset;
    return p[e[0]] ^ p[e[1]] ^ p[e[2]];
#elif defined(SF_32)
    const uint32_t *p = (uint32_t *)sf->array + bucket_offset;
    return p[e[0]] ^ p[e[1]] ^ p[e[2]];
#else
    return get_value(sf->array, e[0] + bucket_offset, sf->width)
         ^ get_value(sf->array, e[1] + bucket_offset, sf->width)
         ^ get_value(sf->array, e[2] + bucket_offset, sf->width);
#endif
```

This exploits the fact that power-of-2 widths dividing 64 are naturally
aligned in the packed bit array, and little-endian byte order makes the
cast valid.  Odd widths (9, 17, 33) fall through to `get_value()`.

**Test harness output format:**

Adapt the existing `test_sf3_uint64_t.c` / `test_csf3_uint64_t.c` pattern
to print all four statistics in sux-rs format:

```
Min: 12.1 Median: 13.5 Max: 15.2 Average: 13.8
```

instead of the current `Median: X.XXXs; X.XXX ns/key`.

### 2. Java Builder (`ecs-bench/java/`)

A Maven project that pulls Sux4J (and dependencies like `dsiutils`,
`fastutil`, etc.) from Maven Central.

**Main class** accepts CLI arguments:

```
java -cp ... BenchBuilder gov3 <n> <bits> <output-file>
java -cp ... BenchBuilder gov3compressed <n> --geometric <output-file>
java -cp ... BenchBuilder gov3compressed <n> --zipf 2000 <output-file>
java -cp ... BenchBuilder gov3compressed <n> --uniform 256 <output-file>
```

Key generation:
- Generates keys 0, 1, ..., n-1 as 8-byte little-endian byte arrays
  (using `Long.reverseBytes()` or a ByteBuffer with LE order).
- For filters: builds `GOV3Function` with `.dictionary(bits)`.
- For compressed: builds `GOV3CompressedFunction` with values from the
  specified distribution.

Value generation for compressed functions (ported from sux-rs
`comp_vfunc.rs`):
- **Geometric:** `Long.numberOfTrailingZeros(rng.nextLong())`
- **Zipf(K):** inverse CDF sampling with exponent 1 over K symbols
- **Uniform(K):** `rng.nextLong() % K` (unsigned)

The PRNG seed does not need to match sux-rs (we measure query performance,
not correctness against the Rust structure).

Calls `dump(DataOutputStream)` to write the C-friendly binary file.

### 3. C Compilation (in `run.py`)

`run.py` compiles the test harness from `Sux4J/c/`:

| Benchmark         | Compile flags                    |
|-------------------|----------------------------------|
| Filter, 8-bit     | `-DSF_8 -DFILTER`                |
| Filter, 16-bit    | `-DSF_16 -DFILTER`               |
| Filter, 32-bit    | `-DSF_32 -DFILTER`               |
| Filter, 9/17/33   | `-DFILTER`                       |
| Compressed func   | (no extra flags)                 |

Binary names follow a convention like `test_sf3_filter8`,
`test_sf3_filter16`, `test_csf3_uint64`, etc.

### 4. `java` Subcommand in `run.py`

New `cmd_java(args)` function:

1. **Ensure Java built:** Run `mvn -q package` in `java/` if the jar
   doesn't exist.
2. **Ensure C harnesses compiled:** Compile the needed test binaries from
   `Sux4J/c/` with appropriate flags.
3. **For each (n, bits) in the work list:**
   a. Skip if result file exists (resume support).
   b. Run Java builder to create the dump file.
   c. Record dump file size in `.size`.
   d. Run C query harness (pinned) on the dump.
   e. Store combined output in `.err`.
   f. Delete dump file (unless `--keep-data`).

CLI:
```
python bench/run.py java --method GOV3Function --bits 8 --max-log2 30
python bench/run.py java --method all  # runs all GOV3 methods
```

### 5. Method Definitions in `configs.py`

New entries in `METHODS`:

```python
# Filters
"GOV3Function": {
    "tool": "java",
    "label": "GOV3 (Sux4J)",
    "java_type": "gov3",
    "max_log2": 30,
},

# Compressed functions
"GOV3Compressed-Geo": {
    "tool": "java",
    "label": "GOV3Comp(geo)",
    "java_type": "gov3compressed",
    "distribution": ["--geometric"],
    "max_log2": 28,
},
"GOV3Compressed-Zipf": {
    "tool": "java",
    "label": "GOV3Comp(zipf)",
    "java_type": "gov3compressed",
    "distribution": ["--zipf", "2000"],
    "max_log2": 28,
},
"GOV3Compressed-Uni256": {
    "tool": "java",
    "label": "GOV3Comp(uni256)",
    "java_type": "gov3compressed",
    "distribution": ["--uniform", "256"],
    "max_log2": 28,
},
```

### 6. `java` Table Preset

New entry in `TABLES`:

```python
"java": {
    "rows": [
        ("VFilter", "FuseLge3Shards"),
        ("GOV3", "GOV3Function"),
    ],
    "bits_order": [8, 16, 32, 9, 17, 33],
    "sizes": [1 << 20, 1 << 30],
    # Columns: query median (ns/key), overhead (%)
},
"java-comp": {
    "rows": [
        ("CompVFunc", ["CompVFunc-Geo", "CompVFunc-Zipf", "CompVFunc-Uni256"]),
        ("GOV3Comp", ["GOV3Compressed-Geo", "GOV3Compressed-Zipf", "GOV3Compressed-Uni256"]),
    ],
    "dist_labels": ["Geometric", "Zipf", "Uniform"],
    "sizes": [1 << 20, 1 << 28],
},
```

### 7. Parser Extension

Add `"java"` to the `_PARSE_TOOL_MAP` so results are parsed with the
`"sux"` parser (since the C harness will emit sux-rs-compatible output):

```python
_PARSE_TOOL_MAP = {"csf": "burr", "comp-vfunc": "sux", "phast": "sux",
                   "mmphf": "sux", "lemonhash": "sux", "java": "sux"}
```

## File Layout

```
ecs-bench/
  Sux4J/                    # git submodule (C code only)
  java/
    pom.xml                 # Maven project pulling Sux4J from Central
    src/main/java/
      BenchBuilder.java     # main class for building + dumping
  bench/
    run.py                  # + cmd_java, ensure_java_built, ensure_sux4j_c_built
    configs.py              # + GOV3 method definitions, java table preset
    parse.py                # + "java" in parse tool map
  results/
    GOV3Function/           # filter results
    GOV3Compressed-Geo/     # compressed function results
    ...
```

## What Changes Where

| Component          | Repository | Changes                               |
|--------------------|-----------|---------------------------------------|
| `sf3.c`            | Sux4J     | `#ifdef FILTER`, `SF_16`/`SF_32`      |
| Test harness output| Sux4J     | Emit Min/Median/Max/Average format    |
| `java/`            | ecs-bench | New Maven project + BenchBuilder.java |
| `bench/run.py`     | ecs-bench | `java` subcommand + C build helpers   |
| `bench/configs.py` | ecs-bench | GOV3 method defs + table presets      |
| `bench/parse.py`   | ecs-bench | `"java": "sux"` in parse tool map     |
| `.gitmodules`      | ecs-bench | Sux4J submodule entry                 |
