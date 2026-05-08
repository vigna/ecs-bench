# Sux4J Java Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `java` target to `bench/run.py` that compares Sux4J GOV3Function (filter) and GV3CompressedFunction query performance against sux-rs, using Java for construction and Sux4J's C evaluation code for queries.

**Architecture:** Java Maven project builds structures and dumps C-compatible binaries. C test harnesses (from Sux4J submodule, extended with `#ifdef FILTER` and `SF_16`/`SF_32`) load dumps and run timed queries. `run.py` orchestrates the pipeline and stores results in the existing format.

**Tech Stack:** Java 11+ (Maven, Sux4J from Central), C (clang/gcc, Sux4J `c/` directory), Python (bench framework)

---

### Task 1: Add Sux4J Git Submodule

**Files:**
- Modify: `.gitmodules`
- Modify: `Makefile` (add Sux4J to `init` target)

- [ ] **Step 1: Add the submodule**

```bash
cd /Users/vigna/git/ecs-bench
git submodule add git@github.com:vigna/Sux4J.git Sux4J
```

- [ ] **Step 2: Update the Makefile `init` target to include Sux4J**

In `Makefile`, change the `init` target from:

```makefile
init:
	git submodule update --init --recursive BuRR BuRR-VL sux-rs
	git submodule update --init LearnedStaticFunction
	git submodule update --init MMPHF-Experiments
	cd MMPHF-Experiments && git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init cpp/extlib/LeMonHash
	cd MMPHF-Experiments/cpp/extlib/LeMonHash && git submodule update --init --recursive
```

to:

```makefile
init:
	git submodule update --init --recursive BuRR BuRR-VL sux-rs
	git submodule update --init Sux4J
	git submodule update --init LearnedStaticFunction
	git submodule update --init MMPHF-Experiments
	cd MMPHF-Experiments && git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init cpp/extlib/LeMonHash
	cd MMPHF-Experiments/cpp/extlib/LeMonHash && git submodule update --init --recursive
```

- [ ] **Step 3: Verify the submodule is present and `c/` directory exists**

```bash
ls Sux4J/c/sf3.c Sux4J/c/csf3.c Sux4J/c/spooky.c
```

Expected: all three files listed.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules Sux4J Makefile
git commit -m "Add Sux4J as submodule for C evaluation benchmarks"
```

---

### Task 2: Extend Sux4J C Code — `SF_16`, `SF_32`, and `#ifdef FILTER`

These changes go into the Sux4J submodule. They will be committed there
and the submodule pointer updated in ecs-bench.

**Files:**
- Modify: `Sux4J/c/sf3.c`

- [ ] **Step 1: Add `SF_16`/`SF_32` to `sf3_get_byte_array`**

In `Sux4J/c/sf3.c`, replace the existing `#ifdef SF_8` block in
`sf3_get_byte_array` (lines 59–64):

```c
#ifdef SF_8
	const uint8_t *p = (uint8_t *)sf->array + bucket_offset;
	return p[e[0]] ^ p[e[1]] ^ p[e[2]];
#else
	return get_value(sf->array, e[0] + bucket_offset, sf->width) ^ get_value(sf->array, e[1] + bucket_offset, sf->width) ^ get_value(sf->array, e[2] + bucket_offset, sf->width);
#endif
```

with:

```c
	/* SF_8/SF_16/SF_32: direct typed-pointer access for power-of-2 widths.
	   Requires little-endian byte order (the packed bit array's byte layout
	   must match typed-pointer indexing). */
#ifdef SF_8
	const uint8_t *p = (uint8_t *)sf->array + bucket_offset;
	const int64_t result = p[e[0]] ^ p[e[1]] ^ p[e[2]];
#elif defined(SF_16)
	const uint16_t *p = (uint16_t *)((uint8_t *)sf->array) + bucket_offset;
	const int64_t result = p[e[0]] ^ p[e[1]] ^ p[e[2]];
#elif defined(SF_32)
	const uint32_t *p = (uint32_t *)((uint8_t *)sf->array) + bucket_offset;
	const int64_t result = p[e[0]] ^ p[e[1]] ^ p[e[2]];
#else
	const int64_t result = get_value(sf->array, e[0] + bucket_offset, sf->width) ^ get_value(sf->array, e[1] + bucket_offset, sf->width) ^ get_value(sf->array, e[2] + bucket_offset, sf->width);
#endif
#ifdef FILTER
	return ((result ^ signature[0]) & ((UINT64_C(1) << sf->width) - 1)) == 0;
#else
	return result;
#endif
```

- [ ] **Step 2: Apply the same pattern to `sf3_get_uint64_t`**

Replace the `#ifdef SF_8` block in `sf3_get_uint64_t` (lines 76–81)
with the same pattern as Step 1 (identical code — the `signature` local
is already available from line 68).

- [ ] **Step 3: Apply the same pattern to `sf3_get_signature`**

Replace the `#ifdef SF_8` block in `sf3_get_signature` (lines 91–96)
with the same pattern. Note: for the signature variant, the `signature`
parameter is the function argument (line 84).

- [ ] **Step 4: Verify the file compiles with each flag combination**

```bash
cd Sux4J/c
cc -O3 -c -DSF_8 -DFILTER sf3.c sf.c spooky.c && echo "SF_8 FILTER ok"
cc -O3 -c -DSF_16 -DFILTER sf3.c sf.c spooky.c && echo "SF_16 FILTER ok"
cc -O3 -c -DSF_32 -DFILTER sf3.c sf.c spooky.c && echo "SF_32 FILTER ok"
cc -O3 -c -DFILTER sf3.c sf.c spooky.c && echo "generic FILTER ok"
cc -O3 -c sf3.c sf.c spooky.c && echo "no-filter ok"
rm -f *.o
```

Expected: all five lines print "ok".

- [ ] **Step 5: Commit inside the Sux4J submodule**

```bash
cd Sux4J
git add c/sf3.c
git commit -m "Add SF_16/SF_32 optimizations and #ifdef FILTER for benchmark comparisons"
cd ..
```

---

### Task 3: Create C Test Harness for uint64_t Queries

Create a new `test_uint64_t.c` template in `Sux4J/c/` that replaces the
existing `test_signature.c` pattern. It generates random uint64 keys,
queries the function, and prints `Min: X Median: X Max: X Average: X`.

**Files:**
- Create: `Sux4J/c/test_uint64_t.c`
- Create: `Sux4J/c/test_sf3_uint64_t.c` (thin wrapper)
- Create: `Sux4J/c/test_csf3_uint64_t.c` (thin wrapper)

- [ ] **Step 1: Create the generic `test_uint64_t.c` template**

Create `Sux4J/c/test_uint64_t.c`:

```c
/*
 * Sux: Succinct data structures
 *
 * Copyright (C) 2018-2025 Sebastiano Vigna
 *
 *  This library is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by the Free
 *  Software Foundation; either version 3 of the License, or (at your option)
 *  any later version.
 *
 *  This library is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 *  for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <inttypes.h>
#include <fcntl.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#define SAMPLES 11

static uint64_t get_system_time(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

static int cmp_uint64_t(const void *a, const void *b) {
	return *(uint64_t *)a < *(uint64_t *)b ? -1 : *(uint64_t *)a > *(uint64_t *)b ? 1 : 0;
}

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static uint64_t s[2];

static uint64_t next(void) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = s0 + s1;
	s1 ^= s0;
	s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
	s[1] = rotl(s1, 37);
	return result;
}

int main(int argc, char *argv[]) {
	int h = open(argv[1], O_RDONLY);
	assert(h >= 0);
	SUX4J_MAP *map = SUX4J_LOAD_MAP(h);
	close(h);

#define NKEYS 10000000
	uint64_t u = 0;
	uint64_t sample[SAMPLES];

	for (int k = SAMPLES; k-- != 0;) {
		s[0] = 0x5603141978c51071;
		s[1] = 0x3bbddc01ebdf4b72;

		int64_t elapsed = -get_system_time();
		for (int i = 0; i < NKEYS; ++i)
			u += SUX4J_GET_UINT64_T(map, next());
		elapsed += get_system_time();

		sample[k] = elapsed;
	}
	const volatile uint64_t unused = u;

	qsort(sample, SAMPLES, sizeof *sample, cmp_uint64_t);

	double min_ns = sample[0] * 1000.0 / NKEYS;
	double median_ns = sample[SAMPLES / 2] * 1000.0 / NKEYS;
	double max_ns = sample[SAMPLES - 1] * 1000.0 / NKEYS;
	double sum = 0;
	for (int k = 0; k < SAMPLES; k++) sum += sample[k];
	double avg_ns = sum * 1000.0 / NKEYS / SAMPLES;

	printf("Min: %.1f Median: %.1f Max: %.1f Average: %.1f\n",
	       min_ns, median_ns, max_ns, avg_ns);
}
```

- [ ] **Step 2: Create `test_sf3_uint64_t.c`**

Create `Sux4J/c/test_sf3_uint64_t.c`:

```c
#include "sf3.h"

#define SUX4J_MAP sf
#define SUX4J_LOAD_MAP load_sf
#define SUX4J_GET_UINT64_T sf3_get_uint64_t

#include "test_uint64_t.c"
```

- [ ] **Step 3: Create `test_csf3_uint64_t.c`**

Create `Sux4J/c/test_csf3_uint64_t.c`:

```c
#include "csf3.h"

#define SUX4J_MAP csf
#define SUX4J_LOAD_MAP load_csf
#define SUX4J_GET_UINT64_T csf3_get_uint64_t

#include "test_uint64_t.c"
```

- [ ] **Step 4: Test-compile both harnesses**

```bash
cd Sux4J/c
cc -O3 -DFILTER -DSF_8 -o test_sf3_filter8 test_sf3_uint64_t.c sf3.c sf.c spooky.c && echo "sf3 filter8 ok"
cc -O3 -o test_csf3_uint64 test_csf3_uint64_t.c csf3.c csf.c spooky.c && echo "csf3 ok"
rm -f test_sf3_filter8 test_csf3_uint64
```

Expected: both "ok" lines print.

- [ ] **Step 5: Commit inside the Sux4J submodule**

```bash
cd Sux4J
git add c/test_uint64_t.c c/test_sf3_uint64_t.c c/test_csf3_uint64_t.c
git commit -m "Add uint64_t test harnesses with Min/Median/Max/Average output"
cd ..
```

---

### Task 4: Create Java Maven Project

**Files:**
- Create: `java/pom.xml`
- Create: `java/src/main/java/it/unimi/dsi/ecs/BenchBuilder.java`

- [ ] **Step 1: Create `java/pom.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
                             http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>it.unimi.dsi</groupId>
  <artifactId>ecs-bench-java</artifactId>
  <version>1.0-SNAPSHOT</version>
  <packaging>jar</packaging>

  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>

  <dependencies>
    <dependency>
      <groupId>it.unimi.dsi</groupId>
      <artifactId>sux4j</artifactId>
      <version>5.4.1</version>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-jar-plugin</artifactId>
        <version>3.3.0</version>
        <configuration>
          <archive>
            <manifest>
              <mainClass>it.unimi.dsi.ecs.BenchBuilder</mainClass>
            </manifest>
          </archive>
        </configuration>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-dependency-plugin</artifactId>
        <version>3.6.1</version>
        <executions>
          <execution>
            <id>copy-dependencies</id>
            <phase>package</phase>
            <goals><goal>copy-dependencies</goal></goals>
            <configuration>
              <outputDirectory>${project.build.directory}/lib</outputDirectory>
            </configuration>
          </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
</project>
```

- [ ] **Step 2: Create `BenchBuilder.java`**

Create `java/src/main/java/it/unimi/dsi/ecs/BenchBuilder.java`:

```java
package it.unimi.dsi.ecs;

import it.unimi.dsi.sux4j.mph.GOV3Function;
import it.unimi.dsi.sux4j.mph.GV3CompressedFunction;
import it.unimi.dsi.sux4j.mph.codec.Codec;
import it.unimi.dsi.bits.TransformationStrategies;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.SplittableRandom;

/**
 * Builds Sux4J GOV3Function (filter) or GV3CompressedFunction structures
 * over uint64 keys encoded as little-endian byte arrays, then dumps
 * them in the C-compatible binary format for benchmarking.
 *
 * Usage:
 *   java ... BenchBuilder gov3 <n> <bits> <output>
 *   java ... BenchBuilder gov3compressed <n> --geometric <output>
 *   java ... BenchBuilder gov3compressed <n> --zipf <K> <output>
 *   java ... BenchBuilder gov3compressed <n> --uniform <K> <output>
 */
public class BenchBuilder {

	private static ObjectArrayList<byte[]> generateKeys(final long n) {
		final ObjectArrayList<byte[]> keys = new ObjectArrayList<>();
		final ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
		for (long i = 0; i < n; i++) {
			buf.clear();
			buf.putLong(i);
			keys.add(buf.array().clone());
		}
		return keys;
	}

	private static LongArrayList generateGeometric(final long n) {
		final SplittableRandom rng = new SplittableRandom(0);
		final LongArrayList values = new LongArrayList();
		for (long i = 0; i < n; i++)
			values.add(Long.numberOfTrailingZeros(rng.nextLong()));
		return values;
	}

	private static double[] zipfCdf(final int k) {
		double h = 0;
		for (int i = 1; i <= k; i++) h += 1.0 / i;
		final double[] cdf = new double[k];
		double acc = 0;
		for (int i = 1; i <= k; i++) {
			acc += 1.0 / (i * h);
			cdf[i - 1] = acc;
		}
		return cdf;
	}

	private static int sampleZipf(final double[] cdf, final SplittableRandom rng) {
		final double u = (rng.nextLong() >>> 11) / (double)(1L << 53);
		int lo = 0, hi = cdf.length;
		while (lo < hi) {
			final int mid = (lo + hi) >>> 1;
			if (cdf[mid] < u) lo = mid + 1;
			else hi = mid;
		}
		return lo;
	}

	private static LongArrayList generateZipf(final long n, final int k) {
		final SplittableRandom rng = new SplittableRandom(0);
		final double[] cdf = zipfCdf(k);
		final LongArrayList values = new LongArrayList();
		for (long i = 0; i < n; i++)
			values.add(sampleZipf(cdf, rng));
		return values;
	}

	private static LongArrayList generateUniform(final long n, final int k) {
		final SplittableRandom rng = new SplittableRandom(0);
		final LongArrayList values = new LongArrayList();
		for (long i = 0; i < n; i++)
			values.add(Long.remainderUnsigned(rng.nextLong(), k));
		return values;
	}

	public static void main(final String[] args) throws IOException {
		if (args.length < 3) {
			System.err.println("Usage: BenchBuilder gov3 <n> <bits> <output>");
			System.err.println("       BenchBuilder gov3compressed <n> --geometric <output>");
			System.err.println("       BenchBuilder gov3compressed <n> --zipf <K> <output>");
			System.err.println("       BenchBuilder gov3compressed <n> --uniform <K> <output>");
			System.exit(1);
		}

		final String type = args[0];
		final long n = Long.parseLong(args[1]);
		final ObjectArrayList<byte[]> keys = generateKeys(n);

		if ("gov3".equals(type)) {
			final int bits = Integer.parseInt(args[2]);
			final String output = args[3];

			System.err.println("Building GOV3Function: n=" + n + " bits=" + bits);
			final GOV3Function<byte[]> f = new GOV3Function.Builder<byte[]>()
				.keys(keys)
				.transform(TransformationStrategies.rawByteArray())
				.dictionary(bits)
				.build();

			System.err.println("Dumping to " + output);
			f.dump(output);

		} else if ("gov3compressed".equals(type)) {
			final String distFlag = args[2];
			final LongArrayList values;
			final String output;

			if ("--geometric".equals(distFlag)) {
				values = generateGeometric(n);
				output = args[3];
			} else if ("--zipf".equals(distFlag)) {
				final int k = Integer.parseInt(args[3]);
				values = generateZipf(n, k);
				output = args[4];
			} else if ("--uniform".equals(distFlag)) {
				final int k = Integer.parseInt(args[3]);
				values = generateUniform(n, k);
				output = args[4];
			} else {
				System.err.println("Unknown distribution: " + distFlag);
				System.exit(1);
				return;
			}

			System.err.println("Building GV3CompressedFunction: n=" + n + " dist=" + distFlag);
			final GV3CompressedFunction<byte[]> f = new GV3CompressedFunction.Builder<byte[]>()
				.keys(keys)
				.transform(TransformationStrategies.rawByteArray())
				.values(values)
				.codec(new Codec.Huffman())
				.build();

			System.err.println("Dumping to " + output);
			f.dump(output);

		} else {
			System.err.println("Unknown type: " + type + " (expected gov3 or gov3compressed)");
			System.exit(1);
		}

		System.err.println("Done.");
	}
}
```

- [ ] **Step 3: Build the Maven project**

```bash
cd java && mvn -q package -DskipTests
```

Expected: `BUILD SUCCESS`, jar at `java/target/ecs-bench-java-1.0-SNAPSHOT.jar`.

- [ ] **Step 4: Smoke test the Java builder (small n)**

```bash
cd java
java -cp "target/ecs-bench-java-1.0-SNAPSHOT.jar:target/lib/*" \
     it.unimi.dsi.ecs.BenchBuilder gov3 1024 8 /tmp/test_gov3.bin
ls -la /tmp/test_gov3.bin
rm /tmp/test_gov3.bin
```

Expected: file created, stderr shows "Building GOV3Function" + "Dumping" + "Done.".

- [ ] **Step 5: Commit**

```bash
cd /Users/vigna/git/ecs-bench
git add java/
git commit -m "Add Java builder for Sux4J GOV3Function/GV3CompressedFunction benchmarks"
```

---

### Task 5: Add Method Definitions and Parser Extension to `configs.py` / `parse.py`

**Files:**
- Modify: `bench/configs.py`
- Modify: `bench/parse.py`

- [ ] **Step 1: Add GOV3 method definitions to `configs.py`**

Add after the existing `METHODS` entries (before the closing `}`):

```python
    # -----------------------------------------------------------------
    # Sux4J GOV3 methods (Java build + C query)
    # -----------------------------------------------------------------
    #
    # Build: Java BenchBuilder creates GOV3Function/GV3CompressedFunction
    #        with LE-encoded uint64 keys, dumps C-binary.
    # Query: C test harness (sf3/csf3) loads dump, benchmarks queries.
    "GOV3Function": {
        "tool": "java",
        "label": "GOV3 (Sux4J)",
        "java_type": "gov3",
        "max_log2": 30,
    },
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

- [ ] **Step 2: Add table presets to `configs.py`**

Add to the `TABLES` dict:

```python
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
            ("GOV3Comp", ["GOV3Compressed-Geo", "GOV3Compressed-Zipf", "GOV3Compressed-Uni256"]),
        ],
        "dist_labels": ["Geometric", "Zipf", "Uniform"],
        "sizes": [1 << 20, 1 << 28],
    },
```

- [ ] **Step 3: Add `"java"` to the parse tool map in `parse.py`**

In `bench/parse.py`, change `_PARSE_TOOL_MAP` (line 94 of `run.py`) — wait, it's
actually in `run.py`. Find the line:

```python
_PARSE_TOOL_MAP = {"csf": "burr", "comp-vfunc": "sux", "phast": "sux",
                   "mmphf": "sux", "lemonhash": "sux"}
```

and change it to:

```python
_PARSE_TOOL_MAP = {"csf": "burr", "comp-vfunc": "sux", "phast": "sux",
                   "mmphf": "sux", "lemonhash": "sux", "java": "sux"}
```

- [ ] **Step 4: Commit**

```bash
git add bench/configs.py bench/run.py
git commit -m "Add GOV3 method definitions and java parse-tool mapping"
```

---

### Task 6: Add C Build Helpers and Java Build Helper to `run.py`

**Files:**
- Modify: `bench/run.py`

- [ ] **Step 1: Add `ensure_java_built` function**

Add after the existing `ensure_mmphf_built` function (around line 286):

```python
def ensure_java_built():
    """Compile the Java benchmark builder via Maven if the jar is missing."""
    java_dir = Path(REPO_ROOT) / "java"
    jar = java_dir / "target" / "ecs-bench-java-1.0-SNAPSHOT.jar"
    if jar.exists():
        return True
    print("Building Java benchmark builder…")
    r = subprocess.run(
        ["mvn", "-q", "package", "-DskipTests"],
        cwd=str(java_dir), capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("WARNING: Java build failed, skipping GOV3 methods")
        return False
    return True
```

- [ ] **Step 2: Add `ensure_sux4j_c_built` function**

Add after `ensure_java_built`:

```python
# Map (bits, is_compressed) to (binary_name, compile_flags).
_SUX4J_C_VARIANTS = {}
for b in EVEN_BITS:
    flag = f"-DSF_{b}"
    _SUX4J_C_VARIANTS[(b, False)] = (f"test_sf3_filter{b}", [flag, "-DFILTER"])
for b in ODD_BITS:
    _SUX4J_C_VARIANTS[(b, False)] = (f"test_sf3_filter{b}", ["-DFILTER"])
_SUX4J_C_VARIANTS[("csf", True)] = ("test_csf3_uint64", [])


def ensure_sux4j_c_built(bits_values):
    """Compile the Sux4J C test harnesses for the requested bit widths.

    Produces binaries like test_sf3_filter8, test_sf3_filter16, etc.
    in the Sux4J/c/ directory.
    """
    sux4j_c = Path(REPO_ROOT) / "Sux4J" / "c"
    if not sux4j_c.exists():
        print("WARNING: Sux4J/c/ not found, skipping GOV3 methods")
        return False

    cc = os.environ.get("CC", "cc")
    cflags = ["-O3", "-march=native"]

    # Filter harnesses (one per bit width)
    for bits in bits_values:
        key = (bits, False)
        if key not in _SUX4J_C_VARIANTS:
            continue
        binary_name, extra_flags = _SUX4J_C_VARIANTS[key]
        binary_path = sux4j_c / binary_name
        if binary_path.exists():
            continue
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

    # CSF harness (one binary for all distributions)
    key = ("csf", True)
    binary_name, extra_flags = _SUX4J_C_VARIANTS[key]
    binary_path = sux4j_c / binary_name
    if not binary_path.exists():
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
```

- [ ] **Step 3: Verify the file still parses**

```bash
python -c "import bench.run"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add bench/run.py
git commit -m "Add ensure_java_built and ensure_sux4j_c_built helpers"
```

---

### Task 7: Add `_bench_java` Helper and `cmd_java` Subcommand

**Files:**
- Modify: `bench/run.py`

- [ ] **Step 1: Add Java command constructors**

Add after the existing command constructors section (around line 460):

```python
def java_build_cmd(cfg, n, bits, outfile):
    """Assemble the Java BenchBuilder command line."""
    java_dir = Path(REPO_ROOT) / "java"
    jar = java_dir / "target" / "ecs-bench-java-1.0-SNAPSHOT.jar"
    classpath = f"{jar}:{java_dir / 'target' / 'lib' / '*'}"
    cmd = ["java", "-cp", classpath, "it.unimi.dsi.ecs.BenchBuilder"]

    if cfg["java_type"] == "gov3":
        cmd += ["gov3", str(n), str(bits), str(outfile)]
    elif cfg["java_type"] == "gov3compressed":
        cmd += ["gov3compressed", str(n)]
        cmd += cfg["distribution"]
        cmd += [str(outfile)]
    return cmd


def java_query_cmd(cfg, bits, dumpfile):
    """Assemble the Sux4J C query harness command line."""
    sux4j_c = Path(REPO_ROOT) / "Sux4J" / "c"
    if cfg["java_type"] == "gov3compressed":
        binary = "test_csf3_uint64"
    else:
        binary = f"test_sf3_filter{bits}"
    return [str(sux4j_c / binary), str(dumpfile)]
```

- [ ] **Step 2: Add `_bench_java` helper**

Add after the Java command constructors:

```python
def _bench_java(cfg, method_name, n, bits, log2_n, rdir, args):
    """Run a single Sux4J benchmark: Java build+dump, C query, clean up."""
    ddir = data_dir(args.results_dir, method_name)
    is_compressed = cfg["java_type"] == "gov3compressed"
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

    # Record construction output (Java stderr) + dump file size
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
```

- [ ] **Step 3: Add `cmd_java` subcommand function**

Add before the CLI section:

```python
def cmd_java(args):
    """Build Sux4J structures (Java), query them (C), and clean up."""
    methods = resolve_methods(args.method)
    bits_list = resolve_bits(args.bits)

    # Filter to java-tool methods only.
    java_methods = [m for m in methods if METHODS[m]["tool"] == "java"]
    if not java_methods:
        print("No Java methods selected.")
        return

    # Ensure Java + C are built.
    if not ensure_java_built():
        return

    filter_methods = [m for m in java_methods
                      if METHODS[m]["java_type"] == "gov3"]
    comp_methods = [m for m in java_methods
                    if METHODS[m]["java_type"] == "gov3compressed"]

    if filter_methods:
        if not ensure_sux4j_c_built(bits_list):
            return
    if comp_methods:
        if not ensure_sux4j_c_built([]):  # just builds CSF harness
            return

    # Build work list, ordered by size first.
    work = []
    for method_name in java_methods:
        cfg = METHODS[method_name]
        max_log2 = min(args.max_log2, cfg.get("max_log2", 33))
        is_compressed = cfg["java_type"] == "gov3compressed"
        method_bits = [0] if is_compressed else bits_list

        for bits in method_bits:
            for log2_n in range(MIN_LOG2, max_log2 + 1):
                work.append((log2_n, method_name, bits))

    work.sort(key=lambda t: t[0])

    for log2_n, method_name, bits in work:
        cfg = METHODS[method_name]
        n = 1 << log2_n
        rdir = result_dir(args.results_dir, method_name)

        bench_err = rdir / f"bench.{n}.{bits}.err"
        tool_for_parse = _parse_tool(cfg["tool"])
        if not args.force and result_complete(bench_err, tool_for_parse):
            continue

        _bench_java(cfg, method_name, n, bits, log2_n, rdir, args)

    print("Done.")
```

- [ ] **Step 4: Register `cmd_java` in the CLI parser and dispatch**

In `make_parser()`, add after the `clean` subparser:

```python
    # --- java ---
    jv = sub.add_parser("java", help="Build Sux4J structures (Java), query (C)")
    add_common(jv)
    jv.add_argument("--keep-data", action="store_true", default=False,
                    help="Keep dump files after query")
```

In the `dispatch` dict in `main()`, add:

```python
        "java": cmd_java,
```

- [ ] **Step 5: Verify the CLI parses correctly**

```bash
python bench/run.py java --help
```

Expected: help text for the `java` subcommand.

- [ ] **Step 6: Dry-run test**

```bash
python bench/run.py java --method GOV3Function --bits 8 --max-log2 20 --dry-run
```

Expected: prints dry-run command lines for Java build + C query.

- [ ] **Step 7: Commit**

```bash
git add bench/run.py
git commit -m "Add java subcommand for Sux4J benchmark comparison"
```

---

### Task 8: Wire `java` Tool Into Existing `bench` Command

The `java` tool methods should also work when invoked through the regular
`bench` subcommand (not just the dedicated `java` subcommand).

**Files:**
- Modify: `bench/run.py`

- [ ] **Step 1: Add Java dispatching to `cmd_bench`**

In `cmd_bench`, after the existing tool checks (around line 517), add:

```python
    java_needed = any(METHODS[m]["tool"] == "java" for m in methods)
    java_ok = True
    if java_needed:
        java_ok = ensure_java_built()
        if java_ok:
            filter_bits = bits_list if any(
                METHODS[m].get("java_type") == "gov3" for m in methods
                if METHODS[m]["tool"] == "java") else []
            java_ok = ensure_sux4j_c_built(filter_bits)
```

In the dispatch block inside the work loop (after the `elif cfg["tool"] == "lemonhash":` block), add:

```python
        elif cfg["tool"] == "java":
            if not java_ok:
                continue
            _bench_java(cfg, method_name, n, bits, log2_n,
                        rdir, args)
```

- [ ] **Step 2: Verify `bench --method GOV3Function` dry-run**

```bash
python bench/run.py bench --method GOV3Function --bits 8 --max-log2 20 --dry-run
```

Expected: prints dry-run lines for Java build + C query.

- [ ] **Step 3: Commit**

```bash
git add bench/run.py
git commit -m "Wire java tool into existing bench subcommand"
```

---

### Task 9: Add `java` and `java-comp` Table Renderers

**Files:**
- Modify: `bench/run.py`

- [ ] **Step 1: Add `_table_java` renderer**

Add after the existing `_table_comp` function:

```python
def _table_java(tbl, args):
    """Render the 'java' table: filter query + overhead comparison."""
    bits_order = tbl["bits_order"]
    ncols = len(bits_order)
    colspec = "l" + "r" * ncols * 2

    for n in tbl["sizes"]:
        log2_n = n.bit_length() - 1

        print(f"% Java comparison, n = 2^{{{log2_n}}}")
        print(r"\begin{tabular}{" + colspec + "}")
        print(r"\toprule")

        print(f"& \\multicolumn{{{ncols}}}{{c}}{{Query (ns/key)}}"
              f" & \\multicolumn{{{ncols}}}{{c}}{{Overhead (\\%)}} \\\\")

        hdr = ""
        for bits in bits_order:
            hdr += f" & {bits}"
        hdr += hdr
        print(hdr + r" \\")
        print(r"\midrule")

        for label, variants in tbl["rows"]:
            method_name = variants["query"]
            cfg = METHODS.get(method_name)
            if cfg is None:
                continue
            rdir = Path(args.results_dir) / method_name
            tool_for_parse = _parse_tool(cfg["tool"])

            row = label
            query_cells = ""
            overhead_cells = ""
            for bits in bits_order:
                bench_err = rdir / f"bench.{n}.{bits}.err"
                size_file = rdir / f"bench.{n}.{bits}.size"

                qval = ""
                oval = ""
                if bench_err.exists():
                    text = bench_err.read_text()
                    qd = parse_query(text, tool_for_parse)
                    v = qd.get("query_min")
                    if v is not None:
                        qval = f"{v:.1f}"

                if size_file.exists():
                    file_bytes = int(size_file.read_text().strip())
                    overhead = compute_overhead(file_bytes, n, bits)
                    oval = f"{overhead:.1f}"

                query_cells += f" & {qval}"
                overhead_cells += f" & {oval}"

            print(row + query_cells + overhead_cells + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()
```

- [ ] **Step 2: Add `_table_java_comp` renderer**

Add after `_table_java`:

```python
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
```

- [ ] **Step 3: Register the table renderers in `cmd_table`**

In `cmd_table`, add to the dispatch logic:

```python
    elif args.table == "java":
        _table_java(tbl, args)
    elif args.table == "java-comp":
        _table_java_comp(tbl, args)
```

- [ ] **Step 4: Verify table rendering with no data (empty cells)**

```bash
python bench/run.py table --table java
python bench/run.py table --table java-comp
```

Expected: LaTeX table structure with empty data cells (no results yet).

- [ ] **Step 5: Commit**

```bash
git add bench/run.py
git commit -m "Add java and java-comp table renderers"
```

---

### Task 10: End-to-End Smoke Test

- [ ] **Step 1: Run a small filter benchmark**

```bash
python bench/run.py java --method GOV3Function --bits 8 --max-log2 20
```

Expected: Java builds GOV3Function at 2^20 with 8 bits, C harness queries
it, results stored in `results/GOV3Function/bench.1048576.8.err` and
`.size`.

- [ ] **Step 2: Check the result file has valid query output**

```bash
cat results/GOV3Function/bench.1048576.8.err
```

Expected: Java construction output, then `--- query ---`, then
`Min: X Median: X Max: X Average: X`.

- [ ] **Step 3: Run a small compressed function benchmark**

```bash
python bench/run.py java --method GOV3Compressed-Geo --max-log2 20
```

Expected: similar output in `results/GOV3Compressed-Geo/bench.1048576.0.err`.

- [ ] **Step 4: Verify the report parses results**

```bash
python bench/run.py report --method GOV3Function --bits 8
```

Expected: TSV row with query timing.

- [ ] **Step 5: Verify the table renders with data**

```bash
python bench/run.py table --table java
```

Expected: LaTeX table with populated timing and overhead cells.

- [ ] **Step 6: Commit the submodule pointer update**

```bash
git add Sux4J
git commit -m "Update Sux4J submodule to include C harness changes"
```

---

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `java` target documentation to the Benchmarks section**

Add after the existing `par-query` example in CLAUDE.md:

```markdown
# Sux4J comparison (Java build + C query)
python bench/run.py java --method GOV3Function --bits 8 --max-log2 30
python bench/run.py java --method GOV3Compressed-Geo --max-log2 28

# Comparison tables
python bench/run.py table --table java
python bench/run.py table --table java-comp
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Document java benchmark target in CLAUDE.md"
```
