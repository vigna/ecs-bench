"""
Output parsers for sux-rs and BuRR benchmark stderr/stdout.

Each parser returns a dict of extracted metrics. Missing or unparseable
fields are silently omitted (the caller decides what to do).

Output conventions:
  - sux-rs prints timing to stderr.
  - BuRR prints timing to stdout (LOG1 = std::cout).

Key lines we look for:

  sux-rs build:
    "Construction … 45.3 ns/key"

  sux-rs query (bench_vfilter / bench_vfunc):
    "Min: 12.1 Median: 13.5 Max: 15.2 Average: 13.8"

  BuRR build:
    "Completed in 123ms, 1048576 keys, 45.3 ns/key"
    "Ribbon size: 12345 Bytes = …"

  BuRR query:
    Same Min/Median/Max/Average format as sux-rs.
"""

import math
import re


def parse_build(text, tool):
    """Extract construction ns/key from build output.

    Args:
        text: the captured output (stderr for sux, stdout for BuRR).
        tool: "sux" or "burr".

    Returns:
        dict with "constr_ns" (float) if found.
    """
    result = {}
    if tool == "sux":
        # sux-rs logs via dsi-progress-logger.  The final construction
        # summary looks like:
        #   "Construction from memory … 45.3 ns/key"
        # or
        #   "Construction completed … 45.3 ns/key"
        # We match any line with "onstruction" and "ns/key".
        # For large instances, the unit may be μs/key or ms/key.
        # Compound structures (LcpMmphf, Lcp2Mmphf, VFunc2) emit
        # multiple lines; we take the last one (the overall total).
        matches = re.findall(r'onstruction.*?(\d+\.?\d*)\s*(ns|μs|ms)/key', text)
        if not matches:
            matches = re.findall(r'Elapsed:.*?(\d+\.?\d*)\s*(ns|μs|ms)/key', text)
        if matches:
            val_str, unit = matches[-1]
            val = float(val_str)
            if unit == "μs":
                val *= 1000.0
            elif unit == "ms":
                val *= 1_000_000.0
            result["constr_ns"] = val
    elif tool == "burr":
        # BuRR logs: "Completed in Xms, N keys, X ns/key"
        m = re.search(r'Completed in.*?(\d+\.?\d*)\s*ns/key', text)
        if m:
            result["constr_ns"] = float(m.group(1))
    return result


def parse_query(text, tool):
    """Extract query timing from benchmark output.

    Both sux-rs and BuRR use the same summary format:
        "Min: X Median: X Max: X Average: X"
    BuRR appends " (0)" or " (1)" for the dead-code guard; we ignore it.

    Returns:
        dict with query_min, query_median, query_max, query_avg (floats).
    """
    result = {}
    m = re.search(
        r'Min:\s*(\S+)\s+Median:\s*(\S+)\s+Max:\s*(\S+)\s+Average:\s*(\S+)',
        text,
    )
    if m:
        result["query_min"] = float(m.group(1))
        result["query_median"] = float(m.group(2))
        result["query_max"] = float(m.group(3))
        result["query_avg"] = float(m.group(4))
    return result


def parse_burr_size(text):
    """Extract in-memory structure size from BuRR / BuRR-VL output.

    Matches:
        "Ribbon size: 12345 Bytes = …"  (standard BuRR)
        "Size: 12345 Bytes = …"         (BuRR-VL / bench_csf, LeMonHash)

    Returns:
        int (bytes) or None.
    """
    m = re.search(r'(?:Ribbon )?[Ss]ize:\s*(\d+)\s*Bytes', text)
    return int(m.group(1)) if m else None


def parse_bits_per_key(text, n=None):
    """Extract the self-reported in-memory bits/key from build output.

    Every builder logs the in-memory occupancy of the structure it just
    built.  Recognized formats:

        "Bits/key: 8.001 ..."            (sux-rs, Sux4J BenchBuilder, bench_phast)
        "= 12.345 bits/key"              (bench_csf, bench_lemonhash[_str])
        "Ribbon size: 12345 Bytes ..."   (BuRR) — combined with ``n`` below.

    The BuRR line gives only a byte count, so we compute bits/key from
    the byte count and ``n`` when supplied.

    Returns:
        float (bits/key) or None.
    """
    m = re.search(r'Bits/key:\s*([\d.]+)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'=\s*([\d.]+)\s*bits/key', text)
    if m:
        return float(m.group(1))
    if n is not None and n > 0:
        bytes_ = parse_burr_size(text)
        if bytes_ is not None:
            return bytes_ * 8.0 / n
    return None


def distribution_entropy(distribution):
    """Compute the entropy (bits) of a value distribution.

    Args:
        distribution: the config's "distribution" list, e.g.
            ["--geometric"], ["--zipf", "2000"], ["--uniform", "256"].

    Returns:
        float (bits per key), or None if unknown.
    """
    if distribution is None:
        return None
    flag = distribution[0]
    if flag == "--geometric":
        return 2.0
    elif flag == "--uniform" and len(distribution) > 1:
        K = int(distribution[1])
        return math.log2(K) if K > 0 else None
    elif flag == "--zipf" and len(distribution) > 1:
        K = int(distribution[1])
        if K <= 0:
            return None
        H = sum(1.0 / (k + 1) for k in range(K))
        return sum((1.0 / ((k + 1) * H)) * math.log2((k + 1) * H)
                   for k in range(K))
    return None


def compute_overhead(actual_bpk, ideal_bpk):
    """Compute space overhead as a percentage above the optimum.

    For filters, the optimum is the bit width.
    For compressed functions, the optimum is the value distribution's
    entropy.

    Args:
        actual_bpk: self-reported bits/key of the built structure.
        ideal_bpk:  optimum bits/key (bit width or entropy).

    Returns:
        float: percentage overhead (0% = exactly optimal).
    """
    if ideal_bpk <= 0:
        return 0.0
    return (actual_bpk / ideal_bpk) * 100.0 - 100.0
