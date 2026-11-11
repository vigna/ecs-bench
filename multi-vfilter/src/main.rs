//! BuRR-style parallel query benchmark for VFilter.
//!
//! Replicates `parbench.cpp` of BuRR, which the BuRR paper uses for its
//! highly parallel workload: `k` filters are built on the same keys
//! [0 . . n) with different seeds, using `b` construction threads, and then
//! `t` query threads perform `q` positive and then `q` negative queries each,
//! on random (key, filter) pairs. Pairs are generated in blocks of 2²⁰
//! outside of the timed loop, and the reported time per query is the total
//! query time of all threads divided by the number of queries, as in
//! `parbench.cpp`. With fewer filters than construction threads, each filter
//! is built with several threads.
//!
//! Unlike the original `parbench.cpp`, which our copy of BuRR has been fixed
//! to match, the filter of a query is chosen using a random word independent
//! from the one used for the key: otherwise, the filter would be a function
//! of the key, and only `n` distinct pairs would be queried.
//!
//! With `--func`, the benchmark uses static functions instead of filters:
//! [`VFunc`], or a [`PHastR`] with an array of values (PHastR + array).
//! Each function maps key *i* to *i* using ⌈lg *n*⌉ bits per value, as in
//! the single-structure benchmarks, so a positive query checks that the
//! value is the key. There are no negative queries.
//!
//! The last line of the output is a `RESULT type=agg` line using the same
//! keys as the corresponding line of `parbench.cpp` (`tcons` in
//! milliseconds, `tpospq` and `tneqpq` in nanoseconds per query).

use anyhow::Result;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use sux::bits::AtomicBitFieldVec;
use sux::func::{PHastR, PHastRBuilder, VFunc};
use sux::traits::AtomicBitFieldSlice;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::cli::ShardEdgeType;
use sux::dict::VFilter;
use sux::func::VBuilder;
use sux::func::shard_edge::{Fuse3NoShards, FuseAug3Shards, Mwhc3Shards};
use sux::utils::FromCloneableIntoIterator;

#[derive(Parser, Debug)]
#[command(about = "BuRR-style parallel query benchmark for VFilter.", long_about = None)]
struct Args {
    /// The number of keys of each filter.​
    #[arg(short)]
    n: usize,
    /// The number of filters.​
    #[arg(short, default_value_t = 1280)]
    k: usize,
    /// The number of queries performed by each query thread.​
    #[arg(short, default_value_t = 100_000_000)]
    q: usize,
    /// The number of query threads.​
    #[arg(short, default_value_t = 96)]
    t: usize,
    /// The number of construction threads.​
    #[arg(short, default_value_t = 96)]
    b: usize,
    /// The number of bits of the hashes used by the filters (8, 16, and 32
    /// use a boxed slice, other widths a bit-field vector).​
    #[arg(long, default_value_t = 8)]
    bits: usize,
    /// The type of shard edge.​
    #[arg(long = "shard-edge", short = 'E', value_enum, default_value_t)]
    shard_edge: ShardEdgeType,
    /// The seed of the first filter; filter i uses seed + i.​
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// The sharding overhead (default: that of VBuilder).​
    #[arg(long)]
    eps: Option<f64>,
    /// Use static functions instead of filters (--bits and --shard-edge are
    /// ignored; VFunc uses FuseAug3Shards).​
    #[arg(long, value_enum)]
    func: Option<FuncType>,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum FuncType {
    /// VFunc.
    Vfunc,
    /// PHastR + array of values.
    Phastr,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn fast_range(r: u64, n: usize) -> usize {
    ((r as u128 * n as u128) >> 64) as usize
}

fn bench<F: Send + Sync + MemSize>(
    args: &Args,
    negative: bool,
    build: impl Fn(u64, usize) -> Result<F> + Sync,
    contains: impl Fn(&F, usize) -> bool + Send + Sync + Copy,
) -> Result<()> {
    let (n, k, q, t) = (args.n, args.k, args.q, args.t);

    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let built: Mutex<Vec<(usize, F)>> = Mutex::new(Vec::with_capacity(k));
    let error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    // With fewer filters than construction threads, each filter is built
    // with several threads, as in parbench.cpp
    let builders = args.b.min(k).max(1);
    let filter_threads = (args.b / builders).max(1);
    std::thread::scope(|s| {
        for _ in 0..builders {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= k {
                        break;
                    }
                    match build(args.seed + i as u64, filter_threads) {
                        Ok(filter) => built.lock().unwrap().push((i, filter)),
                        Err(e) => {
                            *error.lock().unwrap() = Some(e);
                            // Stop the other construction threads, too
                            next.store(k, Ordering::Relaxed);
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let mut built = built.into_inner().unwrap();
    built.sort_by_key(|(i, _)| *i);
    let filters: Vec<F> = built.into_iter().map(|(_, f)| f).collect();
    let cons_ns = start.elapsed().as_nanos() as f64;
    let bytes: usize = filters.iter().map(|f| f.mem_size(SizeFlags::default())).sum();
    println!(
        "Parallel construction of {k} filters with {} threads took {:.2} s = {:.2} ns per item; total size {:.3} GB = {:.3} bits/key",
        args.b,
        cons_ns / 1e9,
        cons_ns / (n * k) as f64,
        bytes as f64 / 1e9,
        bytes as f64 * 8.0 / (n * k) as f64
    );

    let mut per_query = [0.0; 2];
    let mut fps = 0;
    for positive in [true, false] {
        if !positive && !negative {
            break;
        }
        let query_ns = AtomicU64::new(0);
        let found = AtomicUsize::new(0);
        let wall = Instant::now();
        std::thread::scope(|s| {
            for id in 0..t {
                let (filters, query_ns, found) = (&filters, &query_ns, &found);
                s.spawn(move || {
                    const BLOCK: usize = 1 << 20;
                    let mut queries = vec![(0_usize, 0_usize); BLOCK];
                    let mut state = (id as u64 + if positive { 0 } else { t as u64 })
                        .wrapping_mul(0xd1b54a32d192ed03)
                        ^ 0x5851f42d4c957f2d;
                    let (mut count, mut my_found, mut t_query) = (0, 0_usize, 0_u128);
                    while count < q {
                        let block = BLOCK.min(q - count);
                        for item in &mut queries[..block] {
                            let r = splitmix64(&mut state);
                            let key = fast_range(r, n) + if positive { 0 } else { n };
                            // The filter must come from an independent random
                            // word: the high bits of r ^ c, which fast_range
                            // would use, are a function of the key
                            *item = (key, fast_range(splitmix64(&mut state), k));
                        }
                        count += block;
                        let start = Instant::now();
                        for &(key, f) in &queries[..block] {
                            // SAFETY: f < k = filters.len()
                            my_found += contains(unsafe { filters.get_unchecked(f) }, key) as usize;
                        }
                        t_query += start.elapsed().as_nanos();
                    }
                    query_ns.fetch_add(t_query as u64, Ordering::Relaxed);
                    found.fetch_add(std::hint::black_box(my_found), Ordering::Relaxed);
                });
            }
        });
        let wall_ns = wall.elapsed().as_nanos() as f64;
        let found = found.into_inner();
        let total = (q * t) as f64;
        per_query[!positive as usize] = query_ns.into_inner() as f64 / total;
        if positive {
            anyhow::ensure!(found == q * t, "positive check failed: found {found} of {}", q * t);
        } else {
            fps = found;
        }
        println!(
            "{} queries with {t} threads took {:.2} s total; queries took {:.2} ns per key; found {found} of {} ({:.5}%)",
            if positive { "Positive" } else { "Negative" },
            wall_ns / 1e9,
            per_query[!positive as usize],
            q * t,
            found as f64 * 100.0 / total
        );
    }
    if negative {
        println!(
            "RESULT type=agg n={n} k={k} bits={} tcons={} totalbytes={bytes} tpospq={} tneqpq={} fps={fps} fpr={} threads={t} build_threads={}",
            args.bits,
            cons_ns / 1e6,
            per_query[0],
            per_query[1],
            fps as f64 / (q * t) as f64,
            args.b,
        );
    } else {
        println!(
            "RESULT type=agg n={n} k={k} bits={} tcons={} totalbytes={bytes} tpospq={} threads={t} build_threads={}",
            args.bits,
            cons_ns / 1e6,
            per_query[0],
            args.b,
        );
    }
    Ok(())
}

// Builds the filters with a boxed slice of W (8, 16, or 32 bits)
macro_rules! run_boxed {
    ($args: expr, $w: ty, $s: ty, $e: ty) => {{
        let args = &$args;
        bench(
            args,
            true,
            |seed, threads| {
                <VFilter<usize, Box<[$w]>, $s, $e>>::try_new_with_builder(
                    FromCloneableIntoIterator::from(0..args.n),
                    builder(args, seed, threads),
                    no_logging![],
                )
            },
            |filter: &VFilter<usize, Box<[$w]>, $s, $e>, key| filter.contains(key),
        )
    }};
}

// Builds the filters with a bit-field vector, as vfilter does for widths
// other than 8, 16, 32, and 64 bits without --unaligned
macro_rules! run_bit_field_vec {
    ($args: expr, $w: ty, $s: ty, $e: ty) => {{
        let args = &$args;
        bench(
            args,
            true,
            |seed, threads| {
                <VFilter<usize, BitFieldVec<Box<[$w]>>, $s, $e>>::try_new_with_builder(
                    FromCloneableIntoIterator::from(0..args.n),
                    args.bits,
                    builder(args, seed, threads),
                    no_logging![],
                )
            },
            |filter: &VFilter<usize, BitFieldVec<Box<[$w]>>, $s, $e>, key| filter.contains(key),
        )
    }};
}

/// PHastR + array of values.
struct PHastRArray {
    func: PHastR<usize>,
    values: BitFieldVec<Box<[usize]>>,
}

impl MemSize for PHastRArray {
    fn mem_size_rec(&self, flags: SizeFlags, refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        self.func.mem_size_rec(flags, refs) + self.values.mem_size_rec(flags, refs)
    }
}

/// Runs a closure in a thread pool with the given number of threads, so that
/// the parallel parts of construction use only the threads of a builder.
fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?
        .install(f)
}

fn run_func(args: &Args, func: FuncType) -> Result<()> {
    let n = args.n;
    let width = (usize::BITS - (n - 1).max(1).leading_zeros()) as usize;
    match func {
        FuncType::Vfunc => bench(
            args,
            false,
            |seed, threads| {
                in_pool(threads, || {
                    <VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], FuseAug3Shards>>::try_new_with_builder(
                        FromCloneableIntoIterator::from(0..n),
                        FromCloneableIntoIterator::from(0..n),
                        builder(args, seed, threads),
                        no_logging![],
                    )
                })
            },
            |f: &VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], FuseAug3Shards>, key| {
                f.get(key) == key
            },
        ),
        FuncType::Phastr => bench(
            args,
            false,
            |seed, threads| {
                in_pool(threads, || {
                    let func = <PHastR<usize>>::try_new_with_builder(
                        FromCloneableIntoIterator::from(0..n),
                        PHastRBuilder::default().seed(seed),
                        no_logging![],
                    )?;
                    // Unaligned reads need a padding word; the array is filled
                    // through an atomic view, as distinct values may share a
                    // word
                    let mut values = BitFieldVec::<Box<[usize]>>::new_padded(width, n);
                    {
                        let atomic: AtomicBitFieldVec<&mut [AtomicUsize]> =
                            BitFieldVec::<&mut [usize]>::from(&mut values).try_into()?;
                        (0..n).into_par_iter().for_each(|i| {
                            atomic.set_atomic(func.get(i), i, Ordering::Relaxed);
                        });
                    }
                    Ok(PHastRArray { func, values })
                })
            },
            |f: &PHastRArray, key| {
                // SAFETY: the output of the PHastR is smaller than n
                unsafe { f.values.get_unaligned_unchecked(f.func.get(key)) == key }
            },
        ),
    }
}

fn builder<D, S, E: Default>(args: &Args, seed: u64, threads: usize) -> VBuilder<D, S, E> {
    let builder = VBuilder::default()
        .seed(seed)
        .max_num_threads(threads)
        .expected_num_keys(args.n);
    match args.eps {
        Some(eps) => builder.eps(eps),
        None => builder,
    }
}

macro_rules! dispatch_edge {
    ($args: expr, $run: ident, $w: ty) => {{
        match $args.shard_edge {
            ShardEdgeType::Fuse3NoShards64 => $run!($args, $w, [u64; 1], Fuse3NoShards),
            ShardEdgeType::Fuse3NoShards128 => $run!($args, $w, [u64; 2], Fuse3NoShards),
            ShardEdgeType::FuseAug3Shards => $run!($args, $w, [u64; 2], FuseAug3Shards),
            ShardEdgeType::Mwhc3 => $run!($args, $w, [u64; 2], Mwhc3Shards),
            other => anyhow::bail!("unsupported shard edge {other:?}"),
        }
    }};
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(func) = args.func {
        return run_func(&args, func);
    }
    match args.bits {
        8 => dispatch_edge!(args, run_boxed, u8),
        16 => dispatch_edge!(args, run_boxed, u16),
        32 => dispatch_edge!(args, run_boxed, u32),
        1..=63 => dispatch_edge!(args, run_bit_field_vec, u64),
        bits => anyhow::bail!("unsupported number of bits {bits} (use 1 to 63)"),
    }
}
