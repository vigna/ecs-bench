//! Benchmarks a static function made of a [`PHastR`] minimal perfect hash
//! function and an array of values on 64-bit integer keys, for comparison
//! with [`VFunc`](sux::func::VFunc).
//!
//! The keys are the integers in [0 . . *n*), and the *i*-th key is mapped to
//! *i* using ⌈lg *n*⌉ bits per value. The keys are never materialized: they
//! are streamed to the construction, and then again to fill the array of
//! values.

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use sux::bits::{AtomicBitFieldVec, BitFieldVec};
use sux::cli::{INCR, bench, init_env_logger};
use sux::func::{PHastR, PHastRBuilder};
use sux::traits::AtomicBitFieldSlice;
use sux::utils::FromCloneableIntoIterator;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks PHastR + array with 64-bit integers", long_about = None)]
struct Args {
    /// The number of 64-bit keys.​
    n: usize,
    /// The maximum number of samples to take when testing query speed.​
    samples: usize,
    /// Use this number of threads.​
    threads: usize,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Keep the signatures of the keys on disk during construction.​
    #[arg(long)]
    offline: bool,
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();
    let n = args.n as u64;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;

    let start = std::time::Instant::now();

    let func = pool.install(|| {
        <PHastR<u64>>::try_new_with_builder(
            FromCloneableIntoIterator::new(0..n),
            PHastRBuilder::default().offline(args.offline),
            no_logging![],
        )
    })?;

    // Unaligned reads need a padding word at the end of the vector. We fill
    // it in parallel through an atomic view, as distinct values may share a
    // word.
    let mut output =
        BitFieldVec::<Box<[usize]>>::new_padded((args.n - 1).ilog2() as usize + 1, args.n);
    {
        let atomic: AtomicBitFieldVec<&mut [AtomicUsize]> =
            BitFieldVec::<&mut [usize]>::from(&mut output).try_into()?;
        pool.install(|| {
            (0..n).into_par_iter().for_each(|i| {
                atomic.set_atomic(func.get(i), i as usize, Ordering::Relaxed);
            })
        });
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let ns_per_key = elapsed.as_nanos() as f64 / args.n as f64;
    eprintln!(
        "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
        secs, args.n, ns_per_key
    );

    let bytes = func.mem_size(SizeFlags::default()) + output.mem_size(SizeFlags::default());
    eprintln!("Bits/key: {:.3}", (bytes * 8) as f64 / args.n as f64);

    bench(args.samples, args.repeats, || {
        let mut key: u64 = 0;
        let mut acc: usize = 0;
        for _ in 0..args.samples {
            key = key.wrapping_add(INCR as u64);
            acc ^= unsafe { output.get_unaligned_unchecked(func.get(key)) };
        }
        std::hint::black_box(acc);
    });

    Ok(())
}
