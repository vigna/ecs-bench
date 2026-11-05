//! Benchmarks queries on a HEM-sharded filter.
//!
//! The measurement protocol is that of `bench_vfilter` in sux-rs, down to the
//! query sequence and the reported statistics, so that the two numbers can be
//! compared directly: 10⁸ queries over a Weyl sequence (essentially all
//! negative), repeated five times, reporting the minimum.

use anyhow::Result;
use clap::Parser;
use hem_vfilter::{Aligned, Header, HemVFilter, Packed, PackedU, Values, read_header};
use std::io::{BufReader, Read};

const INCR: usize = 0x9e37_79b9_7f4a_7c15;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks queries on a HEM-sharded VFilter.", long_about = None)]
struct Args {
    /// The number of queries to perform (with --check, the number of keys).
    n: usize,
    /// The name of the serialized filter.
    filter: String,
    /// The number of repetitions.
    #[arg(short, long, default_value_t = 5)]
    repeats: usize,
    /// Instead of timing, check that the keys [0 . . n) are all found and
    /// report the false-positive rate on a disjoint range.
    #[arg(long)]
    check: bool,
}

fn bench(n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
        eprintln!("{} ns/key", timings.last().unwrap());
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "Min: {} Median: {} Max: {} Average: {}",
        timings[0],
        timings[timings.len() / 2],
        timings.last().unwrap(),
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}

fn run<V: Values>(args: &Args, header: Header, data: V) -> Result<()> {
    let bit_width = header.bit_width;
    let filter = HemVFilter::from_parts(header, data);

    if args.check {
        let missing = (0..args.n)
            .filter(|&i| !filter.contains::<usize>(i))
            .count();
        let fp = (0..args.n)
            .filter(|&i| filter.contains::<usize>(usize::MAX - i))
            .count();
        eprintln!(
            "Keys: {} Missing: {} FP rate: {} (expected {})",
            args.n,
            missing,
            fp as f64 / args.n as f64,
            1.0 / (1u64 << bit_width) as f64
        );
        if missing != 0 {
            std::process::exit(1);
        }
        return Ok(());
    }

    bench(args.n, args.repeats, || {
        let mut u = 0usize;
        let mut key: usize = 0;
        for _ in 0..args.n {
            key = key.wrapping_add(INCR);
            u ^= filter.contains::<usize>(key) as usize;
        }
        std::hint::black_box(u);
    });
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut r = BufReader::new(std::fs::File::open(&args.filter)?);
    let header = read_header(&mut r)?;
    let num_bytes = header.num_bytes;

    // Rebuild the same backend the builder used, so that the query path is the
    // one VFilter would take for this width.
    match header.bit_width {
        8 => {
            let data = Aligned::<u8>::from_boxed(read_boxed::<u8>(&mut r, num_bytes)?);
            run(&args, header, data)
        }
        16 => {
            let data = Aligned::<u16>::from_boxed(read_boxed::<u16>(&mut r, num_bytes)?);
            run(&args, header, data)
        }
        32 => {
            let data = Aligned::<u32>::from_boxed(read_boxed::<u32>(&mut r, num_bytes)?);
            run(&args, header, data)
        }
        b if !header.unaligned => {
            let words = read_boxed::<u64>(&mut r, num_bytes)?;
            let n = header.num_values;
            let data = Packed::from_parts(words, b, n);
            run(&args, header, data)
        }
        b => {
            let words = read_boxed::<u64>(&mut r, num_bytes)?;
            let n = header.num_values;
            let data = PackedU::from_parts(words, b, n);
            run(&args, header, data)
        }
    }
}

/// Reads `num_bytes` bytes into a boxed slice of `W`.
///
/// The bytes are read directly into a freshly allocated, correctly aligned
/// slice, so that, unlike reading into a byte buffer and then copying, the
/// memory used is just that of the values: this matters when many instances
/// run in parallel.
fn read_boxed<W: Copy + Default>(r: &mut impl Read, num_bytes: usize) -> Result<Box<[W]>> {
    assert!(num_bytes % size_of::<W>() == 0, "misaligned value array");
    let mut out = vec![W::default(); num_bytes / size_of::<W>()].into_boxed_slice();
    // SAFETY: the slice has exactly num_bytes bytes, and any bit pattern is a
    // valid unsigned integer.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, num_bytes) };
    r.read_exact(bytes)?;
    Ok(out)
}
