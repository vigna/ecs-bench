//! Builds a HEM-sharded filter and serializes it.
//!
//! The command line mirrors that of `vfilter` in sux-rs, so that the harness
//! can drive the two the same way.

use anyhow::Result;
use clap::Parser;
use hem_vfilter::{Aligned, Packed, PackedU, Values, vfilter_shard_high_bits};
use std::io::BufWriter;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Builds a HEM-sharded VFilter and serializes it.", long_about = None)]
struct Args {
    /// The number of keys; the 64-bit keys [0 . . n) are used.
    #[arg(short, long)]
    n: usize,
    /// A name for the serialized filter.
    filter: String,
    /// The number of bits of the hashes used by the filter.
    #[arg(short, long, default_value_t = 8)]
    bits: u32,
    /// The seed of the key-to-signature hash.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// The ε used to pick the number of shards, when it is not given
    /// explicitly. The default matches that of VFilter.
    #[arg(long, default_value_t = 0.001)]
    eps: f64,
    /// The base-2 logarithm of the number of shards. The default is the number
    /// of shards VFilter would use for the same key set, which makes the
    /// comparison isolate the layout.
    #[arg(long)]
    shard_high_bits: Option<u32>,
    /// Read the bit-packed backend with unaligned loads, as VFilter does when
    /// given the same flag. Ignored for power-of-two widths.
    #[arg(long, short)]
    unaligned: bool,
}

fn run<V: Values>(args: &Args, make: impl Fn(usize) -> V) -> Result<()> {
    let start = Instant::now();
    let filter = hem_vfilter::build::<V>(
        args.n,
        args.bits,
        args.seed,
        args.eps,
        args.shard_high_bits,
        args.unaligned,
        make,
    )?;
    let elapsed = start.elapsed();

    eprintln!(
        "HEM (fuse3 + plain peeling); shards: 2^{}; vertices: {}",
        (filter.num_shards() as f64).log2() as u32,
        filter.num_vertices()
    );
    eprintln!(
        "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
        elapsed.as_secs_f64(),
        args.n,
        elapsed.as_nanos() as f64 / args.n as f64
    );
    // Bits/key includes the offset and seed array: that array is part of what a
    // HEM structure must store, and is exactly what ε-cost sharding saves.
    eprintln!(
        "Bits/key: {:.3}",
        filter.num_bits() as f64 / args.n as f64
    );
    eprintln!(
        "Offset/seed array: {} bytes ({:.5} bits/key)",
        filter.metadata_bits() / 8,
        filter.metadata_bits() as f64 / args.n as f64
    );

    let mut w = BufWriter::new(std::fs::File::create(&args.filter)?);
    filter.serialize(&mut w)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!(
        "Building {} keys, {} bits, 2^{} shards",
        args.n,
        args.bits,
        args.shard_high_bits
            .unwrap_or_else(|| vfilter_shard_high_bits(args.n, args.eps))
    );

    match (args.bits, args.unaligned) {
        (8, _) => run(&args, |n| Aligned::<u8>::new(n)),
        (16, _) => run(&args, |n| Aligned::<u16>::new(n)),
        (32, _) => run(&args, |n| Aligned::<u32>::new(n)),
        (b, false) => run(&args, move |n| Packed::new(b, n)),
        (b, true) => run(&args, move |n| PackedU::new(b, n)),
    }
}
