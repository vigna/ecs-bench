//! HEM-sharded [`VFilter`](sux::dict::VFilter): the sharding ablation for the
//! ε-cost sharding paper.
//!
//! This crate builds a static filter that is *identical* to `VFilter` in every
//! respect but one: shards are laid out as in the Heuristic External Memory
//! scheme of Botelho, Pagh and Ziviani rather than by ε-cost sharding. In
//! particular it uses
//!
//! - the same 128-bit signatures ([`ToSig`], same seed, same hash);
//! - the same shard selection (the high bits of the first signature word);
//! - the same fuse 3-hypergraph edge generation, bit for bit (see
//!   [`local_edge`], which mirrors `edge_64` of `sux::func::shard_edge`);
//! - beyond 2·10⁷ keys, the same expansion factor ([`Fuse3NoShards::c`]) and
//!   the same segment-size law ([`Fuse3Shards::log2_seg_size`]);
//! - the same value backends (a slice of `u8`/`u16`/`u32` for power-of-two
//!   widths, an unaligned bit-packed array otherwise);
//! - by default, the same number of shards that `VFilter` would choose for the
//!   same key set, so that the comparison isolates the layout and nothing else.
//!
//! What differs is the layout, and it differs exactly as HEM prescribes:
//!
//! - shards have their *natural*, unequal sizes, so each one needs a different
//!   number of vertices; the start of each shard is therefore stored in an
//!   array of offsets, which a query must load;
//! - each shard carries a *local seed*, so that a shard whose graph fails to
//!   peel can be retried on its own instead of restarting the construction;
//!   the seed is stored in the high bits of the offset word, as in Sux4J.
//!
//! A query must thus load the offset word of its shard before it can generate
//! its edge. That load, and the dependency it introduces in the chain of
//! operations, is what ε-cost sharding removes and what this crate measures.
//!
//! # What is *not* identical
//!
//! - Augmented peeling is not implemented: a shard that fails to peel is
//!   retried with a different local seed. Up to 2·10⁷ keys `VFilter` solves
//!   its shards by augmented peeling, which works below the peeling threshold;
//!   there, we peel the same shards with the expansion factor and the
//!   segment-size law above, so our graphs are larger (by 1.8–4.2% between 2²⁰
//!   and 2²⁴ keys) and may have larger segments. A query performs the same
//!   operations on either graph, so the comparison still isolates the layout.
//!   The difference grows as the key set shrinks, and this crate refuses to
//!   build fewer than a million keys.
//! - The local seed remixes the local signature by multiplying it by an odd
//!   multiplier derived from the seed. Sux4J instead re-runs SpookyHash on the
//!   whole signature, which is far more expensive, so the query cost measured
//!   here is a *lower bound* on the cost of a faithful HEM query. Some remix
//!   is intrinsic to HEM — a local seed that did not change the edges would be
//!   useless — and this is the cheapest form we could find that works.
//!
//!   Two weaker remixes do *not* work, and both fail in the same instructive
//!   way. A signature contributes only ⌈lg 𝓁s³⌉ bits to its edge (46 of the 64
//!   in a typical shard), so two signatures that collide agree exactly on
//!   those bits and may differ arbitrarily on the rest. XORing the seed in
//!   preserves every bit-agreement, so the two keep colliding at every seed.
//!   Multiplying by a *small* odd number such as 2·seed + 1 is barely better:
//!   the difference of the two signatures is confined to the bits the edge
//!   ignores, and multiplying it by a small number leaves it confined, so the
//!   agreement survives. A multiplier with entropy across the whole word
//!   spreads that difference over all 64 bits and does separate them.
//!
//!   The multiplier is 1 when the seed is zero, so a shard that peels at the
//!   first attempt has bit-for-bit the edges and stored hashes of `VFilter`.
//! - At the coarse shard counts `VFilter` chooses, rounding the vertex count of
//!   a shard up to a whole number of segments often absorbs the (sub-percent)
//!   spread in shard sizes, so the stored offsets can come out equally spaced.
//!   The structure stores and loads them regardless, which is the cost being
//!   measured; the builder reports how many distinct vertex counts it produced
//!   so that this can be checked. A corollary is that at these shard counts HEM
//!   buys no space back: it pays the offset array for nothing.
//! - The segment size is global rather than per shard. Shard sizes differ by a
//!   few percent, and the segment-size law is a step function of the size, so
//!   in practice every shard would receive the same segment size anyway; making
//!   it global lets a query recover the per-shard segment count from the two
//!   offsets alone, exactly as Sux4J recovers a chunk's variable count.

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;
use std::io::{Read, Write};
use sux::func::shard_edge::{Fuse3NoShards, Fuse3Shards, FuseAug3Shards, ShardEdge};
use sux::utils::ToSig;

/// Bits of the offset word reserved for the vertex offset. The remaining bits
/// hold the local seed, as in Sux4J's `offsetAndSeed` array.
const OFFSET_BITS: u32 = 48;
const OFFSET_MASK: u64 = (1u64 << OFFSET_BITS) - 1;

/// Maximum number of local seeds tried before a shard is declared unsolvable.
/// The seed is stored in the 64 − [`OFFSET_BITS`] bits above the offset.
const MAX_LOCAL_SEEDS: u64 = 1 << 12;

/// The smallest key set we build. Below 2·10⁷ keys `VFilter` solves its shards
/// by augmented peeling, which we do not implement, and the difference between
/// its expansion factor and ours grows as the key set shrinks.
const MIN_KEYS: usize = 1_000_000;

/// The finalization step of Austin Appleby's MurmurHash3, as in
/// `sux::utils::mix64` (which is crate-private).
#[inline(always)]
pub const fn mix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^= k >> 33;
    k
}

/// Returns the odd multiplier a local seed induces.
///
/// It is 1 for seed zero, so that an unretried shard uses `VFilter`'s own local
/// signature, and otherwise has entropy across the whole word, which is what
/// lets a retry separate two signatures that collided on an edge.
#[inline(always)]
pub const fn seed_multiplier(seed: u64) -> u64 {
    seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
}

/// Returns the three vertices of the fuse-graph edge of a local signature,
/// relative to the start of the shard.
///
/// This is `edge_64` of `sux::func::shard_edge` with `shard` fixed to zero: the
/// shard displacement, which there is the arithmetic `shard * (l + 2) <<
/// log2_seg_size`, is here the loaded offset, and is added by the caller.
#[inline(always)]
pub fn local_edge(log2_seg_size: u32, l: u32, sig: u64) -> [usize; 3] {
    let v0 = ((sig as u128 * (((l as u64) << log2_seg_size) as u128)) >> 64) as usize;
    let seg_size = 1usize << log2_seg_size;
    let seg_size_mask = seg_size - 1;

    let mut v1 = v0 + seg_size;
    v1 ^= (sig as usize) & seg_size_mask;
    let mut v2 = v1 + seg_size;
    v2 ^= ((sig >> log2_seg_size) as usize) & seg_size_mask;
    [v0, v1, v2]
}

// =====================================================================
//  Value storage
// =====================================================================

/// A slice of `b`-bit values.
///
/// The two implementations mirror the two backends `VFilter` uses: a slice of
/// the unsigned type of the right width when `b` is 8, 16 or 32, and an
/// unaligned bit-packed array otherwise.
pub trait Values: Send + Sync {
    /// Returns the value of the given index.
    ///
    /// # Safety
    ///
    /// The index must be smaller than the length of the slice.
    unsafe fn get(&self, index: usize) -> u64;
    /// Sets the value of the given index.
    fn set(&mut self, index: usize, value: u64);
    /// Returns the number of values.
    fn len(&self) -> usize;
    /// Returns whether the slice is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the backing bytes, for serialization.
    fn as_bytes(&self) -> &[u8];
}

/// A slice of values of a power-of-two bit width, stored one per word.
pub struct Aligned<W> {
    data: Box<[W]>,
}

macro_rules! impl_aligned {
    ($($w:ty),*) => {$(
        impl Aligned<$w> {
            pub fn new(len: usize) -> Self {
                Self { data: vec![0 as $w; len].into_boxed_slice() }
            }

            pub fn from_boxed(data: Box<[$w]>) -> Self {
                Self { data }
            }
        }

        impl Values for Aligned<$w> {
            #[inline(always)]
            unsafe fn get(&self, index: usize) -> u64 {
                unsafe { *self.data.get_unchecked(index) as u64 }
            }

            #[inline(always)]
            fn set(&mut self, index: usize, value: u64) {
                self.data[index] = value as $w;
            }

            fn len(&self) -> usize {
                self.data.len()
            }

            fn as_bytes(&self) -> &[u8] {
                // SAFETY: any bit pattern of an unsigned integer is a valid
                // sequence of bytes, and the lifetime is that of self.
                unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const u8,
                        std::mem::size_of_val(&*self.data),
                    )
                }
            }
        }
    )*};
}

impl_aligned!(u8, u16, u32);

/// A bit-packed slice of values of arbitrary width, read one or two words at a
/// time.
///
/// The extraction code is that of `BitFieldVec::get_value_unchecked`, which is
/// what `VFilter` uses for non-power-of-two widths unless it is asked for
/// unaligned reads; see [`PackedU`] for that case. A padding word is kept at
/// the end so that both readers stay inside the allocation.
pub struct Packed {
    data: Box<[u64]>,
    bit_width: u32,
    mask: u64,
    len: usize,
}

impl Packed {
    pub fn new(bit_width: u32, len: usize) -> Self {
        assert!(bit_width > 0 && bit_width <= 58, "unsupported bit width");
        // One padding word, as in BitFieldVec::try_into_unaligned.
        let words = (len * bit_width as usize).div_ceil(64) + 1;
        Self {
            data: vec![0u64; words].into_boxed_slice(),
            bit_width,
            mask: (1u64 << bit_width) - 1,
            len,
        }
    }

    pub fn from_parts(data: Box<[u64]>, bit_width: u32, len: usize) -> Self {
        Self {
            data,
            bit_width,
            mask: (1u64 << bit_width) - 1,
            len,
        }
    }
}

impl Values for Packed {
    #[inline(always)]
    unsafe fn get(&self, index: usize) -> u64 {
        let bit_width = self.bit_width as usize;
        let pos = index * bit_width;
        let word_index = pos / 64;
        let bit_index = pos % 64;
        // SAFETY: index is in bounds by contract, and the padding word makes
        // the second read legal when a value straddles two words.
        unsafe {
            if bit_index + bit_width <= 64 {
                (*self.data.get_unchecked(word_index) >> bit_index) & self.mask
            } else {
                ((*self.data.get_unchecked(word_index) >> bit_index)
                    | (*self.data.get_unchecked(word_index + 1) << (64 - bit_index)))
                    & self.mask
            }
        }
    }

    #[inline(always)]
    fn set(&mut self, index: usize, value: u64) {
        let pos = index * self.bit_width as usize;
        let word = pos / 64;
        let bit = pos % 64;
        let value = value & self.mask;
        self.data[word] &= !(self.mask << bit);
        self.data[word] |= value << bit;
        if bit + self.bit_width as usize > 64 {
            let done = 64 - bit;
            self.data[word + 1] &= !(self.mask >> done);
            self.data[word + 1] |= value >> done;
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: as in Aligned::as_bytes.
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                std::mem::size_of_val(&*self.data),
            )
        }
    }
}

/// Like [`Packed`], but read with a single unaligned load.
///
/// The extraction code is that of `BitFieldVec::get_unaligned_unchecked`, which
/// is what `VFilter` uses for non-power-of-two widths when it is asked for
/// unaligned reads (the harness passes `--unaligned` exactly then). The width
/// must satisfy the same constraint as there: at most 58, or exactly 60 or 64.
pub struct PackedU(Packed);

impl PackedU {
    pub fn new(bit_width: u32, len: usize) -> Self {
        assert!(
            bit_width <= 58 || bit_width == 60 || bit_width == 64,
            "bit width {bit_width} does not admit unaligned reads"
        );
        Self(Packed::new(bit_width, len))
    }

    pub fn from_parts(data: Box<[u64]>, bit_width: u32, len: usize) -> Self {
        Self(Packed::from_parts(data, bit_width, len))
    }
}

impl Values for PackedU {
    #[inline(always)]
    unsafe fn get(&self, index: usize) -> u64 {
        let start_bit = index * self.0.bit_width as usize;
        // SAFETY: index is in bounds by contract, and the padding word makes
        // the eight-byte read legal for every legal index.
        unsafe {
            let ptr = (self.0.data.as_ptr() as *const u8).add(start_bit / 8) as *const u64;
            (std::ptr::read_unaligned(ptr) >> (start_bit % 8)) & self.0.mask
        }
    }

    #[inline(always)]
    fn set(&mut self, index: usize, value: u64) {
        self.0.set(index, value);
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

// =====================================================================
//  The filter
// =====================================================================

/// A static filter using fuse 3-hypergraphs and HEM sharding.
pub struct HemVFilter<V: Values> {
    /// The seed of the key-to-signature hash.
    seed: u64,
    /// Shards are selected as `sig[0] >> shard_bits_shift >> 1`, exactly as in
    /// `FuseAug3Shards::shard`.
    shard_bits_shift: u32,
    /// The (global) base-2 logarithm of the segment size.
    log2_seg_size: u32,
    /// The number of bits of the hashes stored by the filter.
    bit_width: u32,
    /// Whether the bit-packed backend is read with unaligned loads. Mirrors
    /// VFilter's `--unaligned`; meaningless for power-of-two widths.
    unaligned: bool,
    /// The mask of the hashes stored by the filter.
    filter_mask: u64,
    /// `num_shards + 1` words: the vertex offset of each shard in the low
    /// [`OFFSET_BITS`] bits, its local seed in the remaining ones. This is the
    /// array HEM needs and ε-cost sharding does not.
    offset_and_seed: Box<[u64]>,
    /// The values.
    data: V,
}

impl<V: Values> HemVFilter<V> {
    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.offset_and_seed.len() - 1
    }

    /// Returns the number of vertices.
    pub fn num_vertices(&self) -> usize {
        (self.offset_and_seed[self.num_shards()] & OFFSET_MASK) as usize
    }

    /// Returns the number of bits of the hashes stored by the filter.
    pub fn bit_width(&self) -> u32 {
        self.bit_width
    }

    /// Returns the size in bits of the whole structure, offset array included.
    pub fn num_bits(&self) -> usize {
        self.data.as_bytes().len() * 8 + self.offset_and_seed.len() * 64
    }

    /// Returns the size in bits of the offset and seed array alone.
    pub fn metadata_bits(&self) -> usize {
        self.offset_and_seed.len() * 64
    }

    /// Returns whether the key with the given signature is likely in the set.
    ///
    /// This is `VFilter::contains_by_sig` with the ε-cost shard displacement
    /// replaced by the HEM offset load.
    #[inline(always)]
    pub fn contains_by_sig(&self, sig: [u64; 2]) -> bool {
        let shard = (sig[0] >> self.shard_bits_shift >> 1) as usize;
        // SAFETY: shard < num_shards by construction of shard_bits_shift, and
        // the array has num_shards + 1 entries.
        let (w, next) = unsafe {
            (
                *self.offset_and_seed.get_unchecked(shard),
                *self.offset_and_seed.get_unchecked(shard + 1),
            )
        };
        let offset = (w & OFFSET_MASK) as usize;
        let num_vertices = (next & OFFSET_MASK) as usize - offset;
        // The local seed lives in the bits above the offset; it is what lets a
        // single shard be retried on its own. The multiplier is odd, and is 1
        // when the seed is zero, so the local signature is then VFilter's own.
        let local_sig = sig[1].wrapping_mul(seed_multiplier(w >> OFFSET_BITS));
        let l = (num_vertices >> self.log2_seg_size) as u32 - 2;
        let e = local_edge(self.log2_seg_size, l, local_sig);
        let expected = mix64(local_sig) & self.filter_mask;
        // SAFETY: the vertices of a local edge are below num_vertices, and
        // offset + num_vertices is at most the length of the data.
        unsafe {
            (self.data.get(offset + e[0])
                ^ self.data.get(offset + e[1])
                ^ self.data.get(offset + e[2]))
                == expected
        }
    }

    /// Returns whether a key is likely in the set.
    #[inline(always)]
    pub fn contains<K: ?Sized + ToSig<[u64; 2]>>(&self, key: impl std::borrow::Borrow<K>) -> bool {
        self.contains_by_sig(K::to_sig(key, self.seed))
    }
}

// =====================================================================
//  Construction
// =====================================================================

/// Returns the number of high bits `VFilter` would use to shard `n` keys with
/// the given ε, by asking `VFilter`'s own sharding logic.
pub fn vfilter_shard_high_bits(n: usize, eps: f64) -> u32 {
    let mut shard_edge = FuseAug3Shards::default();
    shard_edge.set_up_shards(n, eps, Fuse3Shards::DUP_EDGE_RETRY_PROB);
    shard_edge.shard_high_bits()
}

/// Solves one shard by peeling, retrying with a fresh local seed on failure.
///
/// Returns the local seed that worked and the values of the shard's vertices.
fn solve_shard(
    sigs: &[[u64; 2]],
    num_vertices: usize,
    log2_seg_size: u32,
    filter_mask: u64,
) -> Option<(u64, Vec<u64>)> {
    let l = (num_vertices >> log2_seg_size) as u32 - 2;

    for seed_index in 0..MAX_LOCAL_SEEDS {
        // The seed is stored in the bits above the offset; the multiplier it
        // induces is the same one the query recomputes.
        let stored_seed = seed_index << OFFSET_BITS;
        let mult = seed_multiplier(seed_index);

        // Local signatures. Duplicates would give duplicate edges; since the
        // filter answers identically for them, we drop them, exactly as the
        // Sux4J filters do (and as VFilter does with its signatures).
        let mut local: Vec<u64> = sigs.iter().map(|s| s[1].wrapping_mul(mult)).collect();
        local.sort_unstable();
        local.dedup();

        let m = local.len();
        if m > u32::MAX as usize {
            return None;
        }

        // XOR-graph: per vertex, the degree and the XOR of the incident edges.
        let mut degree = vec![0u32; num_vertices];
        let mut xor_edge = vec![0u32; num_vertices];
        for (edge, &ls) in local.iter().enumerate() {
            for v in local_edge(log2_seg_size, l, ls) {
                degree[v] += 1;
                xor_edge[v] ^= edge as u32;
            }
        }

        // Peel: repeatedly strip an edge that has a vertex of degree one.
        let mut peeled: Vec<(u32, u8)> = Vec::with_capacity(m);
        let mut stack: Vec<u32> = (0..num_vertices as u32)
            .filter(|&v| degree[v as usize] == 1)
            .collect();

        while let Some(v) = stack.pop() {
            if degree[v as usize] != 1 {
                continue;
            }
            let edge = xor_edge[v as usize];
            let vertices = local_edge(log2_seg_size, l, local[edge as usize]);
            let side = vertices.iter().position(|&x| x as u32 == v).unwrap() as u8;
            peeled.push((edge, side));
            for x in vertices {
                degree[x] -= 1;
                xor_edge[x] ^= edge;
                if degree[x] == 1 {
                    stack.push(x as u32);
                }
            }
        }

        if peeled.len() != m {
            // A non-trivial 2-core. Retrying this shard alone, rather than
            // restarting the whole construction, is the one operational
            // advantage HEM has over ε-cost sharding.
            eprintln!(
                "shard of {m} keys failed to peel with local seed {seed_index} \
                 ({} edges left in the 2-core); retrying this shard alone",
                m - peeled.len()
            );
            continue;
        }

        // Assign values in reverse peeling order.
        let mut values = vec![0u64; num_vertices];
        for &(edge, side) in peeled.iter().rev() {
            let ls = local[edge as usize];
            let vertices = local_edge(log2_seg_size, l, ls);
            let mut value = mix64(ls) & filter_mask;
            for (i, v) in vertices.iter().enumerate() {
                if i != side as usize {
                    value ^= values[*v];
                }
            }
            values[vertices[side as usize]] = value;
        }

        return Some((stored_seed, values));
    }

    None
}

/// Builds a HEM-sharded filter over the keys `0`, `1`, …, `n` − 1.
///
/// `shard_high_bits` defaults to what `VFilter` would use for the same key set,
/// so that the two structures differ only in their layout.
pub fn build<V: Values>(
    n: usize,
    bit_width: u32,
    seed: u64,
    eps: f64,
    shard_high_bits: Option<u32>,
    unaligned: bool,
    make_values: impl Fn(usize) -> V,
) -> Result<HemVFilter<V>> {
    ensure!(
        n >= MIN_KEYS,
        "this crate does not implement augmented peeling, so it does not build \
         fewer than {MIN_KEYS} keys"
    );

    let shard_high_bits = shard_high_bits.unwrap_or_else(|| vfilter_shard_high_bits(n, eps));
    let num_shards = 1usize << shard_high_bits;
    let shard_bits_shift = 63 - shard_high_bits;

    // Signatures, exactly as VFilter computes them.
    let mut sigs: Vec<[u64; 2]> = (0..n)
        .into_par_iter()
        .map(|i| <usize as ToSig<[u64; 2]>>::to_sig(i, seed))
        .collect();

    // Count sort by shard: unlike ε-cost sharding, we accept whatever sizes
    // the hash gives us, which is why we will need the offset array.
    let shard_of = |sig: &[u64; 2]| (sig[0] >> shard_bits_shift >> 1) as usize;
    let mut sizes = vec![0usize; num_shards];
    for sig in &sigs {
        sizes[shard_of(sig)] += 1;
    }
    let mut starts = Vec::with_capacity(num_shards + 1);
    let mut acc = 0;
    for &s in &sizes {
        starts.push(acc);
        acc += s;
    }
    starts.push(acc);

    let mut sorted = vec![[0u64; 2]; n];
    {
        let mut cursor = starts.clone();
        for sig in sigs.drain(..) {
            let s = shard_of(&sig);
            sorted[cursor[s]] = sig;
            cursor[s] += 1;
        }
    }
    drop(sigs);

    let max_shard = *sizes.iter().max().unwrap();
    ensure!(sizes.iter().all(|&m| m > 0), "an empty shard was generated");

    // The segment-size law and the expansion factor are those VFilter uses
    // when it peels, that is, beyond 2·10⁷ keys.
    let log2_seg_size = Fuse3Shards::log2_seg_size(max_shard);
    let seg_size = 1usize << log2_seg_size;

    // Each shard gets the number of vertices its own size calls for, rounded
    // up to a whole number of segments. This is where HEM and ε-cost sharding
    // part company: here the shards differ in size, so their starting points
    // have to be stored.
    let mut num_vertices = Vec::with_capacity(num_shards);
    for &m in &sizes {
        let c = Fuse3NoShards::c(m);
        let v = ((c * m as f64).ceil() as usize).div_ceil(seg_size) * seg_size;
        num_vertices.push(v.max(3 * seg_size));
    }

    // Report the spread: HEM's shards are unequal by construction, and it is
    // that inequality that forces the offset array to exist.
    eprintln!(
        "Shard keys: min {} max {} ({:+.3}%); shard vertices: min {} max {}; distinct vertex counts: {}",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        100.0 * (*sizes.iter().max().unwrap() as f64 * num_shards as f64 / n as f64 - 1.0),
        num_vertices.iter().min().unwrap(),
        num_vertices.iter().max().unwrap(),
        num_vertices.iter().collect::<std::collections::HashSet<_>>().len(),
    );

    let mut offsets = Vec::with_capacity(num_shards + 1);
    let mut acc = 0usize;
    for &v in &num_vertices {
        offsets.push(acc);
        acc += v;
    }
    offsets.push(acc);
    ensure!(
        acc as u64 <= OFFSET_MASK,
        "too many vertices for a {OFFSET_BITS}-bit offset"
    );

    let filter_mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    // Solve the shards independently, as HEM does.
    let solved: Vec<Option<(u64, Vec<u64>)>> = (0..num_shards)
        .into_par_iter()
        .map(|s| {
            solve_shard(
                &sorted[starts[s]..starts[s + 1]],
                num_vertices[s],
                log2_seg_size,
                filter_mask,
            )
        })
        .collect();

    let mut data = make_values(acc);
    let mut offset_and_seed = Vec::with_capacity(num_shards + 1);
    for (s, solution) in solved.into_iter().enumerate() {
        let Some((local_seed, values)) = solution else {
            bail!("shard {s} could not be solved with any local seed");
        };
        offset_and_seed.push(offsets[s] as u64 | local_seed);
        for (i, v) in values.into_iter().enumerate() {
            data.set(offsets[s] + i, v);
        }
    }
    offset_and_seed.push(offsets[num_shards] as u64);

    Ok(HemVFilter {
        seed,
        shard_bits_shift,
        log2_seg_size,
        bit_width,
        unaligned,
        filter_mask,
        offset_and_seed: offset_and_seed.into_boxed_slice(),
        data,
    })
}

// =====================================================================
//  Serialization
// =====================================================================

const MAGIC: u64 = 0x48454d5f_56463031; // "HEM_VF01"

fn write_u64(w: &mut impl Write, x: u64) -> Result<()> {
    w.write_all(&x.to_le_bytes())?;
    Ok(())
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

impl<V: Values> HemVFilter<V> {
    /// Serializes the filter.
    pub fn serialize(&self, w: &mut impl Write) -> Result<()> {
        write_u64(w, MAGIC)?;
        write_u64(w, self.seed)?;
        write_u64(w, self.shard_bits_shift as u64)?;
        write_u64(w, self.log2_seg_size as u64)?;
        write_u64(w, self.bit_width as u64)?;
        write_u64(w, self.unaligned as u64)?;
        write_u64(w, self.offset_and_seed.len() as u64)?;
        for &x in &self.offset_and_seed {
            write_u64(w, x)?;
        }
        write_u64(w, self.data.len() as u64)?;
        let bytes = self.data.as_bytes();
        write_u64(w, bytes.len() as u64)?;
        w.write_all(bytes)?;
        Ok(())
    }
}

/// The header of a serialized filter, read before the values so that the
/// caller can pick the right backend.
pub struct Header {
    pub seed: u64,
    pub shard_bits_shift: u32,
    pub log2_seg_size: u32,
    pub bit_width: u32,
    pub unaligned: bool,
    pub offset_and_seed: Box<[u64]>,
    pub num_values: usize,
    pub num_bytes: usize,
}

/// Reads the header of a serialized filter.
pub fn read_header(r: &mut impl Read) -> Result<Header> {
    ensure!(read_u64(r)? == MAGIC, "not a HEM filter");
    let seed = read_u64(r)?;
    let shard_bits_shift = read_u64(r)? as u32;
    let log2_seg_size = read_u64(r)? as u32;
    let bit_width = read_u64(r)? as u32;
    let unaligned = read_u64(r)? != 0;
    let len = read_u64(r)? as usize;
    let mut offset_and_seed = Vec::with_capacity(len);
    for _ in 0..len {
        offset_and_seed.push(read_u64(r)?);
    }
    let num_values = read_u64(r)? as usize;
    let num_bytes = read_u64(r)? as usize;
    Ok(Header {
        seed,
        shard_bits_shift,
        log2_seg_size,
        bit_width,
        unaligned,
        offset_and_seed: offset_and_seed.into_boxed_slice(),
        num_values,
        num_bytes,
    })
}

impl<V: Values> HemVFilter<V> {
    /// Assembles a filter from a header and a value slice.
    pub fn from_parts(header: Header, data: V) -> Self {
        let filter_mask = if header.bit_width == 64 {
            u64::MAX
        } else {
            (1u64 << header.bit_width) - 1
        };
        Self {
            seed: header.seed,
            shard_bits_shift: header.shard_bits_shift,
            log2_seg_size: header.log2_seg_size,
            bit_width: header.bit_width,
            unaligned: header.unaligned,
            filter_mask,
            offset_and_seed: header.offset_and_seed,
            data,
        }
    }
}
