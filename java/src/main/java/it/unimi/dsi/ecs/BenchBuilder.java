package it.unimi.dsi.ecs;

import it.unimi.dsi.sux4j.mph.GOV3Function;
import it.unimi.dsi.sux4j.mph.GV3CompressedFunction;
import it.unimi.dsi.sux4j.mph.codec.Codec;
import it.unimi.dsi.bits.TransformationStrategies;
import it.unimi.dsi.fastutil.longs.LongArrayList;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.SplittableRandom;

public class BenchBuilder {

	/** The keys 0, 1, …, n&minus;1, little-endian, generated lazily.
	 *
	 * <p>Materializing them would cost about 34 GB at n = 2<sup>30</sup> (an
	 * object header and a reference per key), which is the size at which the
	 * paper's tables are built. Sux4J only iterates over the keys — twice, if
	 * a construction has to be retried — so a fresh iterator per call is
	 * enough. The sequence is identical to the materialized one.
	 */
	private static Iterable<byte[]> generateKeys(final long n) {
		return () -> new Iterator<byte[]>() {
			private final ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
			private long i = 0;

			@Override
			public boolean hasNext() {
				return i < n;
			}

			@Override
			public byte[] next() {
				if (i >= n) throw new NoSuchElementException();
				buf.clear();
				buf.putLong(i++);
				return buf.array().clone();
			}
		};
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
		final double u = rng.nextDouble();
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
			values.add(rng.nextLong(k));
		return values;
	}

	public static void main(final String[] args) throws IOException {
		if (args.length < 3) {
			System.err.println("Usage: BenchBuilder GOV3Function <n> <bits> <output>");
			System.err.println("       BenchBuilder GV3CompressedFunction <n> --geometric <output>");
			System.err.println("       BenchBuilder GV3CompressedFunction <n> --zipf <K> <output>");
			System.err.println("       BenchBuilder GV3CompressedFunction <n> --uniform <K> <output>");
			System.exit(1);
		}

		final String type = args[0];
		final long n = Long.parseLong(args[1]);
		final Iterable<byte[]> keys = generateKeys(n);

		if ("GOV3Function".equals(type)) {
			final int bits = Integer.parseInt(args[2]);
			final String output = args[3];

			System.err.println("Building GOV3Function: n=" + n + " bits=" + bits);
			long t0 = System.nanoTime();
			final GOV3Function<byte[]> f = new GOV3Function.Builder<byte[]>()
				.keys(keys)
				.transform(TransformationStrategies.rawByteArray())
				.dictionary(bits)
				.build();
			long elapsed = System.nanoTime() - t0;
			System.err.println("Construction completed in " + (elapsed / 1e9) + " seconds (" + n + " keys, " + ((double)elapsed / n) + " ns/key)");
			System.err.println("Bits/key: " + ((double) f.numBits() / n));

			System.err.println("Dumping to " + output);
			f.dump(output);

		} else if ("GV3CompressedFunction".equals(type)) {
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
			long t0 = System.nanoTime();
			final GV3CompressedFunction<byte[]> f = new GV3CompressedFunction.Builder<byte[]>()
				.keys(keys)
				.transform(TransformationStrategies.rawByteArray())
				.values(values)
				.codec(new Codec.Huffman(20, 1))
				.build();
			long elapsed = System.nanoTime() - t0;
			System.err.println("Construction completed in " + (elapsed / 1e9) + " seconds (" + n + " keys, " + ((double)elapsed / n) + " ns/key)");
			System.err.println("Bits/key: " + ((double) f.numBits() / n));

			System.err.println("Dumping to " + output);
			f.dump(output);

		} else {
			System.err.println("Unknown type: " + type + " (expected GOV3Function or GV3CompressedFunction)");
			System.exit(1);
		}

		System.err.println("Done.");
	}
}
