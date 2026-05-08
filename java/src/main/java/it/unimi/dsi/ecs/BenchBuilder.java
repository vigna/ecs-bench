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
