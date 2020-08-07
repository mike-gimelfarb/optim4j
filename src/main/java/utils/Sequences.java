package utils;

import java.util.Random;

public final class Sequences {

	public static final int argmin(final int len, final double[] data) {
		int imin = -1;
		double min = 0.0;
		for (int k = 0; k < data.length; ++k) {
			if (k >= len) {
				break;
			}
			if (k == 0 || data[k] < min) {
				min = data[k];
				imin = k;
			}
		}
		return imin;
	}

	public static final int argmax(final int len, final double[] data) {
		int imax = -1;
		double max = 0.0;
		for (int k = 0; k < data.length; ++k) {
			if (k >= len) {
				break;
			}
			if (k == 0 || data[k] > max) {
				max = data[k];
				imax = k;
			}
		}
		return imax;
	}

	public static final int sortedIndex(final double item, final int len, final double... data) {
		int i = 0;
		int j = len;
		while (i < j) {
			final int m = IntMath.average(i, j);
			if (data[m] < item) {
				i = m + 1;
			} else {
				j = m;
			}
		}
		return i;
	}

	@SafeVarargs
	public static final <T> void shuffle(final Random rand, final int i1, final int i2, final T... arr) {
		for (int i = i2; i > i1; --i) {
			final int index = rand.nextInt(i - i1 + 1) + i1;
			swap(arr, index, i);
		}
	}

	public static final <T> void swap(final T[] data, final int i, final int j) {
		if (i == j) {
			return;
		}
		final T temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}

	public static final void shuffle(final Random rand, final int i1, final int i2, final int... arr) {
		for (int i = i2; i > i1; --i) {
			final int index = rand.nextInt(i - i1 + 1) + i1;
			swap(arr, index, i);
		}
	}

	public static final void swap(final int[] data, final int i, final int j) {
		if (i == j) {
			return;
		}
		final int temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}

	public static final int[] range(final int end) {
		final int[] result = new int[end];
		for (int i = 0; i < end; ++i) {
			result[i] = i;
		}
		return result;
	}

	private Sequences() {
	}
}
