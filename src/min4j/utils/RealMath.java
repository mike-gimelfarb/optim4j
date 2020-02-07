package min4j.utils;

/**
 *
 * @author Michael
 */
public final class RealMath {

	/**
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	public static final double hypot(final double a, final double b) {
		final double absa = Math.abs(a);
		final double absb = Math.abs(b);
		double r = 0.0;
		if (absa > absb) {
			r = b / a;
			r = absa * Math.sqrt(1.0 + r * r);
		} else if (b != 0.0) {
			r = a / b;
			r = absb * Math.sqrt(1.0 + r * r);
		}
		return r;
	}

	/**
	 *
	 * @param x
	 * @param y
	 * @return
	 */
	public static final double pow(double x, int y) {

		// negative power
		if (y < 0) {
			return pow(1.0 / x, -y);
		}

		// trivial cases
		switch (y) {
		case 0:
			return 1.0;
		case 1:
			return x;
		case 2:
			return x * x;
		default:
			break;
		}

		// non trivial case
		double res = 1.0;
		while (y != 0) {
			switch (y & 1) {
			case 0:
				x *= x;
				y >>>= 1;
				break;
			default:
				res *= x;
				--y;
				break;
			}
		}
		return res;
	}

	/**
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	public static final double sign(final double a, final double b) {
		return b >= 0.0 ? Math.abs(a) : -Math.abs(a);
	}

	/**
	 *
	 * @param xs
	 * @return
	 */
	public static final double maxAbs(final double... xs) {
		double max = 0.0;
		for (final double x : xs) {
			max = Math.max(max, Math.abs(x));
		}
		return max;
	}

	/**
	 *
	 * @param x
	 * @return
	 */
	public static final int roundInt(final double x) {
		return (int) Math.round(x);
	}

	/**
	 * 
	 * @param x
	 * @return
	 */
	public static final long roundLong(final double x) {
		return (long) Math.round(x);
	}

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	private RealMath() {
	}
}
