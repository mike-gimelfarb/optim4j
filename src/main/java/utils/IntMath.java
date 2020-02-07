package utils;

/**
 *
 * @author Michael
 */
public final class IntMath {

	// ==========================================================================
	// STATIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param n
	 * @return
	 */
	public static final int abs(final int n) {
		final int mask = n >> 31;
		return (n + mask) ^ mask;
	}

	/**
	 *
	 * @param n
	 * @return
	 */
	public static final long abs(final long n) {
		final long mask = n >> 63;
		return (n + mask) ^ mask;
	}

	/**
	 *
	 * @param x
	 * @param y
	 * @return
	 */
	public static final int average(final int x, final int y) {

		// Hacker's delight 2-5 (3)
		return (x & y) + ((x ^ y) >> 1);
	}

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	private IntMath() {
	}
}
