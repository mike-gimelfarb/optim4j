package utils;

/**
 *
 * @author Michael
 */
public final class Constants {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	/**
	 * The machine epsilon.
	 */
	public static final double EPSILON = Math.ulp(1.0);

	/**
	 * The closest {@code double} value to the square root of two.
	 */
	public static final double SQRT2 = Math.sqrt(2.0);

	/**
	 * The closest {@code double} value to the square root of three.
	 */
	public static final double SQRT3 = Math.sqrt(3.0);

	/**
	 * The closest {@code double} value to the square root of five.
	 */
	public static final double SQRT5 = Math.sqrt(5.0);

	/**
	 * The closest {@code double} value to the golden ratio, which has the exact
	 * analytic expression {@code (sqrt(5) + 1) / 2}.
	 */
	public static final double GOLDEN = (SQRT5 + 1.0) / 2.0;

	/**
	 * The closest {@code double} value to the Euler's constant e.
	 */
	public static final double E = Math.E;

	/**
	 * The closest {@code double} value to the constant &pi, which represents the
	 * ratio of the circumference to the diameter of any circle.
	 */
	public static final double PI = Math.PI;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	private Constants() {
	}
}
