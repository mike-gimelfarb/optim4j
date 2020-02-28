/*
Copyright (c) 2020 Mike Gimelfarb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the > "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, > subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package utils;

/**
 *
 */
public final class Constants {

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

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	private Constants() {
	}
}
