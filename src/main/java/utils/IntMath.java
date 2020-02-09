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
