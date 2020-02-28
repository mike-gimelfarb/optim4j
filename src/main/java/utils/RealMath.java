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

	private RealMath() {
	}
}
