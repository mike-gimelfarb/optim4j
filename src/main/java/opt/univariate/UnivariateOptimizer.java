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
package opt.univariate;

import java.util.function.Function;

import opt.Optimizer;

/**
 *
 */
public abstract class UnivariateOptimizer extends Optimizer<Double, Double, Function<? super Double, Double>> {

	/**
	 *
	 * @param f
	 * @param guess
	 * @param factor
	 * @param fev
	 * @return
	 */
	public static final double[] bracket(final Function<? super Double, Double> f, final double guess,
			final double factor, final int maxfev, final int[] fev) {
		fev[0] = 0;
		double a = guess;
		double fa = f.apply(a);
		++fev[0];
		double b = a + 1.0;
		double fb = f.apply(b);
		++fev[0];
		double c, fc;
		if (fa < fb) {
			c = a;
			fc = fa;
			a = b;
			b = c;
			fb = fc;
		}
		c = b + factor * (b - a);
		fc = f.apply(c);
		++fev[0];
		if (fc <= fb) {
			while (true) {
				final double d = c + factor * (c - b);
				if (Math.abs(d) >= 1e100 || fev[0] >= maxfev) {
					return null;
				}
				final double fd = f.apply(d);
				++fev[0];
				a = b;
				b = c;
				fb = fc;
				c = d;
				fc = fd;
				if (fc > fb) {
					break;
				}
			}
		}
		return new double[] { Math.min(a, c), Math.max(a, c) };
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected int myMaxEvals;
	protected double myTol, myRelTol;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public UnivariateOptimizer(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		myTol = absoluteTolerance;
		myRelTol = relativeTolerance;
		myMaxEvals = maxEvaluations;
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param newTolerance
	 */
	public final void setTolerance(final double newTolerance) {
		myTol = newTolerance;
	}
}
