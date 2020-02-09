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
package opt.multivariate.unconstrained.order1;

import java.util.function.Function;

import opt.Optimizer;
import utils.BlasMath;
import utils.Constants;

/**
 *
 */
public abstract class GradientOptimizer extends Optimizer<double[], Double, Function<? super double[], Double>> {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	public static final int MAX_ITERS = (int) 1e6;

	protected static final double TINY = 1e-60;

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final double myTol;
	protected int myEvals, myGEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 */
	public GradientOptimizer(final double tolerance) {
		myTol = tolerance;
	}

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	/**
	 * 
	 * @param f
	 * @param df
	 * @param guess
	 * @return
	 */
	public abstract double[] optimize(Function<? super double[], Double> f, Function<? super double[], double[]> df,
			double[] guess);

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	/**
	 *
	 * @param f
	 * @param guess
	 * @return
	 */
	public double[] optimize(final Function<? super double[], Double> f, final double[] guess) {

		// use forward differencing of function
		final int[] fevs = new int[1];
		final Function<double[], double[]> df = (x) -> {
			final int n = x.length;
			final double fvec = f.apply(x);
			final double[] g = new double[n];
			dfdjcmod(f, n, x, fvec, g, Constants.EPSILON, new double[1]);
			fevs[0] += (n + 1);
			return g;
		};
		final double[] result = optimize(f, df, guess);
		myEvals += fevs[0];
		return result;
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @return
	 */
	public final int countEvaluations() {
		return myEvals;
	}

	/**
	 *
	 * @return
	 */
	public final int countGradientEvaluations() {
		return myGEvals;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static void dfdjcmod(final Function<? super double[], Double> fcn, final int n, final double[] x,
			final double fvec, final double[] fjac, final double epsfcn, final double[] wa) {
		final double epsmch = BlasMath.D1MACH[4 - 1];
		final double eps = Math.sqrt(Math.max(epsfcn, epsmch));
		for (int j = 1; j <= n; ++j) {
			final double temp = x[j - 1];
			double h = eps * Math.abs(temp);
			if (h == 0.0) {
				h = eps;
			}
			x[j - 1] = temp + h;
			wa[0] = fcn.apply(x);
			x[j - 1] = temp;
			fjac[j - 1] = (wa[0] - fvec) / h;
		}
	}
}
