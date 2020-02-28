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
package opt.univariate.order0;

import java.util.function.Function;

import opt.OptimizerSolution;
import opt.univariate.DerivativeFreeOptimizer;
import utils.Constants;

/**
 *
 */
public final class GoldenSectionAlgorithm extends DerivativeFreeOptimizer {

	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public GoldenSectionAlgorithm(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	@Override
	public final OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f, final double a,
			final double b) {

		// prepare variables
		final int[] fev = new int[1];
		final boolean[] converged = new boolean[1];

		// call main subroutine
		final double result = gssearch(f, a, b, myRelTol, myTol, myMaxEvals, fev, converged);
		return new OptimizerSolution<>(result, fev[0], 0, converged[0]);
	}

	private static double gssearch(final Function<? super Double, Double> f, final double a, final double b,
			final double rtol, final double atol, final int mfev, final int[] fev, final boolean[] converged) {

		// INITIALIZE CONSTANTS
		final double GOLD = Constants.GOLDEN;

		// INITIALIZE ITERATION VARIABLES
		double a1 = a;
		double b1 = b;
		double c = b1 - (b1 - a1) / GOLD;
		double d = a1 + (b1 - a1) / GOLD;

		// MAIN LOOP OF GOLDEN SECTION SEARCH
		while (fev[0] < mfev) {

			// CHECK CONVERGENCE
			final double mid = 0.5 * (c + d);
			final double tol = rtol * Math.abs(mid) + atol;
			if (Math.abs(c - d) <= tol) {
				converged[0] = true;
				return mid;
			}

			// evaluate at new points
			final double fc = f.apply(c);
			final double fd = f.apply(d);
			fev[0] += 2;

			// update interval
			if (fc < fd) {
				b1 = d;
			} else {
				a1 = c;
			}
			final double del = (b1 - a1) / GOLD;
			c = b1 - del;
			d = a1 + del;
		}

		// COULD NOT CONVERGE
		return 0.5 * (c + d);
	}
}
