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
package opt.univariate.order1;

import java.util.function.Function;

import opt.OptimizerSolution;
import opt.univariate.DerivativeOptimizer;

/**
 * TODO: find which paper this is based on!
 */
public final class SecantAlgorithm extends DerivativeOptimizer {

	/**
	 *
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public SecantAlgorithm(final double absoluteTolerance, final double relativeTolerance, final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	@Override
	public final OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f,
			final Function<? super Double, Double> df, final double a, final double b) {

		// prepare variables
		final int[] dfev = new int[1];
		final boolean[] converged = new boolean[1];

		// call main subroutine
		final double result = secantMin(df, a, b, myTol, myRelTol, myMaxEvals, dfev, converged);
		return new OptimizerSolution<>(result, 0, dfev[0], converged[0]);
	}

	private static double secantMin(final Function<? super Double, Double> dfunc, double a, double b, final double tol,
			final double reltol, final int maxfev, final int[] dfev, final boolean[] converged) {

		// generate two points
		double dfb = dfunc.apply(b);
		double x0 = a + (b - a) / 3.0;
		double df0 = dfunc.apply(x0);
		double x1 = a + 2.0 * (b - a) / 3.0;
		double df1 = dfunc.apply(x1);
		dfev[0] = 3;
		boolean secant = false;

		// main loop of secant method for f'
		while (true) {

			final double mid = a + 0.5 * (b - a);
			double x2;
			final boolean secant1;
			if (df1 == 0.0 || df1 == -0.0) {

				// first-order condition is satisfied
				converged[0] = true;
				return x1;
			}

			// estimate the second derivative
			final double d2f = (df1 - df0) / (x1 - x0);
			if (d2f == 0.0 || d2f == -0.0) {

				// f" estimate is zero, use bisection instead
				x2 = mid;
				secant1 = false;
			} else {

				// attempt a secant step
				x2 = x1 - df1 / d2f;

				// accept or reject secant step
				if (x2 <= a || x2 >= b) {
					x2 = mid;
					secant1 = false;
				} else {
					secant1 = true;
				}
			}

			// test for budget
			if (dfev[0] >= maxfev) {
				return x1;
			}

			// test sufficient reduction in the size of the bracket
			double xtol = tol + reltol * Math.abs(mid);
			final double df2 = dfunc.apply(x2);
			++dfev[0];
			if (Math.abs(b - a) <= xtol) {
				converged[0] = true;
				return x2;
			}

			// test assuming secant method was used
			if (secant1 && secant) {
				xtol = tol + reltol * Math.abs(x2);
				if (Math.abs(x2 - x1) <= xtol && Math.abs(df2) <= tol) {
					converged[0] = true;
					return x2;
				}
			}

			// update the bracket
			x0 = x1;
			x1 = x2;
			df0 = df1;
			df1 = df2;
			secant = secant1;
			if (df1 * dfb < 0.0) {
				a = x1;
			} else {
				b = x1;
				dfb = df1;
			}
		}
	}
}
