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
import utils.RealMath;

/**
 *
 */
public final class FibonacciSearchAlgorithm extends DerivativeFreeOptimizer {

	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public FibonacciSearchAlgorithm(final double absoluteTolerance, final double relativeTolerance,
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
		final double result = fibsearch(f, a, b, myTol, myRelTol, myMaxEvals, fev, converged);
		return new OptimizerSolution<>(result, fev[0], 0, converged[0]);
	}

	private static double fibsearch(final Function<? super Double, Double> f, final double a, final double b,
			final double abstol, final double reltol, final int maxfev, final int[] fev, final boolean[] converged) {

		// find the smallest n such that 1/F(n) < tolerance / (b - a)
		final double adjtol = abstol / (b - a);
		double fib1 = 1.0, fib2 = 1.0;
		int n = 2;
		while (1.0 / fib2 >= adjtol) {
			final double fib3 = fib1 + fib2;
			fib1 = fib2;
			fib2 = fib3;
			++n;
			if (n > maxfev) {
				return Double.NaN;
			}
		}

		// evaluate initial constants
		final double sqrt5 = Constants.SQRT5;
		final double c = (sqrt5 - 1.0) / 2.0;
		final double s = (1.0 - sqrt5) / (1.0 + sqrt5);
		double p1 = RealMath.pow(s, n);
		double p2 = p1 * s;
		double alpha = c * (1.0 - p1) / (1.0 - p2);
		double x1 = a;
		double x4 = b;
		double x3 = alpha * x4 + (1.0 - alpha) * x1;
		double f3 = f.apply(x3);
		fev[0] = 1;

		// main loop
		for (int i = 1; i <= n - 1; ++i) {

			// introduce new point X2
			final double x2;
			if (i == n - 1) {
				x2 = 0.01 * x1 + 0.99 * x3;
			} else {
				x2 = alpha * x1 + (1.0 - alpha) * x4;
			}
			final double f2 = f.apply(x2);
			++fev[0];

			// update the interval
			if (f2 < f3) {
				x4 = x3;
				x3 = x2;
				f3 = f2;
			} else {
				x1 = x4;
				x4 = x2;
			}

			// update the step size
			final int d = n - i;
			p1 = RealMath.pow(s, d);
			p2 = p1 * s;
			alpha = c * (1.0 - p1) / (1.0 - p2);

			// check tolerance
			final double mid = 0.5 * (x1 + x4);
			final double tol = reltol * Math.abs(mid) + abstol;
			if (Math.abs(x4 - x1) <= tol) {
				converged[0] = true;
				break;
			}

			// check budget
			if (fev[0] >= maxfev) {
				break;
			}
		}
		return 0.5 * (x1 + x4);
	}
}
