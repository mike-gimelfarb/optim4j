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
import utils.Constants;

/**
 * 
 * REFERENCES:
 * 
 * [1] Hager, W. W. "A derivative-based bracketing scheme for univariate
 * minimization and the conjugate gradient method." Computers & Mathematics with
 * Applications 18.9 (1989): 779-795.
 */
public final class CubicInterpolationAlgorithm extends DerivativeOptimizer {

	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public CubicInterpolationAlgorithm(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	@Override
	public final OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f,
			final Function<? super Double, Double> df, final double a, final double b) {

		// prepare variables
		final int[] fev = new int[1];
		final boolean[] converged = new boolean[1];

		// call main subroutine
		final double result = hybridcubic(f, df, a, b, myTol, myRelTol, myMaxEvals, fev, converged);
		return new OptimizerSolution<>(result, fev[0], fev[0], converged[0]);
	}

	private static double hybridcubic(final Function<? super Double, Double> func,
			final Function<? super Double, Double> dfunc, double a, double b, final double tau, final double reltol,
			final int maxfev, final int[] fev, final boolean[] converged) {

		// first convert the guess to a bracket
		double el, c, fc, dfc;
		double fa = func.apply(a);
		double dfa = dfunc.apply(a);
		fev[0] = 1;

		while (true) {

			// step 1 check convergence
			double mid = 0.5 * (a + b);
			double xtol = tau + reltol * Math.abs(mid);
			if (Math.abs(b - a) <= xtol) {
				converged[0] = true;
				return mid;
			}

			// check the budget
			if (fev[0] >= maxfev) {
				return mid;
			}

			// interpolation step
			el = 2.0 * Math.abs(b - a);
			double gamma = cubic(a, fa, dfa, b, func.apply(b), dfunc.apply(b));
			++fev[0];
			c = step(a, b, gamma, tau);
			fc = func.apply(c);
			dfc = dfunc.apply(c);
			++fev[0];
			double[] updt = update(a, fa, dfa, b, c, fc, dfc);
			a = updt[0];
			b = updt[1];
			fa = func.apply(a);
			dfa = dfunc.apply(a);
			++fev[0];

			while (true) {

				// step 2 check convergence
				mid = 0.5 * (a + b);
				xtol = tau + reltol * Math.abs(mid);
				if (Math.abs(b - a) <= xtol) {
					converged[0] = true;
					return mid;
				}

				// check budget
				if (fev[0] >= maxfev) {
					return mid;
				}

				el /= 2.0;
				if (Math.abs(c - a) > el || (dfc - dfa) / (c - a) <= 0.0) {
					break;
				} else {

					// step 4
					gamma = cubic(c, fc, dfc, a, fa, dfa);
					if (gamma < a || gamma > b) {
						break;
					} else {

						// interpolation step
						c = step(a, b, gamma, tau);
						fc = func.apply(c);
						dfc = dfunc.apply(c);
						++fev[0];
						updt = update(a, fa, dfa, b, c, fc, dfc);
						a = updt[0];
						b = updt[1];
						fa = func.apply(a);
						dfa = dfunc.apply(a);
						++fev[0];
					}
				}
			}

			// step 5 bisection step
			c = 0.5 * (a + b);
			fc = func.apply(c);
			dfc = dfunc.apply(c);
			++fev[0];
			updt = update(a, fa, dfa, b, c, fc, dfc);
			a = updt[0];
			b = updt[1];
			fa = func.apply(a);
			dfa = dfunc.apply(a);
			++fev[0];
		}
	}

	private static double[] update(final double a, final double fa, final double dfa, final double b, final double c,
			final double fc, final double dfc) {
		final double a1, b1;
		if (fc > fa) {
			a1 = a;
			b1 = c;
		} else if (fc < fa) {
			if (dfc * (a - c) <= 0.0) {
				a1 = c;
				b1 = a;
			} else {
				a1 = c;
				b1 = b;
			}
		} else if (dfc * (a - c) < 0.0) {
			a1 = c;
			b1 = a;
		} else if (dfa * (b - a) < 0.0) {
			a1 = a;
			b1 = c;
		} else {
			a1 = c;
			b1 = b;
		}
		return new double[] { a1, b1 };
	}

	private static double step(final double a, final double b, final double c, final double tau) {
		final double y = Math.min(a, b);
		final double z = Math.max(a, b);
		if (y + tau <= c && c <= z - tau) {
			return c;
		} else if (c > (a + b) / 2.0) {
			return z - tau;
		} else {
			return y + tau;
		}
	}

	private static double cubic(final double a, final double fa, final double da, final double b, final double fb,
			final double db) {
		final double delta = b - a;
		if (Math.abs(delta) <= Constants.EPSILON) {
			return a - 1.0;
		}
		final double v = da + db - 3.0 * (fb - fa) / delta;
		final double w;
		if (v * v - da * db > 0.0) {
			w = Math.signum(delta) * Math.sqrt(v * v - da * db);
		} else {
			w = 0.0;
		}
		return b - delta * (db + w - v) / (db - da + 2.0 * w);
	}
}
