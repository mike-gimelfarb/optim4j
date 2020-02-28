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

/**
 * 
 * REFERENCES:
 * 
 * [1] Antoniou, Andreas, and Wu-Sheng Lu. Practical optimization: algorithms
 * and engineering applications. Springer Science & Business Media, 2007.
 */
public final class DaviesSwannCampeyAlgorithm extends DerivativeFreeOptimizer {

	private final double myL;

	/**
	 *
	 * @param tolerance
	 * @param deltaDecayFactor
	 * @param maxEvaluations
	 */
	public DaviesSwannCampeyAlgorithm(final double tolerance, final double deltaDecayFactor, final int maxEvaluations) {
		super(tolerance, 0.0, maxEvaluations);
		myL = deltaDecayFactor;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public DaviesSwannCampeyAlgorithm(final double tolerance, final int maxEvaluations) {
		this(tolerance, 0.1, maxEvaluations);
	}

	@Override
	public OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f, final double a,
			final double b) {

		// prepare work arrays
		final int[] fev = new int[1];
		final boolean[] converged = new boolean[1];

		// call main subroutine
		final double result = dsc(f, a, b, myL, myTol, fev, myMaxEvals, converged);
		return new OptimizerSolution<>(result, fev[0], 0, converged[0]);
	}

	private static double dsc(final Function<? super Double, Double> f, final double a, final double b, final double K,
			final double tol, final int[] fev, final int maxfev, final boolean[] converged) {
		final double delta1 = 0.5 * (b - a);
		final double guess = 0.5 * (a + b);
		return dsc1(f, guess, a, b, delta1, K, tol, fev, maxfev, converged);
	}

	private static double dsc1(final Function<? super Double, Double> f, final double guess, final double a,
			final double b, final double delta1, final double K, final double tol, final int[] fev, final int maxfev,
			final boolean[] converged) {

		// step 1: initialization
		double x0 = guess;
		double delta = delta1;
		fev[0] = 0;

		// main loop of DSC algorithm
		while (true) {

			// step 2
			final double xm1 = x0 - delta;
			final double xp1 = x0 + delta;
			final double f0 = f.apply(x0);
			final double fp1 = f.apply(xp1);
			double p;
			fev[0] += 2;

			// step 3
			if (f0 > fp1) {
				p = 1.0;
			} else {
				final double fm1 = f.apply(xm1);
				++fev[0];
				if (fm1 < f0) {
					p = -1.0;
				} else {

					// step 7: update the position of the minimum
					final double num = delta * (fm1 - fp1);
					final double den = 2.0 * (fm1 - 2.0 * f0 + fp1);
					x0 += (num / den);
					x0 = Math.max(x0, a);
					x0 = Math.min(x0, b);
					if (delta <= tol) {
						converged[0] = true;
						return x0;
					} else {
						delta *= K;
						continue;
					}
				}
			}

			// step 4
			double twonm1 = 1.0;
			double fnm2 = f.apply(xm1);
			double xnm1 = x0;
			double fnm1 = f0;
			double xn;
			double fn;
			++fev[0];
			while (true) {
				xn = xnm1 + twonm1 * p * delta;
				fn = f.apply(xn);
				++fev[0];
				if (fn > fnm1) {
					break;
				} else {
					fnm2 = fnm1;
					xnm1 = xn;
					fnm1 = fn;
					twonm1 *= 2.0;
				}
				if (!Double.isFinite(xn)) {
					return x0;
				}
			}

			// step 5
			final double twonm2 = twonm1 / 2.0;
			final double xm = xnm1 + twonm2 * p * delta;
			final double fm = f.apply(xm);
			++fev[0];

			// step 6: update the position of the minimum
			if (fm >= fnm1) {
				final double num = twonm2 * p * delta * (fnm2 - fm);
				final double den = 2.0 * (fnm2 - 2.0 * fnm1 + fm);
				x0 = xnm1 + (num / den);
			} else {
				final double num = twonm2 * p * delta * (fnm1 - fn);
				final double den = 2.0 * (fnm1 - 2.0 * fm + fn);
				x0 = xm + (num / den);
			}
			x0 = Math.max(x0, a);
			x0 = Math.min(x0, b);

			// convergence test
			if (twonm2 * delta <= tol) {
				converged[0] = true;
				return x0;
			}
			if (fev[0] >= maxfev) {
				return x0;
			}
			delta *= K;
		}
	}
}
