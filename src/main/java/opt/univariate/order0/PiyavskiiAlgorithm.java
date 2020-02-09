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

import utils.IntMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] Piyavskii, S. A. "An algorithm for finding the absolute extremum of a
 * function." USSR Computational Mathematics and Mathematical Physics 12.4
 * (1972): 57-67.
 * 
 * [2] Lera, Daniela, and Yaroslav D. Sergeyev. "Acceleration of univariate
 * global optimization algorithms working with Lipschitz functions and Lipschitz
 * first derivatives." SIAM Journal on Optimization 23.1 (2013): 508-529.
 */
public final class PiyavskiiAlgorithm extends DerivativeFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double myR;
	private final double myXi;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 * @param rParam
	 * @param xiParam
	 */
	public PiyavskiiAlgorithm(final double tolerance, final int maxEvaluations, final double rParam,
			final double xiParam) {
		super(tolerance, 0.0, maxEvaluations);
		myR = rParam;
		myXi = xiParam;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public PiyavskiiAlgorithm(final double tolerance, final int maxEvaluations) {
		this(tolerance, maxEvaluations, 1.4, 1e-6);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public double optimize(final Function<? super Double, Double> f, final double a, final double b) {

		// prepare variables
		final int[] fev = new int[1];

		// call main subroutine
		final double result = shubert(f, a, b, myTol, myMaxEvals, fev, myR, myXi);
		myEvals += fev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	public static double shubert(final Function<? super Double, Double> func, final double a, final double b,
			final double tol, final int maxiters, final int[] fev, final double r, final double xi) {
		final double[] xlist = new double[maxiters];
		final double[] zlist = new double[maxiters];
		final double[] llist = new double[maxiters];

		// first two trials
		xlist[0] = a;
		xlist[1] = b;
		zlist[0] = func.apply(a);
		zlist[1] = func.apply(b);
		fev[0] = 2;
		if (zlist[0] != zlist[0] || zlist[1] != zlist[1]) {
			return Double.NaN;
		}

		// main loop of adaptive Piyavskii algorithm
		int k = 2;
		while (fev[0] < maxiters && k < maxiters) {

			// compute the lipschitz constants
			double xmax = 0.0;
			double hmax = 0.0;
			for (int i = 0; i <= k - 2; ++i) {
				final double xnew = xlist[i + 1] - xlist[i];
				final double zdif = zlist[i + 1] - zlist[i];
				final double hnew = Math.abs(zdif) / xnew;
				xmax = Math.max(xmax, xnew);
				hmax = Math.max(hmax, hnew);
			}
			for (int i = 0; i <= k - 2; ++i) {
				final int ilo = Math.max(i - 1, 0);
				final int ihi = Math.min(i + 1, k - 2);
				double lambda = 0.0;
				for (int j = ilo; j <= ihi; ++j) {
					final double zdif = zlist[j + 1] - zlist[j];
					final double xdif = xlist[j + 1] - xlist[j];
					final double hnew = Math.abs(zdif) / xdif;
					lambda = Math.max(lambda, hnew);
				}
				final double xdif = xlist[i + 1] - xlist[i];
				final double gamma = hmax * xdif / xmax;
				llist[i] = r * Math.max(xi, Math.max(lambda, gamma));
			}

			// find an interval where the next trial will be executed
			double rmin = Double.POSITIVE_INFINITY;
			int t = -1;
			for (int i = 0; i <= k - 2; ++i) {
				final double zmid = 0.5 * (zlist[i + 1] + zlist[i]);
				final double xdif = 0.5 * (xlist[i + 1] - xlist[i]);
				final double rnew = zmid - llist[i] * xdif;
				if (rnew < rmin) {
					t = i;
					rmin = rnew;
				}
			}
			final double xleft = xlist[t];
			final double xright = xlist[t + 1];
			final double zleft = zlist[t];
			final double zright = zlist[t + 1];
			final double lip = llist[t];

			// execute the next trial point
			if (xright - xleft > tol) {

				// compute the next trial point
				final double xmid = 0.5 * (xright + xleft);
				final double zdif = 0.5 * (zleft - zright);
				final double xtry = xmid + (zdif / lip);
				final double ztry = func.apply(xtry);
				++fev[0];
				if (ztry != ztry) {
					break;
				}

				// determine where to insert the next point
				final int iins = sortedIndex(xtry, k, xlist);

				// insert the next trial point into memory
				if (iins == 0) {
					System.arraycopy(xlist, 0, xlist, 1, k);
					System.arraycopy(zlist, 0, zlist, 1, k);
				} else if (iins < k) {
					System.arraycopy(xlist, iins, xlist, iins + 1, k - iins);
					System.arraycopy(zlist, iins, zlist, iins + 1, k - iins);
				}
				xlist[iins] = xtry;
				zlist[iins] = ztry;
				++k;
			} else {

				// we have converged
				final int imin = argmin(k, zlist);
				return xlist[imin];
			}
		}
		return Double.NaN;
	}

	private static final int argmin(final int len, final double... data) {
		int k = 0;
		int imin = -1;
		double min = 0;
		for (final double t : data) {
			if (k >= len) {
				break;
			}
			if (k == 0 || t < min) {
				min = t;
				imin = k;
			}
			++k;
		}
		return imin;
	}

	private static final int sortedIndex(final double item, final int len, final double... data) {
		int i = 0;
		int j = len;
		while (i < j) {
			final int m = IntMath.average(i, j);
			if (data[m] < item) {
				i = m + 1;
			} else {
				j = m;
			}
		}
		return i;
	}
}
