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

import java.util.Arrays;
import java.util.function.Function;

import opt.linesearch.LineSearch;
import opt.linesearch.MoreThuenteLineSearch;
import utils.BlasMath;
import utils.Constants;

/**
 *
 */
public final class LBFGSAlgorithm extends GradientOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int mySize, myMaxEvals;
	private final LineSearch myLineSearch;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param maxMemorySize
	 * @param maxEvaluations
	 * @param lineSearch
	 */
	public LBFGSAlgorithm(final double tolerance, final int maxMemorySize, final int maxEvaluations,
			final LineSearch lineSearch) {
		super(tolerance);
		mySize = maxMemorySize;
		myMaxEvals = maxEvaluations;
		myLineSearch = lineSearch;
	}

	/**
	 * 
	 * @param tolerance
	 * @param maxMemorySize
	 * @param maxEvaluations
	 */
	public LBFGSAlgorithm(final double tolerance, final int maxMemorySize, final int maxEvaluations) {
		this(tolerance, maxMemorySize, maxEvaluations, new MoreThuenteLineSearch(Constants.EPSILON * 10.0, 40));
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess) {

		// prepare variables
		final int[] fev = new int[1];
		final int[] dfev = new int[1];

		// call main subroutine
		final int n = guess.length;
		final double[] result = lbfgs(f, df, myLineSearch, n, mySize, guess, fev, dfev, myMaxEvals, myTol);
		myEvals += fev[0];
		myGEvals += dfev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double[] lbfgs(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final LineSearch lsrch, final int n, final int m,
			final double[] guess, final int[] fev, final int[] dfev, final int maxfev, final double tol) {

		// INITIALIZE MEMORY CONTAINING THE HESSIAN MODEL
		final double[][] hdx = new double[m][];
		final double[][] hdg = new double[m][];
		final double[] rho = new double[m];
		final double[] alpha = new double[m];
		final int[] actm = new int[1];
		actm[0] = 0;

		// INITIALIZE THE POSITION AND GRADIENT INFORMATION
		double[] x = Arrays.copyOf(guess, n);
		double[] g = df.apply(x);
		double[] hg = g;
		double[] d = new double[n];
		double[] q = new double[n];
		double fx = f.apply(x);
		fev[0] = dfev[0] = 1;

		while (true) {

			// PREPARATION FOR THE LINE SEARCH
			for (int j = 0; j < n; ++j) {
				d[j] = -hg[j];
			}

			// COMPUTE THE INITIAL STEP SIZE
			final double alf0 = 1.0;

			// CALL THE LINE SEARCH ROUTINE
			lsrch.resetCounter();
			final double[] x1 = lsrch.lineSearch(f, df, x, d, g, fx, alf0).second();
			fev[0] += lsrch.countEvaluations();
			dfev[0] += lsrch.countGradientEvaluations();
			if (x1 == null) {
				break;
			}

			// PERFORM THE LIMITED-MEMORY BFGS UPDATES
			final double[] g1 = df.apply(x1);
			bfgsUpdate(n, actm, m, x1, x, g1, g, hdx, hdg, rho);
			bfgsCompute(n, actm[0], hdx, hdg, g1, q, hg, rho, alpha);
			x = x1;
			g = g1;
			fx = f.apply(x1);
			++fev[0];
			++dfev[0];

			// CHECK CONVERGENCE IN THE GRADIENT VALUES
			final double gnorm = BlasMath.denorm(n, g);
			if (gnorm != gnorm || fev[0] >= maxfev) {
				break;
			} else if (gnorm <= (1.0 + Math.abs(fx)) * tol) {
				return x;
			}
		}
		return null;
	}

	private static void bfgsUpdate(final int n, final int[] actm, final int m, final double[] x, final double[] xp,
			final double[] g, final double[] gp, final double[][] hdx, final double[][] hdg, final double[] rho) {
		if (actm[0] == m) {

			// shift the memory
			for (int j = 1; j <= m - 1; ++j) {
				hdx[j - 1] = hdx[j];
				hdg[j - 1] = hdg[j];
				rho[j - 1] = rho[j];
			}
		} else {
			++actm[0];
		}

		// update history of the hessian approximation
		hdx[actm[0] - 1] = new double[n];
		hdg[actm[0] - 1] = new double[n];
		rho[actm[0] - 1] = 0.0;
		for (int j = 0; j < n; ++j) {
			hdx[actm[0] - 1][j] = x[j] - xp[j];
			hdg[actm[0] - 1][j] = g[j] - gp[j];
		}
		rho[actm[0] - 1] += BlasMath.ddotm(n, hdx[actm[0] - 1], 1, hdg[actm[0] - 1], 1);
	}

	private static void bfgsCompute(final int n, final int actm, final double[][] hdx, final double[][] hdg,
			final double[] arg, final double[] q, final double[] hg, final double[] rho, final double[] alpha) {

		// forward pass
		System.arraycopy(arg, 0, q, 0, n);
		if (actm == 0) {

			// without history we assume H*x = I*x
			System.arraycopy(arg, 0, hg, 0, n);
			return;
		}
		for (int k = actm - 1; k >= 0; --k) {
			final double hdxq = BlasMath.ddotm(n, hdx[k], 1, q, 1) / rho[k];
			alpha[k] = hdxq;
			BlasMath.daxpym(n, -hdxq, hdg[k], 1, q, 1);
		}

		// center update
		final double hdgsq = BlasMath.ddotm(n, hdg[actm - 1], 1, hdg[actm - 1], 1);
		final double hdgdx = BlasMath.ddotm(n, hdg[actm - 1], 1, hdx[actm - 1], 1);
		for (int j = 0; j < n; ++j) {
			hg[j] = q[j] * hdgdx / hdgsq;
		}

		// backward pass
		for (int k = 0; k <= actm - 2; ++k) {
			final double beta = BlasMath.ddotm(n, hdg[k], 1, hg, 1) / rho[k];
			BlasMath.daxpym(n, alpha[k] - beta, hdx[k], 1, hg, 1);
		}
	}
}
