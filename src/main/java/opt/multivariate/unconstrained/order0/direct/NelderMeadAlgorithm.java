/*
Original FORTRAN77 version by R ONeill.
FORTRAN90 version by John Burkardt.
Java version by Mike Gimelfarb.

This code is distributed under the GNU LGPL license.
*/
package opt.multivariate.unconstrained.order0.direct;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.GradientFreeOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.Sequences;

/**
 * A translation of the Nelder Mead algorithm by R. O'Neill (1979) for
 * optimization of a general multivariate function without constraints. A
 * modification suggested by Gao and Han (2012) to adapt the parameters of the
 * simplex that was shown to work better in larger dimensions is also
 * implemented.
 *
 * 
 * REFERENCES:
 * 
 * [1] Nelder, John A., and Roger Mead. "A simplex method for function
 * minimization." The computer journal 7.4 (1965): 308-313.
 * 
 * [2] O'Neill, R. T. "Algorithm AS 47: function minimization using a simplex
 * procedure." Journal of the Royal Statistical Society. Series C (Applied
 * Statistics) 20.3 (1971): 338-345.
 * 
 * [3] Gao, Fuchang, and Lixing Han. "Implementing the Nelder-Mead simplex
 * algorithm with adaptive parameters." Computational Optimization and
 * Applications 51.1 (2012): 259-277.
 */
public final class NelderMeadAlgorithm extends GradientFreeOptimizer {

	// algorithm constants
	private final double eps = 0.001;
	private final boolean myAdaptive;
	private final int myCheckEvery, myMaxEvals;
	private final double myRadius;

	// problem parameters
	private Function<? super double[], Double> myFunc;
	private int n;
	private double[] start;

	// algorithm temporaries
	private boolean converged;
	private int ihi, ilo, jcount, icount;
	private double ccoeff, ecoeff, rcoeff, scoeff, del, rq, y2star, ylo, ystar, ynewlo;
	private double[][] p;
	private double[] p2star, pbar, pstar, y, xmin, step;

	/**
	 *
	 * @param tolerance
	 * @param initialRadius
	 * @param checkEvery
	 * @param maxEvaluations
	 * @param adaptive
	 */
	public NelderMeadAlgorithm(final double tolerance, final double initialRadius, final int checkEvery,
			final int maxEvaluations, final boolean adaptive) {
		super(tolerance);
		myAdaptive = adaptive;
		myCheckEvery = checkEvery;
		myMaxEvals = maxEvaluations;
		myRadius = initialRadius;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialRadius
	 * @param checkEvery
	 * @param maxEvaluations
	 */
	public NelderMeadAlgorithm(final double tolerance, final double initialRadius, final int checkEvery,
			final int maxEvaluations) {
		this(tolerance, initialRadius, checkEvery, maxEvaluations, true);
	}

	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {

		// problem initialization
		n = guess.length;
		myFunc = func;
		start = Arrays.copyOf(guess, n);

		// parameters
		if (myAdaptive) {
			ccoeff = 0.75 - 0.5 / n;
			ecoeff = 1.0 + 2.0 / n;
			rcoeff = 1.0;
			scoeff = 1.0 - 1.0 / n;
		} else {
			ccoeff = 0.5;
			ecoeff = 2.0;
			rcoeff = 1.0;
			scoeff = 0.5;
		}

		// storage
		p = new double[n + 1][n];
		p2star = new double[n];
		pbar = new double[n];
		pstar = new double[n];
		y = new double[n + 1];
		xmin = new double[n];
		step = new double[n];
		Arrays.fill(step, myRadius);

		// Initialization.
		icount = 0;
		jcount = myCheckEvery;
		del = 1.0;
		rq = myTol * myTol * n;
		ynewlo = 0.0;
	}

	@Override
	public void iterate() {

		// YNEWLO is, of course, the HIGHEST value???
		converged = false;
		ihi = Sequences.argmax(y.length, y) + 1;
		ynewlo = y[ihi - 1];

		// Calculate PBAR, the centroid of the simplex vertices
		// excepting the vertex with Y value YNEWLO.
		for (int i = 1; i <= n; ++i) {
			double sum = 0.0;
			for (int k = 1; k <= n + 1; ++k) {
				sum += p[k - 1][i - 1];
			}
			sum -= p[ihi - 1][i - 1];
			sum /= n;
			pbar[i - 1] = sum;
		}

		// Reflection through the centroid.
		for (int k = 1; k <= n; ++k) {
			pstar[k - 1] = pbar[k - 1] + rcoeff * (pbar[k - 1] - p[ihi - 1][k - 1]);
		}
		ystar = myFunc.apply(pstar);
		++icount;

		// Successful reflection, so extension.
		if (ystar < ylo) {

			// Expansion.
			for (int k = 1; k <= n; ++k) {
				p2star[k - 1] = pbar[k - 1] + ecoeff * (pstar[k - 1] - pbar[k - 1]);
			}
			y2star = myFunc.apply(p2star);
			++icount;

			// Retain extension or contraction.
			if (ystar < y2star) {
				System.arraycopy(pstar, 0, p[ihi - 1], 0, n);
				y[ihi - 1] = ystar;
			} else {
				System.arraycopy(p2star, 0, p[ihi - 1], 0, n);
				y[ihi - 1] = y2star;
			}
		} else {

			// No extension.
			int l = 0;
			for (int i = 1; i <= n + 1; ++i) {
				if (ystar < y[i - 1]) {
					++l;
				}
			}
			if (1 < l) {

				// Copy pstar to the worst (HI) point.
				System.arraycopy(pstar, 0, p[ihi - 1], 0, n);
				y[ihi - 1] = ystar;
			} else if (l == 0) {

				// Contraction on the Y(IHI) side of the centroid.
				for (int k = 1; k <= n; ++k) {
					p2star[k - 1] = pbar[k - 1] + ccoeff * (p[ihi - 1][k - 1] - pbar[k - 1]);
				}
				y2star = myFunc.apply(p2star);
				++icount;

				// Contract the whole simplex.
				if (y[ihi - 1] < y2star) {
					for (int j = 1; j <= n + 1; ++j) {
						for (int k = 1; k <= n; ++k) {
							p[j - 1][k - 1] = scoeff * (p[j - 1][k - 1] + p[ilo - 1][k - 1]);
						}
						System.arraycopy(p[j - 1], 0, xmin, 0, n);
						y[j - 1] = myFunc.apply(xmin);
						++icount;
					}
					ilo = Sequences.argmin(n + 1, y) + 1;
					ylo = y[ilo - 1];
					converged = false;
					return;
				} else {

					// Retain contraction.
					System.arraycopy(p2star, 0, p[ihi - 1], 0, n);
					y[ihi - 1] = y2star;
				}
			} else if (l == 1) {

				// Contraction on the reflection side of the centroid.
				for (int k = 1; k <= n; ++k) {
					p2star[k - 1] = pbar[k - 1] + ccoeff * (pstar[k - 1] - pbar[k - 1]);
				}
				y2star = myFunc.apply(p2star);
				++icount;

				// Retain reflection?
				if (y2star <= ystar) {
					System.arraycopy(p2star, 0, p[ihi - 1], 0, n);
					y[ihi - 1] = y2star;
				} else {
					System.arraycopy(pstar, 0, p[ihi - 1], 0, n);
					y[ihi - 1] = ystar;
				}
			}
		}

		// Check if YLO improved.
		if (y[ihi - 1] < ylo) {
			ylo = y[ihi - 1];
			ilo = ihi;
		}
		--jcount;
		if (0 < jcount) {
			converged = false;
			return;
		}

		// Check to see if minimum reached.
		if (icount <= myMaxEvals) {
			jcount = myCheckEvery;
			double sum = 0.0;
			for (int k = 1; k <= n + 1; ++k) {
				sum += y[k - 1];
			}
			sum /= (n + 1.0);
			double sumsq = 0.0;
			for (int k = 1; k <= n + 1; ++k) {
				sumsq += (y[k - 1] - sum) * (y[k - 1] - sum);
			}
			if (sumsq <= rq) {
				converged = true;
			}
		}
	}

	@Override
	public MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func, final double[] guess) {

		// prepare variables
		initialize(func, guess);

		// call main subroutine
		final int ifault = nelmin();
		return new MultivariateOptimizerSolution(xmin, icount, 0, ifault == 0);
	}

	private int nelmin() {

		// Initial or restarted loop.
		while (true) {

			// Start of the restart.
			System.arraycopy(start, 0, p[n], 0, n);
			y[n + 1 - 1] = myFunc.apply(start);
			++icount;

			// Define the initial simplex.
			for (int j = 1; j <= n; ++j) {
				final double x = start[j - 1];
				start[j - 1] += step[j - 1] * del;
				System.arraycopy(start, 0, p[j - 1], 0, n);
				y[j - 1] = myFunc.apply(start);
				++icount;
				start[j - 1] = x;
			}

			// Find highest and lowest Y values. YNEWLO = Y(IHI) indicates
			// the vertex of the simplex to be replaced.
			ilo = Sequences.argmin(n + 1, y) + 1;
			ylo = y[ilo - 1];

			// Inner loop.
			while (icount < myMaxEvals) {
				iterate();
				if (converged) {
					break;
				}
			}

			// Factorial tests to check that YNEWLO is a local minimum.
			System.arraycopy(p[ilo - 1], 0, xmin, 0, n);
			ynewlo = y[ilo - 1];
			if (myMaxEvals < icount) {
				return 2;
			}
			int ifault = 0;
			for (int i = 1; i <= n; ++i) {
				del = step[i - 1] * eps;
				xmin[i - 1] += del;
				double z = myFunc.apply(xmin);
				++icount;
				if (z < ynewlo) {
					ifault = 2;
					break;
				}
				xmin[i - 1] -= (del + del);
				z = myFunc.apply(xmin);
				++icount;
				if (z < ynewlo) {
					ifault = 2;
					break;
				}
				xmin[i - 1] += del;
			}
			if (ifault == 0) {
				return ifault;
			}

			// Restart the procedure.
			System.arraycopy(xmin, 0, start, 0, n);
			del = eps;
		}
	}
}
