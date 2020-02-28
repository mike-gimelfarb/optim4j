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
package opt.multivariate.unconstrained.order0.direct;

import java.util.Arrays;
import java.util.function.Function;

import opt.OptimizerSolution;
import opt.multivariate.GradientFreeOptimizer;
import utils.BlasMath;

/**
 *
 * REFERENCES:
 * 
 * [1] Rosenbrock, HoHo. "An automatic method for finding the greatest or least
 * value of a function." The Computer Journal 3.3 (1960): 175-184.
 * 
 * [2] Swann, W. H. (1964). Report on the Development of a new Direct Search
 * Method of Optimisation, Imperial Chemical Industries Ltd., Central Instrument
 * Laboratory Research Note 64/3.
 * 
 * [3] Palmer, J. R. "An improved procedure for orthogonalising the search
 * vectors in Rosenbrock's and Swann's direct search optimisation methods." The
 * Computer Journal 12.1 (1969): 69-71.
 * 
 * [4] Box, M. J.; Davies, D.; Swann, W. H. (1969). Non-Linear optimisation
 * Techniques. Oliver & Boyd.
 */
public final class RosenbrockAlgorithm extends GradientFreeOptimizer {

	private final double myRho, myStep0;
	private final int myMaxEvals;

	/**
	 *
	 * @param tolerance
	 * @param initialStepSize
	 * @param decreaseFactor
	 * @param maxEvaluations
	 */
	public RosenbrockAlgorithm(final double tolerance, final double initialStepSize, final double decreaseFactor,
			final int maxEvaluations) {
		super(tolerance);
		myStep0 = initialStepSize;
		myRho = decreaseFactor;
		myMaxEvals = maxEvaluations;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialStepSize
	 * @param maxEvaluations
	 */
	public RosenbrockAlgorithm(final double tolerance, final double initialStepSize, final int maxEvaluations) {
		this(tolerance, initialStepSize, 0.1, maxEvaluations);
	}

	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
		// nothing to do here
	}

	@Override
	public void iterate() {
		// nothing to do here
	}

	@Override
	public OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> func,
			final double[] guess) {

		// prepare variables
		final int n = guess.length;
		final double[] x0 = Arrays.copyOf(guess, n);
		final double[] x1 = new double[n];
		final int[] fun = new int[1];
		final int[] ierr = new int[1];

		// call main subroutine
		dsc(func, n, x0, myStep0, myRho, myTol, myMaxEvals, x1, fun, ierr);
		return new OptimizerSolution<>(x1, fun[0], 0, ierr[0] == 0);
	}

	private static void dsc(final Function<? super double[], Double> func, final int n, final double[] x0,
			final double step0, final double rho, final double eps, final int maxfev, final double[] x1,
			final int[] fev, final int[] ierr) {
		final double[][] v = new double[n + 2][n];
		final double[][] vold = new double[n + 2][n];
		final double[][] x = new double[n + 2][n];
		final double[] d = new double[n + 2];
		final double[] temp = new double[n + 1];
		final double[] wa = new double[1];
		int i, ii, jj, j;
		double zn, dxn, step, tmp;

		// INITIALIZATION
		System.arraycopy(x0, 0, x[0], 0, n);
		step = step0;
		for (j = 1; j <= n; ++j) {
			v[j][j - 1] = 1.0;
		}
		fev[0] = 0;
		i = 1;

		while (true) {

			// PERFORM A LINE SEARCH USING LAGRANGE QUADRATIC INTERPOLATION
			// SUGGESTED BY DAVIES, SWANN AND CAMPEY. SEE SWANN (1964) OR
			// BOX ET AL (1969)
			wa[0] = step;
			int err = line_search(func, n, x[i - 1], wa, v[i], x[i], fev, maxfev);
			d[i] = wa[0];

			// REACHED MAXIMUM NUMBER OF EVALUATIONS
			if (err != 0) {
				ierr[0] = 1;
				return;
			}

			// MAIN LOOP
			if (i < n) {
				++i;
				continue;
			}
			if (i == n) {

				// EVENTUALLY DO ONE MORE LINE SEARCH...
				BlasMath.daxpy1(n, -1.0, x[0], 1, x[n], 1, temp, 1);
				zn = BlasMath.denorm(n, temp);
				if (zn > 0.0) {
					BlasMath.dscal1(n, 1.0 / zn, temp, 1, v[n + 1], 1);
					i = n + 1;
					continue;
				} else {
					System.arraycopy(x[n], 0, x[n + 1], 0, n);
					d[n + 1] = 0.0;
				}
			} else {

				// CHECK APPROPRIATENESS OF STEP LENGTH
				dxn = 0.0;
				for (j = 1; j <= n; ++j) {
					tmp = x[n + 1][j - 1] - x[0][j - 1];
					dxn += (tmp * tmp);
				}
				dxn = Math.sqrt(dxn);
				if (dxn >= step) {

					// COPY THE OLD BASIS VECTORS
					for (ii = 1; ii <= n; ++ii) {
						System.arraycopy(v[ii], 0, vold[ii], 0, n);
					}

					// COMPUTE THE QUANTITIES SUM(I:N) D(I)^2 AND PLACE THEM
					// INTO AN AUXILIARY ARRAY
					for (j = n; j >= 1; --j) {
						if (j == n) {
							temp[j] = d[j] * d[j];
						} else {
							temp[j] = temp[j + 1] + (d[j] * d[j]);
						}
					}

					// PERFORM ORTHOGONALIZATION USING A MODIFICATION
					// OF THE GRAHAM-SCHMIDT ORTHOGONALIZATION PROCEDURE BY
					// J. PALMER (1969)
					for (ii = 1; ii <= n; ++ii) {
						if (temp[ii] <= 0.0) {
							continue;
						}
						if (ii == 1) {
							for (j = 1; j <= n; ++j) {
								v[ii][j - 1] = 0.0;
								for (jj = 1; jj <= n; ++jj) {
									v[ii][j - 1] += (d[jj] * vold[jj][j - 1]);
								}
								v[ii][j - 1] /= Math.sqrt(temp[ii]);
							}
						} else {
							for (j = 1; j <= n; ++j) {
								v[ii][j - 1] = 0.0;
								for (jj = ii; jj <= n; ++jj) {
									v[ii][j - 1] += (d[jj] * vold[jj][j - 1]);
								}
								v[ii][j - 1] *= d[ii - 1];
								v[ii][j - 1] -= (vold[ii - 1][j - 1] * temp[ii]);
								v[ii][j - 1] /= Math.sqrt(temp[ii] * temp[ii - 1]);
							}
						}
					}
					d[1] = d[n + 1];
					System.arraycopy(x[n], 0, x[0], 0, n);
					System.arraycopy(x[n + 1], 0, x[1], 0, n);
					i = 2;
					continue;
				}
			}

			// TERMINATION CRITERION
			step *= rho;
			if (step <= eps) {
				System.arraycopy(x[n + 1], 0, x1, 0, n);
				ierr[0] = 0;
				return;
			} else {
				System.arraycopy(x[n + 1], 0, x[0], 0, n);
				i = 1;
			}

			// REACHED MAXIMUM NUMBER OF EVALUATIONS
			if (fev[0] >= maxfev) {
				ierr[0] = 1;
				return;
			}
		}
	}

	private static int line_search(final Function<? super double[], Double> func, final int n, final double[] pos,
			final double[] s, final double[] v, final double[] x, final int[] fev, final int maxfev) {
		final double[] x0 = Arrays.copyOf(pos, n);
		final double[] fs = new double[4];
		double fx, fx0, num, den, stepf;
		int imin;
		boolean goto3;

		// INITIALIZATION
		goto3 = true;
		fx0 = func.apply(x0);
		++fev[0];

		// STEP FORWARD
		BlasMath.daxpy1(n, s[0], v, 1, x0, 1, x, 1);
		fx = func.apply(x);
		++fev[0];

		if (fx > fx0) {

			// STEP BACKWARD
			BlasMath.daxpym(n, -2.0 * s[0], v, 1, x, 1);
			s[0] = -s[0];
			fx = func.apply(x);
			++fev[0];
			if (fx > fx0) {
				goto3 = false;
			}
		}

		if (goto3) {

			// FURTHER STEPS
			do {
				s[0] *= 2.0;
				System.arraycopy(x, 0, x0, 0, n);
				fx0 = fx;
				BlasMath.daxpy1(n, s[0], v, 1, x0, 1, x, 1);
				fx = func.apply(x);
				++fev[0];
				if (fev[0] > maxfev) {
					return 1;
				}
			} while (fx <= fx0 && Math.abs(s[0]) < 1.0e30);

			// PREPARE INTERPOLATION
			s[0] *= 0.5;
			BlasMath.daxpy1(n, s[0], v, 1, x0, 1, x, 1);
		}

		// GENERATE THE FOUR POSSIBLE INTERPOLATION POINTS AND THE VALUES
		BlasMath.daxpy1(n, -s[0], v, 1, x0, 1, x, 1);
		fs[0] = func.apply(x);
		System.arraycopy(x0, 0, x, 0, n);
		fs[1] = func.apply(x);
		BlasMath.daxpy1(n, s[0], v, 1, x0, 1, x, 1);
		fs[2] = func.apply(x);
		BlasMath.daxpy1(n, 2.0 * s[0], v, 1, x0, 1, x, 1);
		fs[3] = func.apply(x);
		fev[0] += 4;

		// IGNORE THE POINT THAT IS FURTHEST FROM THE MINIMUM OF THE FOUR
		// POINTS. FOR THE REMAINING THREE POINTS, COMPUTE THE LAGRANGE
		// QUADRATIC INTERPOLATION BY FITTING A PARABOLA THROUGH THE POINTS
		// AND COMPUTE THE MINIMUM
		imin = argmin(n, fs);
		if (imin == 1) {
			num = s[0] * (fs[0] - fs[2]);
			den = 2.0 * (fs[0] - 2.0 * fs[1] + fs[2]);
			stepf = 0.0;
			if (Math.abs(den) > 0.0) {
				stepf += (num / den);
			}
		} else if (imin == 2) {
			num = s[0] * (fs[1] - fs[3]);
			den = 2.0 * (fs[1] - 2.0 * fs[2] + fs[3]);
			stepf = s[0];
			if (Math.abs(den) > 0.0) {
				stepf += (num / den);
			}
		} else {

			// COULD NOT FIND THE BEST INTERPOLATION POINT SO RETURN THE MIN
			stepf = imin == 0 ? -s[0] : 2.0 * s[0];
			BlasMath.daxpy1(n, stepf, v, 1, x0, 1, x, 1);
			s[0] = stepf;
			return 0;
		}

		// COMPUTE THE POINT AND FUNCTION VALUE AT THE INTERPOLATED STEP
		BlasMath.daxpy1(n, stepf, v, 1, x0, 1, x, 1);
		fx = func.apply(x);
		++fev[0];

		// IF THIS FUNCTION VALUE EXCEEDS F2, THEN RESTORE THE POINT BACK
		// TO THE MIDPOINT OF THE INTERPOLATION INTERVAL
		if ((imin == 1 && fx > fs[1]) || (imin == 2 && fx > fs[2])) {
			stepf = imin == 1 ? 0.0 : s[0];
			BlasMath.daxpy1(n, stepf, v, 1, x0, 1, x, 1);
		}
		s[0] = stepf;
		return 0;
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
}
