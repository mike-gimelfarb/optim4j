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

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import opt.univariate.UnivariateOptimizerSolution;
import opt.univariate.DerivativeFreeOptimizer;

/**
 *
 */
public final class GaussianEdaAlgorithm extends DerivativeFreeOptimizer {

	private static final Random RAND = new Random();

	private final int myNp, myNb; // the size of the population, "elite" group
	private final int myMaxEvals; // maximum number of function evaluations

	/**
	 *
	 * @param tolerance
	 * @param popSize
	 * @param eliteSize
	 * @param maxEvaluations
	 */
	public GaussianEdaAlgorithm(final double tolerance, final int popSize, final int eliteSize,
			final int maxEvaluations) {
		super(tolerance, 0.0, maxEvaluations);
		myNp = popSize;
		myNb = eliteSize;
		myMaxEvals = maxEvaluations;
	}

	/**
	 *
	 * @param tolerance
	 * @param popSize
	 * @param maxEvaluations
	 */
	public GaussianEdaAlgorithm(final double tolerance, final int popSize, final int maxEvaluations) {
		this(tolerance, popSize, popSize / 2, maxEvaluations);
	}

	@Override
	public final UnivariateOptimizerSolution optimize(final Function<? super Double, Double> func, final double a,
			final double b) {

		// prepare variables
		final int[] fev = new int[1];
		final boolean[] converged = new boolean[1];

		// call main subroutine
		final double result = eda(func, a, b, myTol, myMaxEvals, myNp, myNb, fev, converged);
		return new UnivariateOptimizerSolution(result, fev[0], 0, converged[0]);
	}

	private static double eda(final Function<? super Double, Double> func, final double a, final double b,
			final double tol, final int maxfev, final int np, final int nb, final int[] fev,
			final boolean[] converged) {

		// prepare the population by randomization in [lb, ub]
		final double[][] pool = new double[np][2];
		for (int n = 0; n < np; ++n) {
			final double x = (b - a) * RAND.nextDouble() + a;
			final double fx = func.apply(x);
			pool[n][0] = x;
			pool[n][1] = fx;
		}
		fev[0] = np;
		Arrays.sort(pool, (u, v) -> Double.compare(u[1], v[1]));

		// main loop of EDA
		while (fev[0] + np - nb <= maxfev) {

			// compute both the mean and variance of the model distribution
			double mu = 0.0;
			double sigma = 0.0;
			for (int n = 0; n < nb; ++n) {
				final double x = pool[n][0];
				final int np1 = n + 1;
				final double delta = x - mu;
				mu += (delta / np1);
				final double delta2 = x - mu;
				sigma += (delta * delta2);
			}
			sigma /= nb;
			sigma = Math.sqrt(sigma);

			// copy the best members of the population into the next generation
			// but use the new model to replace the remaining members at random
			for (int n = nb; n < np; ++n) {
				double x = 0.0;
				do {
					x = mu + RAND.nextGaussian() * sigma;
				} while (x < a || x > b);
				final double fx = func.apply(x);
				pool[n][0] = x;
				pool[n][1] = fx;
			}
			fev[0] += (np - nb);

			// sort the population members by fitness
			Arrays.sort(pool, (u, v) -> Double.compare(u[1], v[1]));

			// check convergence based on the standard deviation
			if (sigma <= tol) {
				converged[0] = true;
				return pool[0][0];
			}
		}
		return pool[0][0];
	}
}
