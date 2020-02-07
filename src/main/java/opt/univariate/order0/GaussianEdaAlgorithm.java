package opt.univariate.order0;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 *
 * @author Michael
 */
public final class GaussianEdaAlgorithm extends DerivativeFreeOptimizer {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	private static final Random RAND = new Random();

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int myNp; // the population size
	private final int myNb; // the size of the "elite" group
	private final int myMaxEvals; // maximum number of function evaluations

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
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

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double optimize(final Function<? super Double, Double> func, final double a, final double b) {

		// prepare variables
		final int[] fev = new int[1];

		// call main subroutine
		final double result = eda(func, a, b, myTol, myMaxEvals, myNp, myNb, fev);
		myEvals = fev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double eda(final Function<? super Double, Double> func, final double a, final double b,
			final double tol, final int maxfev, final int np, final int nb, final int[] fev) {

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
		while (fev[0] < maxfev) {

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
				return pool[0][0];
			}
		}
		return Double.NaN;
	}
}
