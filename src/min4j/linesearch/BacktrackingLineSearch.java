package min4j.linesearch;

import java.util.function.Function;

import min4j.utils.BlasMath;
import min4j.utils.Pair;

/**
 *
 * @author Michael
 */
public final class BacktrackingLineSearch extends LineSearch {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double myRho, myMaxStepSize;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param decay
	 * @param maximum
	 * @param maxIterations
	 */
	public BacktrackingLineSearch(final double tolerance, final double decay, final double maximum,
			final int maxIterations) {
		super(tolerance, maxIterations);
		myRho = decay;
		myMaxStepSize = maximum;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double initial) {
		final int D = x0.length;

		// prepare initial position and dot products
		final double[] x = new double[D];
		double step = initial;
		double y;
		BlasMath.daxpy1(D, step, dir, 1, x0, 1, x, 1);
		final double dy = BlasMath.ddotm(D, df0, 1, dir, 1);
		final double normdir = BlasMath.denorm(D, dir);

		// main loop of backtracking line search
		for (int i = 0; i < myMaxIters; ++i) {

			// compute new position and function value for step
			BlasMath.daxpy1(D, step, dir, 1, x0, 1, x, 1);
			y = f.apply(x);
			++myEvals;

			// check the approximate Wolfe condition
			if (y <= f0 + C1 * step * dy) {
				return new Pair<>(step, x);
			}

			// update step size
			step = Math.min(step * myRho, myMaxStepSize / normdir);
		}
		return new Pair<>(step, x);
	}
}
