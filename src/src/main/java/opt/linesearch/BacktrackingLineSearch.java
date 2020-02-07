package opt.linesearch;

import java.util.function.Function;

import utils.BlasMath;
import utils.Pair;

/**
 * A backtracking line search algorithm described in Nocedal and Wright (2006).
 * This line search routine is largely academic and should not be used in
 * practical settings.
 * 
 * 
 * [1] Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer
 * Science & Business Media, 2006.
 * 
 * @author Michael
 */
public final class BacktrackingLineSearch extends LineSearch {

	private final double myRho, myMaxStepSize;

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
