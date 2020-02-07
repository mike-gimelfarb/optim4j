package opt.multivariate.unconstrained.leastsquares;

import java.util.function.Function;

import opt.Optimizer;

/**
 *
 * @author Michael
 */
public abstract class LeastSquaresOptimizer
		extends Optimizer<double[], double[], Function<? super double[], double[]>> {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	protected static final int MAX_ITERS = (int) 1e7;

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final double myTol;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 */
	public LeastSquaresOptimizer(final double tolerance) {
		myTol = tolerance;
	}

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	public abstract double[] optimize(Function<? super double[], double[]> func, double[] guess);
}
