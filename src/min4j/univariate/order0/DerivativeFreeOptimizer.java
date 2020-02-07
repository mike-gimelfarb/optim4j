package min4j.univariate.order0;

import java.util.function.Function;

import min4j.univariate.UnivariateOptimizer;
import min4j.utils.Constants;

/**
 *
 * @author Michael
 */
public abstract class DerivativeFreeOptimizer extends UnivariateOptimizer {

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public DerivativeFreeOptimizer(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	public abstract double optimize(Function<? super Double, Double> f, double a, double b);

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public Double optimize(final Function<? super Double, Double> f, final Double guess) {

		// first use guess to compute a bracket [a, b] that contains a min
		final int[] fev = new int[1];
		final double[] brackt = bracket(f, guess, Constants.GOLDEN, myMaxEvals, fev);
		myEvals += fev[0];
		if (brackt == null) {
			return Double.NaN;
		}
		final double a = brackt[0];
		final double b = brackt[1];

		// perform optimization using the bracketed routine
		return optimize(f, a, b);
	}
}
