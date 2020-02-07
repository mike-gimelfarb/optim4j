package min4j.univariate.order0;

import java.util.function.Function;

import min4j.utils.Constants;

/**
 *
 * @author Michael
 */
public final class GoldenSectionAlgorithm extends DerivativeFreeOptimizer {

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public GoldenSectionAlgorithm(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double optimize(final Function<? super Double, Double> f, final double a, final double b) {

		// prepare variables
		final int[] fev = new int[1];

		// call main subroutine
		final double result = gssearch(f, a, b, myRelTol, myTol, myMaxEvals, fev);
		myEvals = fev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double gssearch(final Function<? super Double, Double> f, final double a, final double b,
			final double rtol, final double atol, final int mfev, final int[] fev) {

		// INITIALIZE CONSTANTS
		final double GOLD = Constants.GOLDEN;

		// INITIALIZE ITERATION VARIABLES
		double a1 = a;
		double b1 = b;
		double c = b1 - (b1 - a1) / GOLD;
		double d = a1 + (b1 - a1) / GOLD;

		// MAIN LOOP OF GOLDEN SECTION SEARCH
		while (fev[0] < mfev) {

			// CHECK CONVERGENCE
			final double mid = 0.5 * (c + d);
			final double tol = rtol * Math.abs(mid) + atol;
			if (Math.abs(c - d) <= tol) {
				return mid;
			}

			// evaluate at new points
			final double fc = f.apply(c);
			final double fd = f.apply(d);
			fev[0] += 2;

			// update interval
			if (fc < fd) {
				b1 = d;
			} else {
				a1 = c;
			}
			final double del = (b1 - a1) / GOLD;
			c = b1 - del;
			d = a1 + del;
		}

		// COULD NOT CONVERGE
		return Double.NaN;
	}
}
