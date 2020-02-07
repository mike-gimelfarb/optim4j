package opt.univariate.order1;

import java.util.function.Function;

/**
 *
 * @author michael
 */
public final class SecantAlgorithm extends DerivativeOptimizer {

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public SecantAlgorithm(final double absoluteTolerance, final double relativeTolerance, final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double optimize(final Function<? super Double, Double> f, final Function<? super Double, Double> df,
			final double a, final double b) {

		// prepare variables
		final int[] fev = new int[1];
		final int[] dfev = new int[1];

		// call main subroutine
		final double result = secantMin(df, a, b, myTol, myRelTol, myMaxEvals, fev, dfev);
		myEvals += fev[0];
		myDEvals += dfev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double secantMin(final Function<? super Double, Double> dfunc, double a, double b, final double tol,
			final double reltol, final int maxfev, final int[] fev, final int[] dfev) {
		fev[0] = dfev[0] = 0;

		// generate two points
		double dfb = dfunc.apply(b);
		++dfev[0];
		double x0 = a + (b - a) / 3.0;
		double df0 = dfunc.apply(x0);
		++dfev[0];
		double x1 = a + 2.0 * (b - a) / 3.0;
		double df1 = dfunc.apply(x1);
		++dfev[0];
		boolean secant = false;

		// main loop of secant method for f'
		while (true) {

			final double mid = a + 0.5 * (b - a);
			double x2;
			final boolean secant1;
			if (df1 == 0.0 || df1 == -0.0) {

				// first-order condition is satisfied
				return x1;
			}

			// estimate the second derivative
			final double d2f = (df1 - df0) / (x1 - x0);
			if (d2f == 0.0 || d2f == -0.0) {

				// f" estimate is zero, use bisection instead
				x2 = mid;
				secant1 = false;
			} else {

				// attempt a secant step
				x2 = x1 - df1 / d2f;

				// accept or reject secant step
				if (x2 <= a || x2 >= b) {
					x2 = mid;
					secant1 = false;
				} else {
					secant1 = true;
				}
			}

			// check for convergence
			// test for budget
			if (dfev[0] >= maxfev) {
				return Double.NaN;
			}

			// test sufficient reduction in the size of the bracket
			double xtol = tol + reltol * Math.abs(mid);
			final double df2 = dfunc.apply(x2);
			++dfev[0];
			if (Math.abs(b - a) <= xtol) {
				return x2;
			}

			// test assuming secant method was used
			if (secant1 && secant) {
				xtol = tol + reltol * Math.abs(x2);
				if (Math.abs(x2 - x1) <= xtol && Math.abs(df2) <= tol) {
					return x2;
				}
			}

			// update the bracket
			x0 = x1;
			x1 = x2;
			df0 = df1;
			df1 = df2;
			secant = secant1;
			if (df1 * dfb < 0.0) {
				a = x1;
			} else {
				b = x1;
				dfb = df1;
			}
		}
	}
}
