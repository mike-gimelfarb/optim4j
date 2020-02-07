package opt.univariate.order0;

import java.util.function.Function;

/**
 * Practical Optimization: Algorithms and Engineering Applications
 *
 * @author Michael
 */
public final class DaviesSwannCampeyAlgorithm extends DerivativeFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double myL;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param deltaDecayFactor
	 * @param maxEvaluations
	 */
	public DaviesSwannCampeyAlgorithm(final double tolerance, final double deltaDecayFactor, final int maxEvaluations) {
		super(tolerance, 0.0, maxEvaluations);
		myL = deltaDecayFactor;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public DaviesSwannCampeyAlgorithm(final double tolerance, final int maxEvaluations) {
		this(tolerance, 0.1, maxEvaluations);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public double optimize(final Function<? super Double, Double> f, final double a, final double b) {

		// prepare work arrays
		final int[] fev = new int[1];

		// call main subroutine
		final double result = dsc(f, a, b, myL, myTol, fev, myMaxEvals);
		myEvals += fev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double dsc(final Function<? super Double, Double> f, final double a, final double b, final double K,
			final double tol, final int[] fev, final int maxfev) {
		final double delta1 = 0.5 * (b - a);
		final double guess = 0.5 * (a + b);
		return dsc1(f, guess, delta1, K, tol, fev, maxfev);
	}

	private static double dsc1(final Function<? super Double, Double> f, final double guess, final double delta1,
			final double K, final double tol, final int[] fev, final int maxfev) {

		// step 1: initialization
		double x0 = guess;
		double delta = delta1;
		fev[0] = 0;

		// main loop of DSC algorithm
		while (true) {

			// step 2
			final double xm1 = x0 - delta;
			final double xp1 = x0 + delta;
			final double f0 = f.apply(x0);
			final double fp1 = f.apply(xp1);
			double p;
			fev[0] += 2;

			// step 3
			if (f0 > fp1) {
				p = 1.0;
			} else {
				final double fm1 = f.apply(xm1);
				++fev[0];
				if (fm1 < f0) {
					p = -1.0;
				} else {

					// step 7: update the position of the minimum
					final double num = delta * (fm1 - fp1);
					final double den = 2.0 * (fm1 - 2.0 * f0 + fp1);
					x0 += (num / den);
					if (delta <= tol) {
						return x0;
					} else {
						delta *= K;
						continue;
					}
				}
			}

			// step 4
			double twonm1 = 1.0;
			double fnm2 = f.apply(xm1);
			double xnm1 = x0;
			double fnm1 = f0;
			double xn;
			double fn;
			++fev[0];
			while (true) {
				xn = xnm1 + twonm1 * p * delta;
				fn = f.apply(xn);
				++fev[0];
				if (fn > fnm1) {
					break;
				} else {
					fnm2 = fnm1;
					xnm1 = xn;
					fnm1 = fn;
					twonm1 *= 2.0;
				}
				if (!Double.isFinite(xn)) {
					return Double.NaN;
				}
			}

			// step 5
			final double twonm2 = twonm1 / 2.0;
			final double xm = xnm1 + twonm2 * p * delta;
			final double fm = f.apply(xm);
			++fev[0];

			// step 6: update the position of the minimum
			if (fm >= fnm1) {
				final double num = twonm2 * p * delta * (fnm2 - fm);
				final double den = 2.0 * (fnm2 - 2.0 * fnm1 + fm);
				x0 = xnm1 + (num / den);
			} else {
				final double num = twonm2 * p * delta * (fnm1 - fn);
				final double den = 2.0 * (fnm1 - 2.0 * fm + fn);
				x0 = xm + (num / den);
			}

			// convergence test
			if (twonm2 * delta <= tol) {
				return x0;
			}
			if (fev[0] >= maxfev) {
				return Double.NaN;
			}
			delta *= K;
		}
	}
}
