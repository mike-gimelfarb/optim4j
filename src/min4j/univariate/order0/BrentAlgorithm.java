package min4j.univariate.order0;

import java.util.function.Function;

import min4j.utils.Constants;

/**
 *
 * @author Michael
 */
public final class BrentAlgorithm extends DerivativeFreeOptimizer {

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public BrentAlgorithm(final double absoluteTolerance, final double relativeTolerance, final int maxEvaluations) {
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
		final double result = lbrent(f, a, b, myRelTol, myTol, myMaxEvals, fev);
		myEvals = fev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double lbrent(final Function<? super Double, Double> f, double a, double b, final double rtol,
			final double atol, final int maxfev, final int[] fev) {

		// INITIALIZE CONSTANTS
		final double c = (1.0 / Constants.GOLDEN) / Constants.GOLDEN;

		// INITIALIZE ITERATION VARIABLES
		double d = 0.0, e = 0.0, m = 0.0, p = 0.0, q = 0.0, r = 0.0, t2 = 0.0, tol = 0.0, x = a + c * (b - a), w = x,
				v = w, u, fx = f.apply(x), fw = fx, fv = fw;
		fev[0] = 1;
		boolean goto10 = true;

		while (true) {

			if (goto10) {

				// CHECK CONVERGENCE
				m = 0.5 * (a + b);
				tol = rtol * Math.abs(x) + atol;
				t2 = 2.0 * tol;
				if (Math.abs(x - m) <= t2 - 0.5 * (b - a)) {
					return x;
				}

				// parabolic interpolation
				r = 0.0;
				q = r;
				p = q;
				if (Math.abs(e) > tol) {
					r = (x - w) * (fx - fv);
					q = (x - v) * (fx - fw);
					p = (x - v) * q - (x - w) * r;
					q = 2.0 * (q - r);
					if (q <= 0.0) {
						q = -q;
					} else {
						p = -p;
					}
					r = e;
					e = d;
				}
			}

			// continue with parabolic interpolation
			if (Math.abs(p) < 1e-25 || Math.abs(p) >= Math.abs(0.5 * q * r) || p <= q * (a - x) || p >= q * (b - x)) {
				e = x >= m ? a - x : b - x;
				d = c * e;
			} else {
				d = p / q;
				u = x + d;

				// try golden section search
				if ((u - a) < t2 || (b - u) < t2) {
					d = (x >= m) ? -tol : tol;
				}
			}
			if (Math.abs(d) < tol) {
				u = x + (d <= 0.0 ? -tol : tol);
			} else {
				u = x + d;
			}

			// added a stop if budget is reached
			if (fev[0] >= maxfev) {
				return Double.NaN;
			}

			// update interval
			final double fu = f.apply(u);
			++fev[0];
			if (fu <= fx) {
				if (u >= x) {
					a = x;
				} else {
					b = x;
				}
				v = w;
				fv = fw;
				w = x;
				fw = fx;
				x = u;
				fx = fu;
				goto10 = true;
			} else {
				if (u >= x) {
					b = u;
				} else {
					a = u;
				}
				if (fu <= fw || w == x) {
					v = w;
					fv = fw;
					w = u;
					fw = fu;
					goto10 = true;
				} else if (fu > fv && v != x && v != w) {
					q = -q;
					r = e;
					e = d;
					goto10 = false;
				} else {
					v = u;
					fv = fu;
					goto10 = true;
				}
			}
		}
	}
}
