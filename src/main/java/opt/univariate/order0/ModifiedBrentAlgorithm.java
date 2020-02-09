/**
 * Original FORTRAN77 version by Richard Brent.
 * FORTRAN90 version by John Burkardt.
 * 
 * This code is distributed under the GNU LGPL license. 
 */
package opt.univariate.order0;

import java.util.function.Function;

import utils.RealMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] Brent, Richard P. Algorithms for minimization without derivatives.
 * Courier Corporation, 2013.
 * 
 * [2] Kahaner, David, Cleve Moler, and Stephen Nash. "Numerical methods and
 * software." Englewood Cliffs: Prentice Hall, 1989 (1989).
 */
public final class ModifiedBrentAlgorithm extends DerivativeFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private double c, d, e, fu, fv, fw, fx, midpoint, p, q, r, tol1, tol2, u, v, w, x;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param absTolerance
	 * @param relTolerance
	 * @param maxEvaluations
	 */
	public ModifiedBrentAlgorithm(final double absTolerance, final double relTolerance, final int maxEvaluations) {
		super(absTolerance, relTolerance, maxEvaluations);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double optimize(final Function<? super Double, Double> f, final double a, final double b) {

		// prepare variables
		final int[] status = new int[1];
		final double[] aarr = { a };
		final double[] barr = { b };
		final double[] arg = { 0.5 * (a + b) };
		double value = f.apply(arg[0]);
		++myEvals;

		// main loop
		while (true) {
			localmin(aarr, barr, arg, status, value, myRelTol, myTol);
			if (status[0] == 0) {
				return arg[0];
			} else if (myEvals >= myMaxEvals) {
				return Double.NaN;
			} else {
				value = f.apply(arg[0]);
				++myEvals;
			}
		}
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void localmin(final double[] a, final double[] b, final double[] arg, final int[] status,
			final double value, final double eps, final double tol) {

		if (status[0] == 0) {

			// STATUS (INPUT) = 0, startup.
			if (b[0] <= a[0]) {
				status[0] = -1;
				return;
			}
			c = 0.5 * (3.0 - Math.sqrt(5));
			v = a[0] + c * (b[0] - a[0]);
			w = x = v;
			e = 0.0;
			status[0] = 1;
			arg[0] = x;
			return;
		} else if (status[0] == 1) {

			// STATUS (INPUT) = 1, return with initial function value of FX.
			fx = value;
			fv = fw = fx;
		} else if (2 <= status[0]) {

			// STATUS (INPUT) = 2 or more, update the data.
			fu = value;
			if (fu <= fx) {
				if (x <= u) {
					a[0] = x;
				} else {
					b[0] = x;
				}
				v = w;
				fv = fw;
				w = x;
				fw = fx;
				x = u;
				fx = fu;
			} else {
				if (u < x) {
					a[0] = u;
				} else {
					b[0] = u;
				}
				if (fu <= fw || w == x) {
					v = w;
					fv = fw;
					w = u;
					fw = fu;
				} else if (fu <= fv || v == x || v == w) {
					v = u;
					fv = fu;
				}
			}
		}

		// Take the next step.
		midpoint = 0.5 * (a[0] + b[0]);
		tol1 = eps * Math.abs(x) + tol / 3.0;
		tol2 = 2.0 * tol1;

		// If the stopping criterion is satisfied, we can exit.
		if (Math.abs(x - midpoint) <= (tol2 - 0.5 * (b[0] - a[0]))) {
			status[0] = 0;
			return;
		}

		// Is golden-section necessary?
		if (Math.abs(e) <= tol1) {
			if (midpoint <= x) {
				e = a[0] - x;
			} else {
				e = b[0] - x;
			}
			d = c * e;
		} else {

			// Consider fitting a parabola.
			r = (x - w) * (fx - fv);
			q = (x - v) * (fx - fw);
			p = (x - v) * q - (x - w) * r;
			q = 2.0 * (q - r);
			if (0.0 < q) {
				p = -p;
			}
			q = Math.abs(q);
			r = e;
			e = d;

			// Choose a golden-section step if the parabola is not advised.
			if (Math.abs(0.5 * q * r) <= Math.abs(p) || p <= q * (a[0] - x) || q * (b[0] - x) <= p) {
				if (midpoint <= x) {
					e = a[0] - x;
				} else {
					e = b[0] - x;
				}
				d = c * e;
			} else {

				// Choose a parabolic interpolation step.
				d = p / q;
				u = x + d;
				if (u - a[0] < tol2) {
					d = RealMath.sign(tol1, midpoint - x);
				}
				if (b[0] - u < tol2) {
					d = RealMath.sign(tol1, midpoint - x);
				}
			}
		}

		// F must not be evaluated too close to X.
		if (tol1 <= Math.abs(d)) {
			u = x + d;
		}
		if (Math.abs(d) < tol1) {
			u = x + RealMath.sign(tol1, d);
		}

		// Request value of F(U).
		arg[0] = u;
		++status[0];
	}
}
