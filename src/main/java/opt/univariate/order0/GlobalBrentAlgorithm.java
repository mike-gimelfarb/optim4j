/**
 * Original FORTRAN77 version by Richard Brent.
 * FORTRAN90 version by John Burkardt.
 * 
 * This code is distributed under the GNU LGPL license. 
 */
package opt.univariate.order0;

import java.util.function.Function;

import opt.OptimizerSolution;
import opt.univariate.DerivativeFreeOptimizer;
import utils.Constants;

/**
 * 
 * REFERENCES:
 * 
 * [1] Brent, Richard P. Algorithms for minimization without derivatives.
 * Courier Corporation, 2013.
 */
public final class GlobalBrentAlgorithm extends DerivativeFreeOptimizer {

	private double myM;

	/**
	 *
	 * @param tolerance
	 * @param boundOnD2f
	 * @param maxEvaluations
	 */
	public GlobalBrentAlgorithm(final double tolerance, final double boundOnD2f, final int maxEvaluations) {
		super(tolerance, 0.0, maxEvaluations);
		myM = boundOnD2f;
	}

	@Override
	public final OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f, final double a,
			final double b) {
		final int[] fev = new int[1];
		final boolean[] converged = new boolean[1];

		final double result = gbrent(f, myM, a, b, myTol, myMaxEvals, fev, converged);
		return new OptimizerSolution<>(result, fev[0], 0, converged[0]);
	}

	/**
	 *
	 * @param bound
	 */
	public final void setBound(final double bound) {
		myM = bound;
	}

	private static double gbrent(final Function<? super Double, Double> f, final double boundOnD2f, double a, double b,
			final double tol, final int maxfev, final int[] fev, final boolean[] converged) {
		final double m2 = 0.5 * (1.0 + 16.0 * Constants.EPSILON) * boundOnD2f;
		double a0 = b, a2 = a, a3, c = b, d0, d1, d2, h = 9.0 / 11.0, p, q, qs, r, s, sc = 0.0, x = a0, y0 = f.apply(b),
				y1, y2 = f.apply(a), y = y2, y3, yb = y0, z0, z1, z2;
		fev[0] = 2;
		int k = 3;
		if (y0 < y) {
			y = y0;
		} else {
			x = a;
		}
		if (boundOnD2f <= 0.0 || a >= b) {
			return x;
		}
		sc = (sc <= a || sc >= b) ? 0.5 * (a + b) : c;
		y1 = f.apply(sc);
		++fev[0];
		d0 = a2 - sc;
		if (y1 < y) {
			x = sc;
			y = y1;
		}

		while (true) {
			d1 = a2 - a0;
			d2 = sc - a0;
			z2 = b - a2;
			z0 = y2 - y1;
			z1 = y2 - y0;
			r = d1 * d1 * z0 - d0 * d0 * z1;
			p = r;
			qs = 2.0 * (d0 * z1 - d1 * z0);
			q = qs;
			boolean goto40 = k <= 1.0e6 || y >= y2;
			do {
				if (goto40) {
					final double right = z2 * m2 * r * (z2 * q - r);
					if (q * (r * (yb - y2) + z2 * q * (y2 - y + tol)) < right) {
						a3 = a2 + r / q;
						y3 = f.apply(a3);
						++fev[0];
						if (y3 < y) {
							x = a3;
							y = y3;
						}
					}
				}
				k = (1611 * k) % 1048576;
				q = 1.0;
				r = (b - a) * 1.0e-5 * k;
				goto40 = true;
				if (fev[0] >= maxfev) {
					return x;
				}
			} while (r < z2);

			r = m2 * d0 * d1 * d2;
			s = Math.sqrt((y2 - y + tol) / m2);
			h = 0.5 * (1.0 + h);
			p = h * (p + 2.0 * r * s);
			q += (0.5 * qs);
			r = -0.5 * (d0 + (z0 + 2.01 * tol) / (d0 * m2));
			r = (r >= s && d0 >= 0.0) ? r + a2 : a2 + s;
			a3 = (p * q <= 0.0) ? r : a2 + p / q;

			while (true) {
				if (a3 < r) {
					a3 = r;
				}
				if (a3 < b) {
					y3 = f.apply(a3);
					++fev[0];
				} else {
					a3 = b;
					y3 = yb;
				}
				if (y3 < y) {
					x = a3;
					y = y3;
				}
				d0 = a3 - a2;
				if (a3 <= r) {
					break;
				} else {
					p = 2.0 * (y2 - y3) / (boundOnD2f * d0);
					final double right = (y2 - y) + (y3 - y) + 2.0 * tol;
					if (Math.abs(p) >= (1.0 + 9.0 * Constants.EPSILON) * d0 || 0.5 * m2 * (d0 * d0 + p * p) <= right) {
						break;
					} else {
						a3 = 0.5 * (a2 + a3);
						h *= 0.9;
						if (fev[0] >= maxfev) {
							return x;
						}
					}
				}
			}
			if (a3 >= b) {
				converged[0] = true;
				return x;
			} else {
				a0 = sc;
				sc = a2;
				a2 = a3;
				y0 = y1;
				y1 = y2;
				y2 = y3;
				if (fev[0] >= maxfev) {
					converged[0] = false;
					return x;
				}
			}
		}
	}
}
