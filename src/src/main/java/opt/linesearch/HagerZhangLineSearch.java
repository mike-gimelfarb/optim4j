package opt.linesearch;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import utils.BlasMath;
import utils.Pair;

/**
 * A translation of the code from the Julia package, implementing the
 * Hager-Zhang line search introduced in Hager and Zhang (2006). This should be the
 * preferred line search subroutine in general purpose optimization, along with
 * More-Thuente.
 * 
 * 
 * [1] Hager, William W., and Hongchao Zhang. "Algorithm 851: CG_DESCENT, a
 * conjugate gradient method with guaranteed descent." ACM Transactions on
 * Mathematical Software (TOMS) 32.1 (2006): 113-137.
 *
 * @author Michael
 */
public final class HagerZhangLineSearch extends LineSearch {

	private static class LineStep {

		public double step, slope, value;

		public LineStep(double _step, double _slope, double _value) {
			step = _step;
			slope = _slope;
			value = _value;
		}
	}

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 */
	public HagerZhangLineSearch(final double tolerance, final int maxIterations) {
		super(tolerance, maxIterations);
	}

	@Override
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double initial) {

		// prepare variables
		final int[] fev = new int[1];

		// call main subroutine
		final Pair<Double, double[]> result = hagerzhang(f, df, x0, dir, df0, f0, initial, myTol, myMaxIters, fev);
		myEvals += fev[0];
		return result;
	}

	/**
	 *
	 * @param f
	 * @param df
	 * @param x0
	 * @param dir
	 * @param df0
	 * @param f0
	 * @param psi0
	 * @param psi1
	 * @param psi2
	 * @param epsk
	 * @param maxit
	 * @param k
	 * @param pstep
	 * @param quadstep
	 * @return
	 */
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double psi0, final double psi1, final double psi2, final double epsk,
			final int maxit, final int k, final double pstep, final boolean quadstep) {

		// prepare variables
		final int n = x0.length;
		final int[] fev = new int[1];

		// initialize the step size
		final double xnorm = BlasMath.denorm(n, x0);
		final double gnorm = BlasMath.denorm(n, df0);
		final double c;
		if (k == 0) {

			// estimate of initial step size for the first iteration
			if (Math.abs(xnorm) >= 1.0e-12) {
				c = psi0 * xnorm / gnorm;
			} else if (Math.abs(f0) >= 1.0e-12) {
				c = psi0 * Math.abs(f0) / (gnorm * gnorm);
			} else {
				c = 1.0;
			}
			fev[0] = 0;
		} else if (quadstep) {

			// not the first iteration - attempt quadratic interpolation
			final double phi0 = f0;
			final double dphi0 = BlasMath.ddotm(n, df0, 1, dir, 1);
			final double step = psi1 * pstep;
			final double[] wa = new double[n];
			for (int i = 0; i < n; ++i) {
				wa[i] = x0[i] + step * dir[i];
			}
			final double phi1 = f.apply(wa);
			final double a = (phi1 - phi0 - step * dphi0) / step / step;
			final double b = dphi0;
			if (phi1 <= phi0 && a > 0.0 && Math.abs(b) > 0.0) {

				// interpolation accepted
				c = -b / (2.0 * a);
			} else {

				// interpolation rejected
				c = psi2 * pstep;
			}
			fev[0] = 1;
		} else {
			c = psi2 * pstep;
		}

		// call main subroutine
		final Pair<Double, double[]> result = hagerzhang(f, df, x0, dir, df0, f0, c, epsk, maxit, fev);
		myEvals += fev[0];
		myDEvals += fev[0] - 1;
		return result;
	}

	private static Pair<Double, double[]> hagerzhang(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double initial, final double eps, final int maxit, final int[] fev) {

		// prepare variables
		final int n = x0.length;
		final double[] wa = new double[n];
		final double[] dfarr = new double[n];
		final double phi0 = f0;
		final double dphi0 = BlasMath.ddotm(n, df0, 1, dir, 1);
		final List<LineStep> lsr = new ArrayList<>();
		lsr.add(new LineStep(0.0, dphi0, phi0));

		// call main subroutine
		final double stepf = hagerzhang(lsr, n, dfarr, x0, dir, wa, initial, false, 0.1, 0.9, Double.POSITIVE_INFINITY,
				5.0, eps, 0.66, maxit, 0.1, 10000, f, df, fev);
		BlasMath.daxpy1(n, stepf, dir, 1, x0, 1, wa, 1);
		return new Pair<>(stepf, wa);
	}

	private static double hagerzhang(final List<LineStep> lsr, final int n, final double[] df, final double[] x,
			final double[] d, final double[] wa, double c, boolean canbreak, final double delta, final double sigma,
			double stepmax, final double rho, final double eps, final double gamma, final int lsmax, final double psi3,
			final int itfmax, final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final int[] fev) {

		double phi0 = lsr.get(1 - 1).value;
		double dphi0 = lsr.get(1 - 1).slope;
		double philim = phi0 + eps * Math.abs(phi0);
		double[] val = eval(n, df, x, d, c, wa, true, func, dfunc);
		double phic = val[0];
		double dphic = val[1];
		++fev[0];
		int itf = 1;
		while (!(Double.isFinite(phic) && Double.isFinite(dphic)) && itf < itfmax) {
			canbreak = false;
			++itf;
			c *= psi3;
			val = eval(n, df, x, d, c, wa, true, func, dfunc);
			phic = val[0];
			dphic = val[1];
			++fev[0];
		}
		if (!(Double.isFinite(phic) && Double.isFinite(dphic))) {
			return 0.0;
		}
		lsr.add(new LineStep(c, dphic, phic));
		if (canbreak && satisfiesWolfe(c, phic, dphic, phi0, dphi0, philim, delta, sigma)) {
			return c;
		}
		boolean isbrak = false;
		int ia = 1;
		int ib = 2;
		int it = 1;
		double cold;
		while (!isbrak && it < lsmax) {
			if (dphic >= 0.0) {
				ib = lsr.size();
				for (int i = ib - 1; i >= 1; --i) {
					if (lsr.get(i - 1).value <= philim) {
						ia = i;
						break;
					}
				}
				isbrak = true;
			} else if (lsr.get(lsr.size() - 1).value > philim) {
				ib = lsr.size();
				ia = ib - 1;
				final int[] iab = bisect(lsr, n, df, x, d, wa, ia, ib, philim, func, dfunc, fev);
				ia = iab[0];
				ib = iab[1];
				isbrak = true;
			} else {
				cold = c;
				c *= rho;
				if (c > stepmax) {
					c = (stepmax + cold) / 2.0;
					if (c == cold || nextFloat(c) >= stepmax) {
						return cold;
					}
				}
				val = eval(n, df, x, d, c, wa, true, func, dfunc);
				phic = val[0];
				dphic = val[1];
				++fev[0];
				itf = 1;
				while (!(Double.isFinite(phic) && Double.isFinite(dphic)) && c > nextFloat(cold) && itf < itfmax) {
					stepmax = c;
					++itf;
					c = (cold + c) / 2.0;
					val = eval(n, df, x, d, c, wa, true, func, dfunc);
					phic = val[0];
					dphic = val[1];
					++fev[0];
				}
				if (!(Double.isFinite(phic) && Double.isFinite(dphic))) {
					return cold;
				} else if (dphic < 0.0 && c == stepmax) {
					return c;
				}
				lsr.add(new LineStep(c, dphic, phic));
			}
			++it;
		}
		while (it < lsmax) {
			double a = lsr.get(ia - 1).step;
			double b = lsr.get(ib - 1).step;
			if (b - a <= Math.ulp(b)) {
				return a;
			}
			final int[] sec = secant2(lsr, n, df, x, d, wa, ia, ib, philim, delta, sigma, func, dfunc, fev);
			boolean iswolfe = sec[0] == 1;
			int iA = sec[1];
			int iB = sec[2];
			if (iswolfe) {
				return lsr.get(iA - 1).step;
			}
			double A = lsr.get(iA - 1).step;
			double B = lsr.get(iB - 1).step;
			if (B - A < gamma * (b - a)) {
				if (nextFloat(lsr.get(ia - 1).value) >= lsr.get(ib - 1).value
						&& nextFloat(lsr.get(iA - 1).value) >= lsr.get(iB - 1).value) {
					return A;
				}
				ia = iA;
				ib = iB;
			} else {
				c = (A + B) / 2.0;
				val = eval(n, df, x, d, c, wa, true, func, dfunc);
				phic = val[0];
				dphic = val[1];
				++fev[0];
				lsr.add(new LineStep(c, dphic, phic));
				final int[] iab = update(lsr, n, df, x, d, wa, iA, iB, lsr.size(), philim, func, dfunc, fev);
				ia = iab[0];
				ib = iab[1];
			}
			++it;
		}
		throw new RuntimeException("Line search failed to converge" + " - reached maximum # of iterations.");
	}

	private static int[] secant2(final List<LineStep> lsr, final int n, final double[] df, final double[] x,
			final double[] d, final double[] wa, final int ia, final int ib, final double philim, final double delta,
			final double sigma, final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final int[] fev) {

		double phi0 = lsr.get(1 - 1).value;
		double dphi0 = lsr.get(1 - 1).slope;
		double a = lsr.get(ia - 1).step;
		double b = lsr.get(ib - 1).step;
		double dphia = lsr.get(ia - 1).slope;
		double dphib = lsr.get(ib - 1).slope;
		double c = secant(a, b, dphia, dphib);
		double[] val = eval(n, df, x, d, c, wa, true, func, dfunc);
		double phic = val[0];
		double dphic = val[1];
		++fev[0];
		lsr.add(new LineStep(c, dphic, phic));
		int ic = lsr.size();
		if (satisfiesWolfe(c, phic, dphic, phi0, dphi0, philim, delta, sigma)) {
			return new int[] { 1, ic, ic };
		}
		int[] AB = update(lsr, n, df, x, d, wa, ia, ib, ic, philim, func, dfunc, fev);
		int iA = AB[0];
		int iB = AB[1];
		a = lsr.get(iA - 1).step;
		b = lsr.get(iB - 1).step;
		if (iB == ic) {
			c = secant(lsr, ib, iB);
		} else if (iA == ic) {
			c = secant(lsr, ia, iA);
		}
		if (c >= a && c <= b) {
			val = eval(n, df, x, d, c, wa, true, func, dfunc);
			phic = val[0];
			dphic = val[1];
			++fev[0];
			lsr.add(new LineStep(c, dphic, phic));
			ic = lsr.size();
			if (satisfiesWolfe(c, phic, dphic, phi0, dphi0, philim, delta, sigma)) {
				return new int[] { 1, ic, ic };
			}
			AB = update(lsr, n, df, x, d, wa, iA, iB, ic, philim, func, dfunc, fev);
			iA = AB[0];
			iB = AB[1];
		}
		return new int[] { 0, iA, iB };
	}

	private static boolean satisfiesWolfe(final double c, final double phic, final double dphic, final double phi0,
			final double dphi0, final double philm, final double delta, final double sigma) {

		final boolean wolfe1 = delta * dphi0 >= (phic - phi0) / c && dphic >= sigma * dphi0;
		final boolean wolfe2 = (2.0 * delta - 1.0) * dphi0 >= dphic && dphic >= sigma * dphi0 && phic <= philm;
		return wolfe1 || wolfe2;
	}

	private static int[] update(final List<LineStep> lsr, final int n, final double[] df, final double[] x,
			final double[] d, final double[] wa, final int ia, final int ib, final int ic, final double philim,
			final Function<? super double[], Double> func, final Function<? super double[], double[]> dfunc,
			final int[] fev) {
		double a = lsr.get(ia - 1).step;
		double b = lsr.get(ib - 1).step;
		double c = lsr.get(ic - 1).step;
		double phic = lsr.get(ic - 1).value;
		double dphic = lsr.get(ic - 1).slope;
		if (c < a || c > b) {
			return new int[] { ia, ib };
		} else if (dphic >= 0.0) {
			return new int[] { ia, ic };
		} else if (phic <= philim) {
			return new int[] { ic, ib };
		} else {
			return bisect(lsr, n, df, x, d, wa, ia, ic, philim, func, dfunc, fev);
		}
	}

	private static int[] bisect(final List<LineStep> lsr, final int n, final double[] df, final double[] x,
			final double[] d, final double[] wa, int ia, int ib, final double philim,
			final Function<? super double[], Double> func, final Function<? super double[], double[]> dfunc,
			final int[] fev) {
		double a = lsr.get(ia - 1).step;
		double b = lsr.get(ib - 1).step;
		while (b - a > Math.ulp(b)) {
			final double dd = (a + b) / 2.0;
			final double[] val = eval(n, df, x, d, dd, wa, true, func, dfunc);
			final double phid = val[0];
			final double gphi = val[1];
			++fev[0];
			lsr.add(new LineStep(dd, gphi, phid));
			final int id = lsr.size();
			if (gphi >= 0.0) {
				return new int[] { ia, id };
			} else if (phid <= philim) {
				a = dd;
				ia = id;
			} else {
				b = dd;
				ib = id;
			}
		}
		return new int[] { ia, ib };
	}

	private static double[] eval(final int n, final double[] df, final double[] x, final double[] d, final double step,
			final double[] wa, final boolean grad, final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc) {
		BlasMath.daxpy1(n, step, d, 1, x, 1, wa, 1);
		double gphi = Double.NaN;
		double phi;
		if (grad) {
			phi = func.apply(wa);
			System.arraycopy(dfunc.apply(wa), 0, df, 0, n);
			if (Double.isFinite(phi)) {
				gphi = BlasMath.ddotm(n, df, 1, d, 1);
			}
		} else {
			phi = func.apply(wa);
		}
		return new double[] { phi, gphi };
	}

	private static double secant(final double a, final double b, final double dphia, final double dphib) {
		return (a * dphib - b * dphia) / (dphib - dphia);
	}

	private static double secant(final List<LineStep> lsr, final int ia, final int ib) {
		return secant(lsr.get(ia - 1).step, lsr.get(ib - 1).step, lsr.get(ia - 1).slope, lsr.get(ib - 1).slope);
	}

	private static double nextFloat(final double x) {
		return x + Math.ulp(x);
	}
}
