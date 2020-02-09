/*
Minpack Copyright Notice (1999) University of Chicago.  All rights reserved

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above
copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials
provided with the distribution.

3. The end-user documentation included with the
redistribution, if any, must include the following
acknowledgment:

   "This product includes software developed by the
   University of Chicago, as Operator of Argonne National
   Laboratory.

Alternately, this acknowledgment may appear in the software
itself, if and wherever such third-party acknowledgments
normally appear.

4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS"
WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDER, THE
UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND
THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE
OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY
OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF
THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4)
DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION
UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL
BE CORRECTED.

5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT
HOLDER, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF
ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF
ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER
SUCH LIABILITY IS ASSERTED ON THE BASIS OF CONTRACT, TORT
(INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE,
EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
POSSIBILITY OF SUCH LOSS OR DAMAGES.    
*/
package opt.linesearch;

import java.util.Arrays;
import java.util.function.Function;

import utils.BlasMath;
import utils.Pair;

/**
 * A translation of the approximate line search routine by More and Thuente
 * (1994) from the MINPACK project. This should be the preferred line search
 * subroutine in general purpose optimization, along with Hager-Zhang.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Moré, Jorge J., and David J. Thuente. "Line search algorithms with
 * guaranteed sufficient decrease." ACM Transactions on Mathematical Software
 * (TOMS) 20.3 (1994): 286-307.
 */
public final class MoreThuenteLineSearch extends LineSearch {

	private final double myFTol, myGTol, myXTol, myMinStep, myMaxStep;

	/**
	 *
	 * @param sufficientDecrease
	 * @param curvature
	 * @param stepTolerance
	 * @param minstep
	 * @param maxstep
	 * @param maxevals
	 */
	public MoreThuenteLineSearch(final double sufficientDecrease, final double curvature, final double stepTolerance,
			final double minstep, final double maxstep, final int maxevals) {
		super(1e-6, maxevals);
		myFTol = sufficientDecrease;
		myGTol = curvature;
		myXTol = stepTolerance;
		myMinStep = minstep;
		myMaxStep = maxstep;

		// validate input parameters
		if (myFTol < 0.0 || myGTol < 0.0 || myXTol < 0.0 || myMinStep < 0.0 || myMaxStep < myMinStep
				|| myMaxIters <= 0) {
			throw new IllegalArgumentException("Illegal parameter in line search.");
		}
	}

	/**
	 *
	 * @param stepTolerance
	 * @param minstep
	 * @param maxstep
	 * @param maxevals
	 */
	public MoreThuenteLineSearch(final double stepTolerance, final double minstep, final double maxstep,
			final int maxevals) {
		this(C1, 0.9, stepTolerance, minstep, maxstep, maxevals);
	}

	/**
	 *
	 * @param stepTolerance
	 * @param maxevals
	 */
	public MoreThuenteLineSearch(final double stepTolerance, final int maxevals) {
		this(stepTolerance, 1e-20, 1e+20, maxevals);
	}

	@Override
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			double f0, final double initial) {

		// initialize data
		final int n = x0.length;
		final double[] fin = { f0 };
		final double[] stpin = { initial };
		final double[] x0copy = Arrays.copyOf(x0, x0.length);
		final double[] dircopy = Arrays.copyOf(dir, dir.length);
		final double[] df0copy = Arrays.copyOf(df0, df0.length);
		final double[] wa = new double[x0.length];
		final int[] info = new int[] { 0 };
		final int[] nfev = new int[] { 0 };

		// call search
		cvsrch(f, df, n, x0copy, fin, df0copy, dircopy, stpin, myFTol, myGTol, myXTol, myMinStep, myMaxStep, myMaxIters,
				info, nfev, wa);
		myEvals += nfev[0];
		myDEvals += nfev[0];
		return new Pair<>(stpin[0], x0copy);
	}

	private static void cvsrch(final Function<? super double[], Double> fcn,
			final Function<? super double[], double[]> dfcn, final int n, final double[] x, final double[] f,
			final double[] g, final double[] s, final double[] stp, final double ftol, final double gtol,
			final double xtol, final double stpmin, final double stpmax, final int maxfev, final int[] info,
			final int[] nfev, final double[] wa) {

		final double p5 = 0.5, p66 = 0.66, xtrapf = 4.0;
		final double[] stx = new double[1], sty = new double[1], fx = new double[1], fy = new double[1],
				dgx = new double[1], dgy = new double[1], fxm = new double[1], fym = new double[1],
				dgxm = new double[1], dgym = new double[1];
		final int[] brackt = new int[1], infoc = new int[1];

		info[0] = 0;

		// check for input errors
		if (stp[0] <= 0.0 || ftol < 0.0 || gtol < 0.0 || xtol < 0.0 || stpmin < 0.0 || stpmax < stpmin || maxfev <= 0) {
			return;
		}

		// compute initial gradient
		double dginit = BlasMath.ddotm(n, g, 1, s, 1);
		if (dginit >= 0.0) {
			return;
		}

		// initialize local variables
		brackt[0] = 0;
		infoc[0] = 1;
		nfev[0] = 0;
		int stage1 = 1;
		final double finit = f[0];
		final double dgtest = ftol * dginit;
		double width = stpmax - stpmin;
		double width1 = 2.0 * width;
		System.arraycopy(x, 0, wa, 0, n);

		stx[0] = sty[0] = 0.0;
		fx[0] = fy[0] = finit;
		dgx[0] = dgy[0] = dginit;

		// beginning of iteration
		while (true) {

			// set min and max steps to current interval
			final double stmin, stmax;
			if (brackt[0] == 1) {
				stmin = Math.min(stx[0], sty[0]);
				stmax = Math.max(stx[0], sty[0]);
			} else {
				stmin = stx[0];
				stmax = stp[0] + xtrapf * (stp[0] - stx[0]);
			}

			// force step to within bounds
			stp[0] = Math.max(stp[0], stpmin);
			stp[0] = Math.min(stp[0], stpmax);

			// unusual termination
			if ((brackt[0] == 1 && (stp[0] <= stmin || stp[0] >= stmax)) || nfev[0] >= maxfev - 1 || infoc[0] == 0
					|| (brackt[0] == 1 && stmax - stmin <= xtol * stmax)) {
				stp[0] = stx[0];
			}

			// evaluate function and gradient
			BlasMath.daxpy1(n, stp[0], s, 1, wa, 1, x, 1);
			f[0] = fcn.apply(x);
			System.arraycopy(dfcn.apply(x), 0, g, 0, n);
			++nfev[0];

			final double ftest1 = finit + stp[0] * dgtest;
			final double dg = BlasMath.ddotm(n, g, 1, s, 1);

			// test convergence
			if ((brackt[0] == 1 && (stp[0] <= stmin || stp[0] >= stmax)) || infoc[0] == 0) {
				info[0] = 6;
			}
			if (stp[0] == stpmax && f[0] <= ftest1 && dg <= dgtest) {
				info[0] = 5;
			}
			if (stp[0] == stpmin && (f[0] > ftest1 || dg >= dgtest)) {
				info[0] = 4;
			}
			if (nfev[0] >= maxfev) {
				info[0] = 3;
			}
			if (brackt[0] == 1 && stmax - stmin <= xtol * stmax) {
				info[0] = 2;
			}
			if (f[0] <= ftest1 && Math.abs(dg) <= gtol * (-dginit)) {
				info[0] = 1;
			}

			// check for termination
			if (info[0] != 0) {
				return;
			}

			// look for a new step
			if (stage1 == 1 && f[0] <= ftest1 && dg >= Math.min(ftol, gtol) * dginit) {
				stage1 = 0;
			}

			if (stage1 == 1 && f[0] <= fx[0] && f[0] > ftest1) {

				// modified function/derivative values
				fxm[0] = fx[0] - stx[0] * dgtest;
				fym[0] = fy[0] - sty[0] * dgtest;
				dgxm[0] = dgx[0] - dgtest;
				dgym[0] = dgy[0] - dgtest;
				final double fm = f[0] - stp[0] * dgtest;
				final double dgm = dg - dgtest;

				// call cstep
				cstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax, infoc);

				// reset function and gradient values
				fx[0] = fxm[0] + stx[0] * dgtest;
				fy[0] = fym[0] + sty[0] * dgtest;
				dgx[0] = dgxm[0] + dgtest;
				dgy[0] = dgym[0] + dgtest;
			} else {

				// call cstep
				cstep(stx, fx, dgx, sty, fy, dgy, stp, f[0], dg, brackt, stmin, stmax, infoc);
			}

			// force a sufficient decrease in interval size
			if (brackt[0] == 1) {
				if (Math.abs(sty[0] - stx[0]) >= p66 * width1) {
					stp[0] = stx[0] + p5 * (sty[0] - stx[0]);
				}
				width1 = width;
				width = Math.abs(sty[0] - stx[0]);
			}
		}
	}

	private static void cstep(final double[] stx, final double[] fx, final double[] dx, final double[] sty,
			final double[] fy, final double[] dy, final double[] stp, final double fp, final double dp,
			final int[] brackt, final double stpmin, final double stpmax, final int[] info) {
		final double p66 = 0.66;
		info[0] = 0;

		// check input parameters
		if ((brackt[0] == 1 && (stp[0] <= Math.min(stx[0], sty[0]) || stp[0] >= Math.max(stx[0], sty[0])))
				|| dx[0] * (stp[0] - stx[0]) >= 0.0 || stpmax < stpmin) {
			return;
		}

		// determine whether derivatives have opposite sign
		final double sgnd = dp * (dx[0] / Math.abs(dx[0]));

		// bracketing
		final int bound;
		double stpf;
		if (fp > fx[0]) {

			info[0] = 1;
			bound = 1;
			final double theta = 3.0 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			final double s = max3(theta, dx[0], dp);
			double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] < stx[0]) {
				gamma = -gamma;
			}
			final double p = (gamma - dx[0]) + theta;
			final double q = ((gamma - dx[0]) + gamma) + dp;
			final double r = p / q;
			final double stpc = stx[0] + r * (stp[0] - stx[0]);
			final double temp = (fx[0] - fp) / (stp[0] - stx[0]) + dx[0];
			final double stpq = stx[0] + ((dx[0] / (temp)) / 2) * (stp[0] - stx[0]);
			if (Math.abs(stpc - stx[0]) < Math.abs(stpq - stx[0])) {
				stpf = stpc;
			} else {
				stpf = stpc + (stpq - stpc) / 2;
			}
			brackt[0] = 1;
		} else if (sgnd < 0.0) {

			info[0] = 2;
			bound = 0;
			final double theta = 3.0 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			final double s = max3(theta, dx[0], dp);
			double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] > stx[0]) {
				gamma = -gamma;
			}
			final double p = (gamma - dp) + theta;
			final double q = ((gamma - dp) + gamma) + dx[0];
			final double r = p / q;
			final double stpc = stp[0] + r * (stx[0] - stp[0]);
			final double stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0]);
			if (Math.abs(stpc - stp[0]) > Math.abs(stpq - stp[0])) {
				stpf = stpc;
			} else {
				stpf = stpq;
			}
			brackt[0] = 1;
		} else if (Math.abs(dp) < Math.abs(dx[0])) {

			info[0] = 3;
			bound = 1;
			final double theta = 3.0 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			final double s = max3(theta, dx[0], dp);
			final double temp = (theta / s) * (theta / s) - (dx[0] / s) * (dp / s);
			double gamma = s * Math.sqrt(Math.max(0.0, temp));
			if (stp[0] > stx[0]) {
				gamma = -gamma;
			}
			final double p = (gamma - dp) + theta;
			final double q = (gamma + (dx[0] - dp)) + gamma;
			final double r = p / q;
			final double stpc;
			if (r < 0.0 && gamma != 0.0) {
				stpc = stp[0] + r * (stx[0] - stp[0]);
			} else if (stp[0] > stx[0]) {
				stpc = stpmax;
			} else {
				stpc = stpmin;
			}
			final double stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0]);
			if (brackt[0] == 1) {
				if (Math.abs(stp[0] - stpc) < Math.abs(stp[0] - stpq)) {
					stpf = stpc;
				} else {
					stpf = stpq;
				}
			} else if (Math.abs(stp[0] - stpc) > Math.abs(stp[0] - stpq)) {
				stpf = stpc;
			} else {
				stpf = stpq;
			}
		} else {

			info[0] = 4;
			bound = 0;
			if (brackt[0] == 1) {
				final double theta = 3.0 * (fp - fy[0]) / (sty[0] - stp[0]) + dy[0] + dp;
				final double s = max3(theta, dy[0], dp);
				double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dy[0] / s) * (dp / s));
				if (stp[0] > sty[0]) {
					gamma = -gamma;
				}
				final double p = (gamma - dp) + theta;
				final double q = ((gamma - dp) + gamma) + dy[0];
				final double r = p / q;
				final double stpc = stp[0] + r * (sty[0] - stp[0]);
				stpf = stpc;
			} else if (stp[0] > stx[0]) {
				stpf = stpmax;
			} else {
				stpf = stpmin;
			}
		}

		// update interval
		if (fp > fx[0]) {
			sty[0] = stp[0];
			fy[0] = fp;
			dy[0] = dp;
		} else {
			if (sgnd < 0.0) {
				sty[0] = stx[0];
				fy[0] = fx[0];
				dy[0] = dx[0];
			}
			stx[0] = stp[0];
			fx[0] = fp;
			dx[0] = dp;
		}

		// compute new step and safeguard
		stpf = Math.min(stpmax, stpf);
		stpf = Math.max(stpmin, stpf);
		stp[0] = stpf;
		if (brackt[0] == 1 && bound == 1) {
			if (sty[0] > stx[0]) {
				stp[0] = Math.min(stx[0] + p66 * (sty[0] - stx[0]), stp[0]);
			} else {
				stp[0] = Math.max(stx[0] + p66 * (sty[0] - stx[0]), stp[0]);
			}
		}
	}

	private static double max3(final double a, final double b, final double c) {
		return Math.max(Math.abs(a), Math.max(Math.abs(b), Math.abs(c)));
	}
}
