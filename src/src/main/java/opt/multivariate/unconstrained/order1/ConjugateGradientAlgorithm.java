package opt.multivariate.unconstrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import utils.BlasMath;
import utils.RealMath;

/**
 *
 * @author Michael
 */
public final class ConjugateGradientAlgorithm extends GradientOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int myMaxEvals, myMethod;

	// saved variables for the line search routine
	final double[] stx = new double[1], sty = new double[1], fx = new double[1], fy = new double[1],
			dgx = new double[1], dgy = new double[1], fxm = new double[1], fym = new double[1], dgxm = new double[1],
			dgym = new double[1];

	final int[] infoc = new int[1];

	final boolean[] stage1 = new boolean[1], brackt = new boolean[1];

	double dg, dgm, dgtest, finit, ftest1, fm, stmin, stmax, width, width1;

	// saved variables for the main subroutine
	final double[] dgout = new double[1], stp = new double[1], dgfam = new double[1];

	final int[] info = new int[1], nfev = new int[1];

	double gtol, gnorm, stp1, ftol, xtol, stpmin, stpmax, beta, betafr, betapr, dg0, gg, gg0, dgold, dg1;
	int mp, lp, iter, nfun, maxfev, nrst, ides, ndes;
	boolean isnew, finish;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 * @param method
	 */
	public ConjugateGradientAlgorithm(final double tolerance, final int maxEvaluations, final int method) {
		super(tolerance);
		myMethod = method;
		myMaxEvals = maxEvaluations;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public ConjugateGradientAlgorithm(final double tolerance, final int maxEvaluations) {
		this(tolerance, maxEvaluations, 3);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess) {

		// prepare variables
		final int[] evals = new int[1];

		// call main subroutine
		final double[] result = main(f, df, guess, myTol, evals);
		myEvals += evals[0];
		myGEvals += evals[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private double[] main(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final double[] guess, final double eps,
			final int[] evals) {
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		double[] g = new double[n];
		final double[] d = new double[n];
		final double[] gold = new double[n];
		final double[] w = new double[n];
		final double[] f = new double[1];
		final int[] iflag = { 0 };
		final boolean[] finmain = { false };
		boolean do20 = true;

		evals[0] = 0;
		while (true) {
			if (do20) {

				// Calculate the function and gradient values here
				f[0] = func.apply(x);
				++evals[0];
				g = dfunc.apply(x);
			}

			// Call the main optimization code
			cgfam(n, x, f, g, d, gold, eps, w, iflag, 1, myMethod, finmain);

			// check the flag
			if (iflag[0] <= 0) {
				return iflag[0] < 0 ? null : x;
			} else if (evals[0] >= myMaxEvals) {
				return null;
			} else if (iflag[0] == 1) {
				do20 = true;
			} else {

				// Termination Test.
				final double tlev = eps * (1.0 + Math.abs(f[0]));
				int i = 0;
				while (true) {
					++i;
					if (i > n) {
						finmain[0] = true;
						do20 = false;
						break;
					} else if (Math.abs(g[i - 1]) > tlev) {
						do20 = false;
						break;
					}
				}
			}
		}
	}

	private void cgfam(final int n, final double[] x, final double[] f, final double[] g, final double[] d,
			final double[] gold, final double eps, final double[] w, final int[] iflag, final int irest,
			final int method, final boolean[] finish) {

		// IFLAG = 1 INDICATES A RE-ENTRY WITH NEW FUNCTION VALUES
		if (iflag[0] != 1) {

			// IFLAG = 2 INDICATES A RE-ENTRY WITH A NEW ITERATE
			if (iflag[0] == 2) {
				if (finish[0]) {
					iflag[0] = 0;
					return;
				}
			} else {

				// INITIALIZE
				iter = 0;
				if (n <= 0) {
					iflag[0] = -3;
					return;
				}
				nfun = 1;
				isnew = true;
				nrst = ndes = 0;
				BlasMath.dscal1(n, -1.0, g, 1, d, 1);
				gnorm = BlasMath.denorm(n, g);
				stp1 = 1.0 / gnorm;

				// PARAMETERS FOR LINE SEARCH ROUTINE
				ftol = 1.0e-4;
				gtol = 1.0e-1;
				if (gtol <= 1.0e-4) {
					gtol = 1.0e-2;
				}
				xtol = 1.0e-17;
				stpmin = 1.0e-20;
				stpmax = 1.0e+20;
				maxfev = 40;
			}

			// MAIN ITERATION LOOP
			// WHEN NRST>N AND IREST=1 THEN RESTART
			++iter;
			++nrst;
			info[0] = nfev[0] = 0;
			System.arraycopy(g, 0, gold, 0, n);
			dgold = dgfam[0] = BlasMath.ddotm(n, d, 1, g, 1);
			stp[0] = 1.0;

			// Shanno-Phua's Formula For Trial Step
			if (!isnew) {
				stp[0] = dg0 / dgfam[0];
			}
			if (iter == 1) {
				stp[0] = stp1;
			}
			ides = 0;
			isnew = false;
		}

		while (true) {

			// CALL THE LINE SEARCH ROUTINE OF MOR'E AND THUENTE
			// (modified for our CG method)
			cvsmod(n, x, f, g, d, stp, ftol, gtol, xtol, stpmin, stpmax, maxfev, info, nfev, w, dgfam, dgout);
			if (info[0] == -1) {

				// RETURN TO FETCH FUNCTION AND GRADIENT
				iflag[0] = 1;
				return;
			}
			if (info[0] != 1) {
				iflag[0] = -1;
				return;
			}

			// TEST IF DESCENT DIRECTION IS OBTAINED FOR METHODS 2 AND 3
			gg = BlasMath.ddotm(n, g, 1, g, 1);
			gg0 = BlasMath.ddotm(n, g, 1, gold, 1);
			betapr = (gg - gg0) / gnorm / gnorm;
			if (irest == 1 && nrst > n) {
				nrst = 0;
				isnew = true;
			} else if (method == 1) {
			} else {
				dg1 = -gg + betapr * dgout[0];
				if (dg1 >= 0.0) {
					++ides;
					if (ides > 5) {
						iflag[0] = -2;
						return;
					} else {
						continue;
					}
				}
			}
			break;
		}

		// DETERMINE CORRECT BETA VALUE FOR METHOD CHOSEN
		nfun += nfev[0];
		ndes += ides;
		betafr = gg / gnorm / gnorm;
		if (nrst == 0) {
			beta = 0.0;
		} else {
			switch (method) {
			case 1:
				beta = betafr;
				break;
			case 2:
				beta = betapr;
				break;
			default:
				beta = Math.max(0.0, betapr);
				break;
			}
		}

		// COMPUTE THE NEW DIRECTION
		for (int i = 1; i <= n; ++i) {
			d[i - 1] = -g[i - 1] + beta * d[i - 1];
		}
		dg0 = dgold * stp[0];

		// RETURN TO DRIVER FOR TERMINATION TEST
		gnorm = BlasMath.denorm(n, g);
		iflag[0] = 2;
	}

	private void cvsmod(final int n, final double[] x, final double[] f, final double[] g, final double[] s,
			final double[] stp, final double ftol, final double gtol, final double xtol, final double stpmin,
			final double stpmax, final int maxfev, final int[] info, final int[] nfev, final double[] wa,
			final double[] dginit, final double[] dgout) {
		final double p5 = 0.5, p66 = 0.66, xtrapf = 4.0;
		int flag;
		if (info[0] == -1) {
			flag = 45;
		} else if (info[0] == 1) {
			flag = 321;
		} else {
			infoc[0] = 1;

			// CHECK THE INPUT PARAMETERS FOR ERRORS.
			if (n <= 0 || stp[0] <= 0.0 || ftol < 0.0 || gtol < 0.0 || xtol < 0.0 || stpmin < 0.0 || stpmax < stpmin
					|| maxfev <= 0) {
				return;
			}

			// COMPUTE THE INITIAL GRADIENT IN THE SEARCH DIRECTION
			// AND CHECK THAT S IS A DESCENT DIRECTION.
			if (dginit[0] >= 0.0) {
				return;
			}

			// INITIALIZE LOCAL VARIABLES.
			brackt[0] = false;
			stage1[0] = true;
			nfev[0] = 0;
			finit = f[0];
			dgtest = ftol * dginit[0];
			width = stpmax - stpmin;
			width1 = width / p5;
			System.arraycopy(x, 0, wa, 0, n);
			stx[0] = sty[0] = 0.0;
			fx[0] = fy[0] = finit;
			dgx[0] = dgy[0] = dginit[0];
			flag = 30;
		}

		while (true) {
			if (flag == 30) {

				// START OF ITERATION.
				// SET THE MINIMUM AND MAXIMUM STEPS TO CORRESPOND
				// TO THE PRESENT INTERVAL OF UNCERTAINTY.
				if (brackt[0]) {
					stmin = Math.min(stx[0], sty[0]);
					stmax = Math.max(stx[0], sty[0]);
				} else {
					stmin = stx[0];
					stmax = stp[0] + xtrapf * (stp[0] - stx[0]);
				}

				// FORCE THE STEP TO BE WITHIN THE BOUNDS STPMAX AND STPMIN.
				stp[0] = Math.max(stp[0], stpmin);
				stp[0] = Math.min(stp[0], stpmax);

				// IF AN UNUSUAL TERMINATION IS TO OCCUR THEN LET
				// STP BE THE LOWEST POINT OBTAINED SO FAR.
				if ((brackt[0] && (stp[0] <= stmin || stp[0] >= stmax)) || nfev[0] >= maxfev - 1 || infoc[0] == 0
						|| (brackt[0] && stmax - stmin <= xtol * stmax)) {
					stp[0] = stx[0];
				}

				// EVALUATE THE FUNCTION AND GRADIENT AT STP
				// AND COMPUTE THE DIRECTIONAL DERIVATIVE.
				BlasMath.daxpy1(n, stp[0], s, 1, wa, 1, x, 1);

				// Return to compute function value
				info[0] = -1;
				return;
			}

			if (flag == 45) {
				info[0] = 0;
				++nfev[0];
				dg = BlasMath.ddotm(n, g, 1, s, 1);
				ftest1 = finit + stp[0] * dgtest;

				// TEST FOR CONVERGENCE.
				if ((brackt[0] && (stp[0] <= stmin || stp[0] >= stmax)) || infoc[0] == 0) {
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
				if (brackt[0] && stmax - stmin <= xtol * stmax) {
					info[0] = 2;
				}

				// More's code has been modified so that at least one new
				// function value is computed during the line search (enforcing
				// at least one interpolation is not easy, since the code may
				// override an interpolation)
				if (f[0] <= ftest1 && Math.abs(dg) <= gtol * (-dginit[0]) && nfev[0] > 1) {
					info[0] = 1;
				}

				// CHECK FOR TERMINATION.
				if (info[0] != 0) {
					dgout[0] = dg;
					return;
				}
				flag = 321;
			}

			if (flag == 321) {

				// IN THE FIRST STAGE WE SEEK A STEP FOR WHICH THE MODIFIED
				// FUNCTION HAS A NONPOSITIVE VALUE AND NONNEGATIVE DERIVATIVE.
				if (stage1[0] && f[0] <= ftest1 && dg >= Math.min(ftol, gtol) * dginit[0]) {
					stage1[0] = false;
				}

				// A MODIFIED FUNCTION IS USED TO PREDICT THE STEP
				if (stage1[0] && f[0] <= fx[0] && f[0] > ftest1) {

					// DEFINE THE MODIFIED FUNCTION AND DERIVATIVE VALUES.
					fm = f[0] - stp[0] * dgtest;
					fxm[0] = fx[0] - stx[0] * dgtest;
					fym[0] = fy[0] - sty[0] * dgtest;
					dgm = dg - dgtest;
					dgxm[0] = dgx[0] - dgtest;
					dgym[0] = dgy[0] - dgtest;

					// CALL CSTEPM TO UPDATE THE INTERVAL OF UNCERTAINTY
					// AND TO COMPUTE THE NEW STEP.
					cstepm(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax, infoc);

					// RESET THE FUNCTION AND GRADIENT VALUES FOR F.
					fx[0] = fxm[0] + stx[0] * dgtest;
					fy[0] = fym[0] + sty[0] * dgtest;
					dgx[0] = dgxm[0] + dgtest;
					dgy[0] = dgym[0] + dgtest;
				} else {

					// CALL CSTEPM TO UPDATE THE INTERVAL OF UNCERTAINTY
					// AND TO COMPUTE THE NEW STEP.
					cstepm(stx, fx, dgx, sty, fy, dgy, stp, f[0], dg, brackt, stmin, stmax, infoc);
				}

				// FORCE A SUFFICIENT DECREASE IN THE SIZE OF THE
				// INTERVAL OF UNCERTAINTY.
				if (brackt[0]) {
					if (Math.abs(sty[0] - stx[0]) >= p66 * width1) {
						stp[0] = stx[0] + p5 * (sty[0] - stx[0]);
					}
					width1 = width;
					width = Math.abs(sty[0] - stx[0]);
				}

				// END OF ITERATION.
				flag = 30;
			}
		}
	}

	private static void cstepm(final double[] stx, final double[] fx, final double[] dx, final double[] sty,
			final double[] fy, final double[] dy, final double[] stp, final double fp, final double dp,
			final boolean[] brackt, final double stpmin, final double stpmax, final int[] info) {
		final double p66 = 0.66;
		info[0] = 0;

		// check input parameters
		if ((brackt[0] && (stp[0] <= Math.min(stx[0], sty[0]) || stp[0] >= Math.max(stx[0], sty[0])))
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
			final double s = RealMath.maxAbs(theta, dx[0], dp);
			double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] < stx[0]) {
				gamma = -gamma;
			}
			final double p = (gamma - dx[0]) + theta;
			final double q = ((gamma - dx[0]) + gamma) + dp;
			final double r = p / q;
			final double stpc = stx[0] + r * (stp[0] - stx[0]);
			final double temp = (dx[0] / ((fx[0] - fp) / (stp[0] - stx[0]) + dx[0])) / 2;
			final double stpq = stx[0] + temp * (stp[0] - stx[0]);
			if (Math.abs(stpc - stx[0]) < Math.abs(stpq - stx[0])) {
				stpf = stpc;
			} else {
				stpf = stpc + (stpq - stpc) / 2;
			}
			brackt[0] = true;
		} else if (sgnd < 0.0) {
			info[0] = 2;
			bound = 0;
			final double theta = 3.0 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			final double s = RealMath.maxAbs(theta, dx[0], dp);
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
			brackt[0] = true;
		} else if (Math.abs(dp) < Math.abs(dx[0])) {
			info[0] = 3;
			bound = 1;
			final double theta = 3.0 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			final double s = RealMath.maxAbs(theta, dx[0], dp);
			double gamma = (theta / s) * (theta / s) - (dx[0] / s) * (dp / s);
			gamma = s * Math.sqrt(Math.max(0.0, gamma));
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
			if (brackt[0] == true) {
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
			if (brackt[0] == true) {
				final double theta = 3.0 * (fp - fy[0]) / (sty[0] - stp[0]) + dy[0] + dp;
				final double s = RealMath.maxAbs(theta, dy[0], dp);
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
		if (brackt[0] && bound == 1) {
			if (sty[0] > stx[0]) {
				stp[0] = Math.min(stx[0] + p66 * (sty[0] - stx[0]), stp[0]);
			} else {
				stp[0] = Math.max(stx[0] + p66 * (sty[0] - stx[0]), stp[0]);
			}
		}
	}
}
