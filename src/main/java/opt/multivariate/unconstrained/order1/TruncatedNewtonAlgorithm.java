/*
 * The original code does not have any licensing information, but based on online
 * sources, it is likely in the public domain.
 * 
 * Copyright for the original TNBC Fortran routines:
 * TRUNCATED-NEWTON METHOD:  SUBROUTINES
 * WRITTEN BY:  STEPHEN G. NASH
 * SCHOOL OF INFORMATION TECHNOLOGY & ENGINEERING
 * EORGE MASON UNIVERSITY
 * FAIRFAX, VA 22030
 * 
 * [1] https://fossies.org/linux/scipy/scipy/optimize/tnc/README
 */
package opt.multivariate.unconstrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.GradientOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.BlasMath;
import utils.Constants;

/**
 * An implementation of the truncated Newton optimization of a general
 * non-linear differentiable function without. The original code was written by
 * Stephen Nash and translated to Java.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Nash, Stephen G. "A survey of truncated-Newton methods." Journal of
 * computational and applied mathematics 124.1-2 (2000): 45-59.
 * 
 * [2] Nash, Stephen G. "Preconditioning of truncated-Newton methods." SIAM
 * Journal on Scientific and Statistical Computing 6.3 (1985): 599-616.
 */
public final class TruncatedNewtonAlgorithm extends GradientOptimizer {

	@FunctionalInterface
	private interface Sfun {

		double apply(int n, double[] x, int xi, double[] g, int gi);
	}

	private final int myMaxIters, myMaxEvals;
	private final double myEta, myStepMax;

	// COMMON/SUBSCR/
	private int lgv, lz1, lzk, lv, lsk, lyk, ldiagb, lsr, lyr, loldg, lhg, lhyk, lpk, lemat, lwtest;
	private int myEvals, myGEvals;

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 * @param maxEvals
	 * @param lineSearchParameter
	 * @param lineSearchMaxStepSize
	 */
	public TruncatedNewtonAlgorithm(final double tolerance, final int maxIterations, final int maxEvals,
			final double lineSearchParameter, final double lineSearchMaxStepSize) {
		super(tolerance);
		myMaxIters = maxIterations;
		myMaxEvals = maxEvals;
		myEta = lineSearchParameter;
		myStepMax = lineSearchMaxStepSize;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 * @param maxEvals
	 */
	public TruncatedNewtonAlgorithm(final double tolerance, final int maxIterations, final int maxEvals) {
		this(tolerance, maxIterations, maxEvals, 0.25, 10.0);
	}

	@Override
	public final MultivariateOptimizerSolution optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess) {

		// prepare variables
		final int mgslvl = 1;
		final int n = guess.length;
		final int lw = 14 * n;
		final double accrcy = 100.0 * mchpr1();
		final double xtol = Math.max(myTol, Math.sqrt(accrcy));
		final int[] err = new int[1];
		final double[] x = Arrays.copyOf(guess, n);
		final double[] fx = { f.apply(x) };
		final double[] g = df.apply(x);
		final double[] w = new double[lw];
		myEvals = myGEvals = 1;

		// prepare function
		final Sfun sfun = (nn, xx, xi, gg, gi) -> {
			final double[] x1 = new double[nn];
			System.arraycopy(xx, xi - 1, x1, 0, nn);
			final double[] g1 = df.apply(x1);
			System.arraycopy(g1, 0, gg, gi - 1, nn);
			return f.apply(x1);
		};

		// call main subroutine
		lmqn(err, n, x, fx, g, w, lw, sfun, mgslvl, myMaxIters, myMaxEvals, myEta, myStepMax, accrcy, xtol);
		return new MultivariateOptimizerSolution(x, myEvals, myGEvals, err[0] == 0 || err[0] == 3);
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param f
	 * @param df
	 * @param guess
	 * @param low
	 * @param up
	 * @return
	 */
	public final MultivariateOptimizerSolution optimize(final Function<double[], Double> f,
			final Function<double[], double[]> df, final double[] guess, final double[] low, final double[] up) {

		// prepare variables
		final int mgslvl = 1;
		final int n = guess.length;
		final int lw = 14 * n;
		final double accrcy = 100.0 * mchpr1();
		final double xtol = Math.max(myTol, Math.sqrt(accrcy));
		final int[] err = new int[1];
		final int[] ipivot = new int[n];
		final double[] x = Arrays.copyOf(guess, n);
		final double[] fx = { f.apply(x) };
		final double[] g = df.apply(x);
		final double[] w = new double[lw];
		myEvals = myGEvals = 1;

		// prepare function
		final Sfun sfun = (nn, xx, xi, gg, gi) -> {
			final double[] x1 = new double[nn];
			System.arraycopy(xx, xi - 1, x1, 0, nn);
			final double[] g1 = df.apply(x1);
			System.arraycopy(g1, 0, gg, gi - 1, nn);
			return f.apply(x1);
		};

		// call main subroutine
		lmqnbc(err, n, x, fx, g, w, lw, sfun, low, up, ipivot, mgslvl, myMaxIters, myMaxEvals, myEta, myStepMax, accrcy,
				xtol);
		return new MultivariateOptimizerSolution(x, myEvals, myGEvals, err[0] == 0 || err[0] == 3);
	}

	private void lmqn(final int[] ifail, final int n, final double[] x, final double[] f, final double[] g,
			final double[] w, final int lw, final Sfun sfun, final int msglvl, final int maxit, final int maxfun,
			final double eta, final double stepmx, final double accrcy, final double xtol) {

		boolean lreset = false, upd1;
		int i, isk, idiagb, ireset, icycle, ioldg, ipk, iyk, lhyr, modet, nm1, nmodif;
		double difnew, difold, epsred, fkeep, fold, fstop, gnorm, oldgtp, one = 1.0, pe, pnorm, rtleps, tnytol, toleps,
				yksk = 0.0, yrsr = 0.0, zero = 0.0;
		final int[] ier = new int[1], nwhy = new int[1], nftotl = new int[1], niter = new int[1], nfeval = new int[1],
				nlincg = new int[1], numf = new int[1], ipivot = new int[n];
		final double[] alpha = new double[1], epsmch = new double[1], peps = new double[1], rteps = new double[1],
				rtol = new double[1], rtolsq = new double[1], ftest = new double[1], xnorm = new double[1],
				small = new double[1], tiny = new double[1], fnew = new double[1], fm = new double[1],
				gtg = new double[1], oldf = new double[1], gsk = new double[1], gtpnew = new double[1],
				spe = new double[1], reltol = new double[1], abstol = new double[1], flast = new double[1];

		// INITIALIZE PARAMETERS AND CONSTANTS
		setpar(n);
		upd1 = true;
		ireset = nfeval[0] = nmodif = nlincg[0] = 0;
		fstop = f[0];
		nm1 = n - 1;

		// WITHIN THIS ROUTINE THE ARRAY W(LOLDG) IS SHARED BY W(LHYR)
		lhyr = loldg;

		// CHECK PARAMETERS AND SET CONSTANTS
		chkucp(lwtest, maxfun, nwhy, n, alpha, epsmch, eta, peps, rteps, rtol, rtolsq, stepmx, ftest, xtol, xnorm, x,
				lw, small, tiny, accrcy);
		if (nwhy[0] < 0) {

			// SET IFAIL
			ifail[0] = nwhy[0];
			return;
		}
		setucr(small[0], nftotl, niter, n, f[0], fnew, fm, gtg, oldf, sfun, g, x);
		++myEvals;
		++myGEvals;
		fold = fnew[0];

		// CHECK FOR SMALL GRADIENT AT THE STARTING POINT
		ftest[0] = one + Math.abs(fnew[0]);
		if (gtg[0] < 1.0e-4 * epsmch[0] * ftest[0] * ftest[0]) {
			ifail[0] = 0;
			f[0] = fnew[0];
			return;
		}

		// SET INITIAL VALUES TO OTHER PARAMETERS
		icycle = nm1;
		toleps = rtol[0] + rteps[0];
		rtleps = rtolsq[0] + epsmch[0];
		gnorm = Math.sqrt(gtg[0]);
		difnew = zero;
		epsred = 5.0e-2;
		fkeep = fnew[0];

		// SET THE DIAGONAL OF THE APPROXIMATE HESSIAN TO UNITY
		idiagb = ldiagb;
		for (i = 1; i <= n; ++i) {
			w[idiagb - 1] = one;
			++idiagb;
		}

		// ..................START OF MAIN ITERATIVE LOOP..........
		// COMPUTE THE NEW SEARCH DIRECTION
		modet = msglvl - 3;
		modlnp(modet, w, lpk, w, lgv, w, lz1, w, lv, w, ldiagb, w, lemat, x, g, w, lzk, n, w, lw, niter, maxit, nfeval,
				nmodif, nlincg, upd1, yksk, gsk, yrsr, lreset, sfun, false, ipivot, accrcy, gtpnew, gnorm, xnorm[0]);

		while (true) {

			dcopy(n, g, 1, w, loldg);
			pnorm = BlasMath.dnrm2(n, w, lpk, 1);
			oldf[0] = fnew[0];
			oldgtp = gtpnew[0];

			// PREPARE TO COMPUTE THE STEP LENGTH
			pe = pnorm + epsmch[0];

			// COMPUTE THE ABSOLUTE AND RELATIVE TOLERANCES FOR THE LINEAR SEARCH
			reltol[0] = rteps[0] * (xnorm[0] + one) / pe;
			abstol[0] = -epsmch[0] * ftest[0] / (oldgtp - epsmch[0]);

			// COMPUTE THE SMALLEST ALLOWABLE SPACING BETWEEN POINTS IN
			// THE LINEAR SEARCH
			tnytol = epsmch[0] * (xnorm[0] + one) / pe;
			spe[0] = stepmx / pe;

			// SET THE INITIAL STEP LENGTH
			alpha[0] = step1(fnew[0], fm[0], oldgtp, spe[0]);

			// PERFORM THE LINEAR SEARCH
			linder(n, sfun, small[0], epsmch[0], reltol, abstol, tnytol, eta, zero, spe[0], w, lpk, oldgtp, x, fnew,
					alpha, g, numf, nwhy, w, lw);
			fold = fnew[0];
			++niter[0];
			nftotl[0] += numf[0];
			gtg[0] = BlasMath.ddotm(n, g, 1, g, 1);
			if (nwhy[0] < 0) {

				// SET IFAIL
				ifail[0] = nwhy[0];
				return;
			}
			if (nwhy[0] != 0 && nwhy[0] != 2) {

				// THE LINEAR SEARCH HAS FAILED TO FIND A LOWER POINT
				nwhy[0] = 3;
				oldf[0] = fnew[0];

				// LOCAL SEARCH HERE COULD BE INSTALLED HERE
				f[0] = oldf[0];

				// SET IFAIL
				ifail[0] = nwhy[0];
				return;
			}

			if (nwhy[0] > 1) {
				fnew[0] = sfun.apply(n, x, 1, g, 1);
				++nftotl[0];
				++myEvals;
				++myGEvals;
			}

			// TERMINATE IF MORE THAN MAXFUN EVALUTATIONS HAVE BEEN MADE
			nwhy[0] = 2;
			if (nftotl[0] > maxfun) {

				// LOCAL SEARCH HERE COULD BE INSTALLED HERE
				f[0] = oldf[0];

				// SET IFAIL
				ifail[0] = nwhy[0];
				return;
			}
			nwhy[0] = 0;

			// SET UP PARAMETERS USED IN CONVERGENCE AND RESETTING TESTS
			difold = difnew;
			difnew = oldf[0] - fnew[0];

			// IF THIS IS THE FIRST ITERATION OF A NEW CYCLE, COMPUTE THE
			// PERCENTAGE REDUCTION FACTOR FOR THE RESETTING TEST
			if (icycle == 1) {
				if (difnew > 2.0 * difold) {
					epsred = 2.0 * epsred;
				}
				if (difnew < 0.5 * difold) {
					epsred = 0.5 * epsred;
				}
			}

			gnorm = Math.sqrt(gtg[0]);
			ftest[0] = one + Math.abs(fnew[0]);
			xnorm[0] = BlasMath.dnrm2(n, x, 1, 1);

			// TEST FOR CONVERGENCE
			if ((alpha[0] * pnorm < toleps * (one + xnorm[0]) && Math.abs(difnew) < rtleps * ftest[0]
					&& gtg[0] < peps[0] * ftest[0] * ftest[0]) || gtg[0] < 1.0e-4 * accrcy * ftest[0] * ftest[0]) {
				ifail[0] = 0;
				f[0] = fnew[0];
				return;
			}

			// COMPUTE THE CHANGE IN THE ITERATES AND THE CORRESPONDING CHANGE
			// IN THE GRADIENTS
			isk = lsk;
			ipk = lpk;
			iyk = lyk;
			ioldg = loldg;
			for (i = 1; i <= n; ++i) {
				w[iyk - 1] = g[i - 1] - w[ioldg - 1];
				w[isk - 1] = alpha[0] * w[ipk - 1];
				++ipk;
				++isk;
				++iyk;
				++ioldg;
			}

			// SET UP PARAMETERS USED IN UPDATING THE DIRECTION OF SEARCH
			yksk = BlasMath.ddotm(n, w, lyk, w, lsk);
			lreset = false;
			if (icycle == nm1 || difnew < epsred * (fkeep - fnew[0])) {
				lreset = true;
			}
			if (!lreset) {
				yrsr = BlasMath.ddotm(n, w, lyr, w, lsr);
				if (yrsr <= zero) {
					lreset = true;
				}
			}
			upd1 = false;

			// COMPUTE THE NEW SEARCH DIRECTION
			modet = msglvl - 3;
			modlnp(modet, w, lpk, w, lgv, w, lz1, w, lv, w, ldiagb, w, lemat, x, g, w, lzk, n, w, lw, niter, maxit,
					nfeval, nmodif, nlincg, upd1, yksk, gsk, yrsr, lreset, sfun, false, ipivot, accrcy, gtpnew, gnorm,
					xnorm[0]);
			if (lreset) {

				// RESET
				++ireset;

				// INITIALIZE THE SUM OF ALL THE CHANGES IN X
				dcopy(n, w, lsk, w, lsr);
				dcopy(n, w, lyk, w, lyr);
				fkeep = fnew[0];
				icycle = 1;
			} else {

				// STORE THE ACCUMULATED CHANGE IN THE POINT AND GRADIENT AS AN
				// "AVERAGE" DIRECTION FOR PRECONDITIONING
				BlasMath.dxpym(n, w, lsk, w, lsr);
				BlasMath.dxpym(n, w, lyk, w, lyr);
				++icycle;
			}
		}
		// ...............END OF MAIN ITERATION.......................
	}

	private void lmqnbc(final int[] ifail, final int n, final double[] x, final double[] f, final double[] g,
			final double[] w, final int lw, final Sfun sfun, final double[] low, final double[] up, final int[] ipivot,
			final int msglvl, final int maxit, final int maxfun, final double eta, final double stepmx,
			final double accrcy, final double xtol) {

		int i, isk, ireset, icycle, idiagb, ioldg, ipk, iyk, lhyr, nmodif, nm1, modet;
		double difnew, difold, epsred, fkeep, fold, fstop, gnorm, oldgtp, one = 1.0, pe, pnorm, rtleps, tnytol, toleps,
				yksk = 0.0, yrsr = 0.0, zero = 0.0;
		boolean lreset = false, upd1, newcon;
		final double[] alpha = new double[1], epsmch = new double[1], peps = new double[1], rteps = new double[1],
				rtol = new double[1], rtolsq = new double[1], ftest = new double[1], xnorm = new double[1],
				small = new double[1], tiny = new double[1], fnew = new double[1], fm = new double[1],
				gtg = new double[1], oldf = new double[1], gsk = new double[1], gtpnew = new double[1],
				spe = new double[1], reltol = new double[1], abstol = new double[1], flast = new double[1];
		final int[] ier = new int[1], nwhy = new int[1], nftotl = new int[1], niter = new int[1], nfeval = new int[1],
				nlincg = new int[1], numf = new int[1];
		final boolean[] conv = new boolean[1];

		// CHECK THAT INITIAL X IS FEASIBLE AND THAT THE BOUNDS ARE CONSISTENT
		crash(n, x, ipivot, low, up, ier);
		if (ier[0] != 0) {
			return;
		}

		// INITIALIZE VARIABLES
		setpar(n);
		upd1 = true;
		ireset = nfeval[0] = nmodif = nlincg[0] = 0;
		fstop = f[0];
		conv[0] = false;
		nm1 = n - 1;

		// WITHIN THIS ROUTINE THE ARRAY W(LOLDG) IS SHARED BY W(LHYR)
		lhyr = loldg;

		// CHECK PARAMETERS AND SET CONSTANTS
		chkucp(lwtest, maxfun, nwhy, n, alpha, epsmch, eta, peps, rteps, rtol, rtolsq, stepmx, ftest, xtol, xnorm, x,
				lw, small, tiny, accrcy);
		if (nwhy[0] < 0) {

			// SET IFAIL
			ifail[0] = nwhy[0];
			return;
		}
		setucr(small[0], nftotl, niter, n, f[0], fnew, fm, gtg, oldf, sfun, g, x);
		++myGEvals;
		++myEvals;

		// fold = fnew[0]; //appears to be unnecessary
		flast[0] = fnew[0];

		// TEST THE LAGRANGE MULTIPLIERS TO SEE IF THEY ARE NON-NEGATIVE.
		// BECAUSE THE CONSTRAINTS ARE ONLY LOWER BOUNDS, THE COMPONENTS
		// OF THE GRADIENT CORRESPONDING TO THE ACTIVE CONSTRAINTS ARE THE
		// LAGRANGE MULTIPLIERS. AFTERWORDS, THE PROJECTED GRADIENT IS FORMED
		for (i = 1; i <= n; ++i) {
			if (ipivot[i - 1] == 2 || -ipivot[i - 1] * g[i - 1] >= 0.0) {
				continue;
			}
			ipivot[i - 1] = 0;
		}
		ztime(n, g, 1, ipivot);
		gtg[0] = BlasMath.ddotm(n, g, 1, g, 1);

		// CHECK IF THE INITIAL POINT IS A LOCAL MINIMUM
		ftest[0] = one + Math.abs(fnew[0]);
		if (gtg[0] < 1.0e-4 * epsmch[0] * ftest[0] * ftest[0]) {
			ifail[0] = 0;
			f[0] = fnew[0];
			return;
		}

		// SET INITIAL VALUES TO OTHER PARAMETERS
		icycle = nm1;
		toleps = rtol[0] + rteps[0];
		rtleps = rtolsq[0] + epsmch[0];
		gnorm = Math.sqrt(gtg[0]);
		difnew = zero;
		epsred = 5.0e-2;
		fkeep = fnew[0];

		// SET THE DIAGONAL OF THE APPROXIMATE HESSIAN TO UNITY
		idiagb = ldiagb;
		for (i = 1; i <= n; ++i) {
			w[idiagb - 1] = one;
			++idiagb;
		}

		// ..................START OF MAIN ITERATIVE LOOP..........
		// COMPUTE THE NEW SEARCH DIRECTION
		modet = msglvl - 3;
		modlnp(modet, w, lpk, w, lgv, w, lz1, w, lv, w, ldiagb, w, lemat, x, g, w, lzk, n, w, lw, niter, maxit, nfeval,
				nmodif, nlincg, upd1, yksk, gsk, yrsr, lreset, sfun, true, ipivot, accrcy, gtpnew, gnorm, xnorm[0]);
		while (true) {

			dcopy(n, g, 1, w, loldg);
			pnorm = BlasMath.dnrm2(n, w, lpk, 1);
			oldf[0] = fnew[0];
			oldgtp = gtpnew[0];

			// PREPARE TO COMPUTE THE STEP LENGTH
			pe = pnorm + epsmch[0];

			// COMPUTE THE ABSOLUTE AND RELATIVE TOLERANCES FOR THE LINEAR SEARCH
			reltol[0] = rteps[0] * (xnorm[0] + one) / pe;
			abstol[0] = -epsmch[0] * ftest[0] / (oldgtp - epsmch[0]);

			// COMPUTE THE SMALLEST ALLOWABLE SPACING BETWEEN POINTS IN
			// THE LINEAR SEARCH
			tnytol = epsmch[0] * (xnorm[0] + one) / pe;
			stpmax(stepmx, pe, spe, n, x, w, lpk, ipivot, low, up);

			// SET THE INITIAL STEP LENGTH
			alpha[0] = step1(fnew[0], fm[0], oldgtp, spe[0]);

			// PERFORM THE LINEAR SEARCH
			linder(n, sfun, small[0], epsmch[0], reltol, abstol, tnytol, eta, zero, spe[0], w, lpk, oldgtp, x, fnew,
					alpha, g, numf, nwhy, w, lw);
			newcon = false;
			if (Math.abs(alpha[0] - spe[0]) <= 10.0 * epsmch[0]) {
				newcon = true;
				nwhy[0] = 0;
				modz(n, x, w, lpk, ipivot, epsmch[0], low, up, flast, fnew[0]);
				flast[0] = fnew[0];
			}
			fold = fnew[0];
			++niter[0];
			nftotl[0] += numf[0];

			// IF REQUIRED, PRINT THE DETAILS OF THIS ITERATION
			if (nwhy[0] < 0) {

				// SET IFAIL
				ifail[0] = nwhy[0];
				return;
			}
			if (nwhy[0] != 0 && nwhy[0] != 2) {

				// THE LINEAR SEARCH HAS FAILED TO FIND A LOWER POINT
				nwhy[0] = 3;
				oldf[0] = fnew[0];

				// LOCAL SEARCH COULD BE INSTALLED HERE
				f[0] = fold;

				// SET IFAIL
				ifail[0] = nwhy[0];
				return;
			}
			if (nwhy[0] > 1) {
				fnew[0] = sfun.apply(n, x, 1, g, 1);
				++nftotl[0];
				++myEvals;
				++myGEvals;
			}

			// TERMINATE IF MORE THAN MAXFUN EVALUATIONS HAVE BEEN MADE
			nwhy[0] = 2;
			if (nftotl[0] > maxfun) {

				// LOCAL SEARCH COULD BE INSTALLED HERE
				f[0] = fold;

				// SET IFAIL
				ifail[0] = nwhy[0];
				return;
			}
			nwhy[0] = 0;

			// SET UP PARAMETERS USED IN CONVERGENCE AND RESETTING TESTS
			difold = difnew;
			difnew = oldf[0] - fnew[0];

			// IF THIS IS THE FIRST ITERATION OF A NEW CYCLE, COMPUTE THE
			// PERCENTAGE REDUCTION FACTOR FOR THE RESETTING TEST
			if (icycle == 1) {
				if (difnew > 2.0 * difold) {
					epsred *= 2.0;
				}
				if (difnew < 0.5 * difold) {
					epsred *= 0.5;
				}
			}
			dcopy(n, g, 1, w, lgv);
			ztime(n, w, lgv, ipivot);
			gtg[0] = BlasMath.ddotm(n, w, lgv, w, lgv);
			gnorm = Math.sqrt(gtg[0]);
			ftest[0] = one + Math.abs(fnew[0]);
			xnorm[0] = BlasMath.dnrm2(n, x, 1, 1);

			// TEST FOR CONVERGENCE
			cnvtst(conv, alpha[0], pnorm, toleps, xnorm[0], difnew, rtleps, ftest[0], gtg[0], peps[0], epsmch[0],
					gtpnew[0], fnew[0], flast, g, ipivot, n, accrcy);
			if (conv[0]) {
				ifail[0] = 0;
				f[0] = fnew[0];
				return;
			}
			ztime(n, g, 1, ipivot);

			// COMPUTE THE CHANGE IN THE ITERATES AND THE CORRESPONDING CHANGE
			// IN THE GRADIENTS
			if (!newcon) {
				isk = lsk;
				ipk = lpk;
				iyk = lyk;
				ioldg = loldg;
				for (i = 1; i <= n; ++i) {
					w[iyk - 1] = g[i - 1] - w[ioldg - 1];
					w[isk - 1] = alpha[0] * w[ipk - 1];
					++ipk;
					++isk;
					++iyk;
					++ioldg;
				}

				// SET UP PARAMETERS USED IN UPDATING THE PRECONDITIONING STRATEGY
				yksk = BlasMath.ddotm(n, w, lyk, w, lsk);
				lreset = false;
				if (icycle == nm1 || difnew < epsred * (fkeep - fnew[0])) {
					lreset = true;
				}
				if (!lreset) {
					yrsr = BlasMath.ddotm(n, w, lyr, w, lsr);
					if (yrsr <= zero) {
						lreset = true;
					}
				}
				upd1 = false;
			}

			// COMPUTE THE NEW SEARCH DIRECTION
			modet = msglvl - 3;
			modlnp(modet, w, lpk, w, lgv, w, lz1, w, lv, w, ldiagb, w, lemat, x, g, w, lzk, n, w, lw, niter, maxit,
					nfeval, nmodif, nlincg, upd1, yksk, gsk, yrsr, lreset, sfun, true, ipivot, accrcy, gtpnew, gnorm,
					xnorm[0]);
			if (newcon) {
				continue;
			}
			if (lreset) {

				// RESET
				++ireset;

				// INITIALIZE THE SUM OF ALL THE CHANGES IN X
				dcopy(n, w, lsk, w, lsr);
				dcopy(n, w, lyk, w, lyr);
				fkeep = fnew[0];
				icycle = 1;
			} else {

				// COMPUTE THE ACCUMULATED STEP AND ITS CORRESPONDING
				// GRADIENT DIFFERENCE
				BlasMath.dxpym(n, w, lsk, w, lsr);
				BlasMath.dxpym(n, w, lyk, w, lyr);
				++icycle;
			}
		}
		// ...............END OF MAIN ITERATION.......................
	}

	private static void stpmax(final double stepmx, final double pe, final double[] spe, final int n, final double[] x,
			final double[] p, final int ip, final int[] ipivot, final double[] low, final double[] up) {

		double t;
		int i;

		// COMPUTE THE MAXIMUM ALLOWABLE STEP LENGTH
		spe[0] = stepmx / pe;

		// SPE IS THE STANDARD (UNCONSTRAINED) MAX STEP
		for (i = 1; i <= n; ++i) {
			if (ipivot[i - 1] != 0 || p[i - 1 + ip - 1] == 0.0) {
				continue;
			}
			if (p[i - 1 + ip - 1] > 0.0) {
				t = up[i - 1] - x[i - 1];
				if (t < spe[0] * p[i - 1 + ip - 1]) {
					spe[0] = t / p[i - 1 + ip - 1];
				}
			} else {
				t = low[i - 1] - x[i - 1];
				if (t > spe[0] * p[i - 1 + ip - 1]) {
					spe[0] = t / p[i - 1 + ip - 1];
				}
			}
		}
	}

	private static void modz(final int n, final double[] x, final double[] p, final int ip, final int[] ipivot,
			final double epsmch, final double[] low, final double[] up, final double[] flast, final double fnew) {

		double tol;
		int i;

		// UPDATE THE CONSTRAINT MATRIX IF A NEW CONSTRAINT IS ENCOUNTERED
		for (i = 1; i <= n; ++i) {
			if (ipivot[i - 1] != 0 || p[i - 1 + ip - 1] == 0.0) {
				continue;
			}
			if (p[i - 1 + ip - 1] > 0.0) {
				tol = 10.0 * epsmch * (Math.abs(up[i - 1]) + 1.0);
				if (up[i - 1] - x[i - 1] <= tol) {
					flast[0] = fnew;
					ipivot[i - 1] = 1;
					x[i - 1] = up[i - 1];
				}
			} else {
				tol = 10.0 * epsmch * (Math.abs(low[i - 1]) + 1.0);
				if (x[i - 1] - low[i - 1] <= tol) {
					flast[0] = fnew;
					ipivot[i - 1] = -1;
					x[i - 1] = low[i - 1];
				}
			}
		}
	}

	private static void cnvtst(final boolean[] conv, final double alpha, final double pnorm, final double toleps,
			final double xnorm, final double difnew, final double rtleps, final double ftest, final double gtg,
			final double peps, final double epsmch, final double gtpnew, final double fnew, final double[] flast,
			final double[] g, final int[] ipivot, final int n, final double accrcy) {

		double one = 1.0, cmax, t;
		int i, imax;
		boolean ltest;

		// TEST FOR CONVERGENCE
		imax = 0;
		cmax = 0.0;
		ltest = flast[0] - fnew <= -0.5 * gtpnew;
		for (i = 1; i <= n; ++i) {
			if (ipivot[i - 1] == 0 || ipivot[i - 1] == 2) {
				continue;
			}
			t = -ipivot[i - 1] * g[i - 1];
			if (t >= 0.0) {
				continue;
			}
			conv[0] = false;
			if (ltest || cmax <= t) {
				continue;
			}
			cmax = t;
			imax = i;
		}

		if (imax == 0) {
			conv[0] = false;
			if ((alpha * pnorm >= toleps * (one + xnorm) || Math.abs(difnew) >= rtleps * ftest
					|| gtg >= peps * ftest * ftest) && gtg >= 1e-4 * accrcy * ftest * ftest) {
				return;
			}
			conv[0] = true;
		} else {
			ipivot[imax - 1] = 0;
			flast[0] = fnew;
		}
	}

	private static void crash(final int n, final double[] x, final int[] ipivot, final double[] low, final double[] up,
			final int[] ier) {

		int i;

		ier[0] = 0;
		for (i = 1; i <= n; ++i) {
			if (x[i - 1] < low[i - 1]) {
				x[i - 1] = low[i - 1];
			}
			if (x[i - 1] > up[i - 1]) {
				x[i - 1] = up[i - 1];
			}
			ipivot[i - 1] = 0;
			if (x[i - 1] == low[i - 1]) {
				ipivot[i - 1] = -1;
			}
			if (x[i - 1] == up[i - 1]) {
				ipivot[i - 1] = 1;
			}
			if (up[i - 1] == low[i - 1]) {
				ipivot[i - 1] = 2;
			}
			if (low[i - 1] > up[i - 1]) {
				ier[0] = -i;
			}
		}
	}

	private void modlnp(final int modet, final double[] zsol, final int izsol, final double[] gv, final int igv,
			final double[] r, final int ir, final double[] v, final int iv, final double[] diagb, final int idiagb,
			final double[] emat, final int iemat, final double[] x, final double[] g, final double[] zk, final int izk,
			final int n, final double[] w, final int lw, final int[] niter, final int maxit, final int[] nfeval,
			final int nmodif, final int[] nlincg, final boolean upd1, final double yksk, final double[] gsk,
			final double yrsr, final boolean lreset, final Sfun sfun, final boolean bounds, final int[] ipivot,
			final double accrcy, final double[] gtp, final double gnorm, final double xnorm) {

		final boolean[] first = new boolean[1];
		int i, k;
		double alpha, beta = 0.0, pr, qold, qnew, qtest, rhsnrm, rnorm, rz, rzold = 0.0, tol, vgv;
		final double[] delta = new double[1];

		// GENERAL INITIALIZATION
		if (maxit == 0) {
			return;
		}
		first[0] = true;
		rhsnrm = gnorm;
		tol = 1.0e-12;
		qold = 0.0;

		// INITIALIZATION FOR PRECONDITIONED CONJUGATE-GRADIENT ALGORITHM
		initpc(diagb, idiagb, emat, iemat, n, w, lw, modet, upd1, yksk, gsk[0], yrsr, lreset);
		for (i = 1; i <= n; ++i) {
			r[i - 1 + ir - 1] = -g[i - 1];
			v[i - 1 + iv - 1] = zsol[i - 1 + izsol - 1] = 0.0;
		}

		// MAIN ITERATION
		for (k = 1; k <= maxit; ++k) {
			++nlincg[0];

			// CG ITERATION TO SOLVE SYSTEM OF EQUATIONS
			if (bounds) {
				ztime(n, r, ir, ipivot);
			}
			msolve(r, ir, zk, izk, n, w, lw, upd1, yksk, gsk, yrsr, lreset, first[0]);
			if (bounds) {
				ztime(n, zk, izk, ipivot);
			}
			rz = BlasMath.ddotm(n, r, ir, zk, izk);
			if (rz / rhsnrm < tol) {

				if (k <= 1) {
					dcopy(n, g, 1, zsol, izsol);
					negvec(n, zsol, izsol);
					if (bounds) {
						ztime(n, zsol, izsol, ipivot);
					}
					gtp[0] = BlasMath.ddotm(n, zsol, izsol, g, 1);
				}

				// STORE (OR RESTORE) DIAGONAL PRECONDITIONING
				dcopy(n, emat, iemat, diagb, idiagb);
				return;
			}
			if (k == 1) {
				beta = 0.0;
			}
			if (k > 1) {
				beta = rz / rzold;
			}
			for (i = 1; i <= n; ++i) {
				v[i - 1 + iv - 1] = zk[i - 1 + izk - 1] + beta * v[i - 1 + iv - 1];
			}
			if (bounds) {
				ztime(n, v, iv, ipivot);
			}
			gtims(v, iv, gv, igv, n, x, g, w, lw, sfun, first, delta, accrcy, xnorm);
			++myEvals;
			++myGEvals;
			if (bounds) {
				ztime(n, gv, igv, ipivot);
			}
			++nfeval[0];
			vgv = BlasMath.ddotm(n, v, iv, gv, igv);
			if (vgv / rhsnrm < tol) {

				if (k <= 1) {
					msolve(g, 1, zsol, izsol, n, w, lw, upd1, yksk, gsk, yrsr, lreset, first[0]);
					negvec(n, zsol, izsol);
					if (bounds) {
						ztime(n, zsol, izsol, ipivot);
					}
					gtp[0] = BlasMath.ddotm(n, zsol, izsol, g, 1);
				}

				// STORE (OR RESTORE) DIAGONAL PRECONDITIONING
				dcopy(n, emat, iemat, diagb, idiagb);
				return;
			}
			ndia3(n, emat, iemat, v, iv, gv, igv, r, ir, vgv, modet);

			// COMPUTE LINEAR STEP LENGTH
			alpha = rz / vgv;

			// COMPUTE CURRENT SOLUTION AND RELATED VECTORS
			BlasMath.daxpym(n, alpha, v, iv, zsol, izsol);
			BlasMath.daxpym(n, -alpha, gv, igv, r, ir);

			// TEST FOR CONVERGENCE
			gtp[0] = BlasMath.ddotm(n, zsol, izsol, g, 1);
			pr = BlasMath.ddotm(n, r, ir, zsol, izsol);
			qnew = 0.5 * (gtp[0] + pr);
			qtest = k * (1.0 - qold / qnew);
			if (qtest < 0.0) {

				// STORE (OR RESTORE) DIAGONAL PRECONDITIONING
				dcopy(n, emat, iemat, diagb, idiagb);
				return;
			}
			qold = qnew;
			if (qtest <= 0.5) {

				// STORE (OR RESTORE) DIAGONAL PRECONDITIONING
				dcopy(n, emat, iemat, diagb, idiagb);
				return;
			}

			// PERFORM CAUTIONARY TEST
			if (gtp[0] > 0) {

				// TRUNCATE ALGORITHM IN CASE OF AN EMERGENCY
				BlasMath.daxpym(n, -alpha, v, iv, zsol, izsol);
				gtp[0] = BlasMath.ddotm(n, zsol, izsol, g, 1);

				// STORE (OR RESTORE) DIAGONAL PRECONDITIONING
				dcopy(n, emat, iemat, diagb, idiagb);
				return;
			}
			rzold = rz;
		}

		// TERMINATE ALGORITHM
		--k;

		// STORE (OR RESTORE) DIAGONAL PRECONDITIONING
		dcopy(n, emat, iemat, diagb, idiagb);
	}

	private static void ztime(final int n, final double[] x, final int ix, final int[] ipivot) {
		for (int i = 1; i <= n; ++i) {
			if (ipivot[i - 1] != 0) {
				x[i - 1 + ix - 1] = 0.0;
			}
		}
	}

	private static void ndia3(final int n, final double[] e, final int ie, final double[] v, final int iv,
			final double[] gv, final int igv, final double[] r, final int ir, final double vgv, final int modet) {
		double vr = BlasMath.ddotm(n, v, iv, r, ir);
		for (int i = 1; i <= n; ++i) {
			e[i - 1 + ie - 1] = e[i - 1 + ie - 1] - r[i - 1 + ir - 1] * r[i - 1 + ir - 1] / vr
					+ gv[i - 1 + igv - 1] * gv[i - 1 + igv - 1] / vgv;
			if (e[i - 1 + ie - 1] <= 1e-6) {
				e[i - 1 + ie - 1] = 1.0;
			}
		}
	}

	private static void negvec(final int n, final double[] v, final int iv) {
		BlasMath.dscalm(n, -1.0, v, iv);
	}

	private static double step1(final double fnew, final double fm, final double gtp, final double smax) {
		double epsmch = mchpr1();
		double d = Math.abs(fnew - fm);
		double alpha = 1.0;
		if (2.0 * d <= (-gtp) && d >= epsmch) {
			alpha = -2.0 * d / gtp;
		}
		if (alpha >= smax) {
			alpha = smax;
		}
		return alpha;
	}

	private static double mchpr1() {
		return Constants.EPSILON;
	}

	private static void chkucp(final int lwtest, final int maxfun, final int[] nwhy, final int n, final double[] alpha,
			final double[] epsmch, final double eta, final double[] peps, final double[] rteps, final double[] rtol,
			final double[] rtolsq, final double stepmx, final double[] test, final double xtol, final double[] xnorm,
			final double[] x, final int lw, final double[] small, final double[] tiny, final double accrcy) {

		// CHECKS PARAMETERS AND SETS CONSTANTS WHICH ARE COMMON TO BOTH
		// DERIVATIVE AND NON-DERIVATIVE ALGORITHMS
		epsmch[0] = mchpr1();
		small[0] = epsmch[0] * epsmch[0];
		tiny[0] = small[0];
		nwhy[0] = -1;
		rteps[0] = Math.sqrt(epsmch[0]);
		rtol[0] = xtol;
		if (Math.abs(rtol[0]) < accrcy) {
			rtol[0] = 10.0 * rteps[0];
		}

		// CHECK FOR ERRORS IN THE INPUT PARAMETERS
		if (lw < lwtest || n < 1 || rtol[0] < 0.0 || eta >= 1.0 || eta < 0.0 || stepmx < rtol[0] || maxfun < 1) {
			return;
		}
		nwhy[0] = 0;

		// SET CONSTANTS FOR LATER
		rtolsq[0] = rtol[0] * rtol[0];
		peps[0] = accrcy * 0.6666;
		xnorm[0] = BlasMath.dnrm2(n, x, 1, 1);
		alpha[0] = test[0] = 0.0;
	}

	private static void setucr(final double small, final int[] nftotl, final int[] niter, final int n, final double f,
			final double[] fnew, final double[] fm, final double[] gtg, final double[] oldf, final Sfun sfun,
			final double[] g, final double[] x) {

		// CHECK INPUT PARAMETERS, COMPUTE THE INITIAL FUNCTION VALUE, SET
		// CONSTANTS FOR THE SUBSEQUENT MINIMIZATION
		fm[0] = f;

		// COMPUTE THE INITIAL FUNCTION VALUE
		fnew[0] = sfun.apply(n, x, 1, g, 1);
		nftotl[0] = 1;

		// SET CONSTANTS FOR LATER
		niter[0] = 0;
		oldf[0] = fnew[0];
		gtg[0] = BlasMath.ddotm(n, g, 1, g, 1);
	}

	private void gtims(final double[] v, final int iv, final double[] gv, final int igv, final int n, final double[] x,
			final double[] g, final double[] w, final int lw, final Sfun sfun, final boolean[] first,
			final double[] delta, final double accrcy, final double xnorm) {
		double dinv, f;
		int i, ihg;

		if (first[0]) {
			delta[0] = Math.sqrt(accrcy) * (1.0 + xnorm);
			first[0] = false;
		}

		dinv = 1.0 / delta[0];
		ihg = lhg;
		for (i = 1; i <= n; ++i) {
			w[ihg - 1] = x[i - 1] + delta[0] * v[i - 1 + iv - 1];
			++ihg;
		}
		f = sfun.apply(n, w, lhg, gv, igv);
		for (i = 1; i <= n; ++i) {
			gv[i - 1 + igv - 1] = (gv[i - 1 + igv - 1] - g[i - 1]) * dinv;
		}
	}

	private void msolve(final double[] g, final int ig, final double[] y, final int iy, final int n, final double[] w,
			final int lw, final boolean upd1, final double yksk, final double[] gsk, final double yrsr,
			final boolean lreset, final boolean first) {
		final int lhyr = loldg;
		mslv(g, ig, y, iy, n, w, lsk, w, lyk, w, ldiagb, w, lsr, w, lyr, w, lhyr, w, lhg, w, lhyk, upd1, yksk, gsk,
				yrsr, lreset, first);
	}

	private static void mslv(final double[] g, final int ig, final double[] y, final int iy, final int n,
			final double[] sk, final int isk, final double[] yk, final int iyk, final double[] diagb, final int idiagb,
			final double[] sr, final int isr, final double[] yr, final int iyr, final double[] hyr, final int ihyr,
			final double[] hg, final int ihg, final double[] hyk, final int ihyk, final boolean upd1, final double yksk,
			final double[] gsk, final double yrsr, final boolean lreset, final boolean first) {
		double rdiagb, ykhyk = 0.0, ghyk, yksr = 0.0, ykhyr = 0.0, yrhyr = 0.0, gsr, ghyr, one = 1.0;
		int i;

		if (upd1) {
			for (i = 1; i <= n; ++i) {
				y[i - 1 + iy - 1] = g[i - 1 + ig - 1] / diagb[i - 1 + idiagb - 1];
			}
			return;
		}
		gsk[0] = BlasMath.ddotm(n, g, ig, sk, isk);
		if (lreset) {

			// COMPUTE GH AND HY WHERE H IS THE INVERSE OF THE DIAGONALS
			for (i = 1; i <= n; ++i) {
				rdiagb = 1.0 / diagb[i - 1 + idiagb - 1];
				hg[i - 1 + ihg - 1] = g[i - 1 + ig - 1] * rdiagb;
				if (first) {
					hyk[i - 1 + ihyk - 1] = yk[i - 1 + iyk - 1] * rdiagb;
				}
			}
			if (first) {
				ykhyk = BlasMath.ddotm(n, yk, iyk, hyk, ihyk);
			}
			ghyk = BlasMath.ddotm(n, g, ig, hyk, ihyk);
			ssbfgs(n, one, sk, isk, yk, iyk, hg, ihg, hyk, ihyk, yksk, ykhyk, gsk[0], ghyk, y, iy);
			return;
		}

		// COMPUTE HG AND HY WHERE H IS THE INVERSE OF THE DIAGONALS
		for (i = 1; i <= n; ++i) {
			rdiagb = 1.0 / diagb[i - 1 + idiagb - 1];
			hg[i - 1 + ihg - 1] = g[i - 1 + ig - 1] * rdiagb;
			if (first) {
				hyk[i - 1 + ihyk - 1] = yk[i - 1 + iyk - 1] * rdiagb;
				hyr[i - 1 + ihyr - 1] = yr[i - 1 + iyr - 1] * rdiagb;
			}
		}
		if (first) {
			yksr = BlasMath.ddotm(n, yk, iyk, sr, isr);
			ykhyr = BlasMath.ddotm(n, yk, iyk, hyr, ihyr);
		}
		gsr = BlasMath.ddotm(n, g, ig, sr, isr);
		ghyr = BlasMath.ddotm(n, g, ig, hyr, ihyr);
		if (first) {
			yrhyr = BlasMath.ddotm(n, yr, iyr, yr, ihyr);
		}
		ssbfgs(n, one, sr, isr, yr, iyr, hg, ihg, hyr, ihyr, yrsr, yrhyr, gsr, ghyr, hg, ihg);
		if (first) {
			ssbfgs(n, one, sr, isr, yr, iyr, hyk, ihyk, hyr, ihyr, yrsr, yrhyr, yksr, ykhyr, hyk, ihyk);
		}
		ykhyk = BlasMath.ddotm(n, hyk, ihyk, yk, iyk);
		ghyk = BlasMath.ddotm(n, hyk, ihyk, g, ig);
		ssbfgs(n, one, sk, isk, yk, iyk, hg, ihg, hyk, ihyk, yksk, ykhyk, gsk[0], ghyk, y, iy);
	}

	private static void ssbfgs(final int n, final double gamma, final double[] sj, final int isj, final double[] yj,
			final int iyj, final double[] hjv, final int ihjv, final double[] hjyj, final int ihjyj, final double yjsj,
			final double yjhyj, final double vsj, final double vhyj, final double[] hjp1v, final int ihjp1v) {

		// SELF-SCALED BFGS
		int i;
		double beta, delta;
		delta = (1.0 + gamma * yjhyj / yjsj) * vsj / yjsj - gamma * vhyj / yjsj;
		beta = -gamma * vsj / yjsj;
		for (i = 1; i <= n; ++i) {
			hjp1v[i - 1 + ihjp1v - 1] = gamma * hjv[i - 1 + ihjv - 1] + delta * sj[i - 1 + isj - 1]
					+ beta * hjyj[i - 1 + ihjyj - 1];
		}
	}

	// ROUTINES TO INITIALIZE PRECONDITIONER
	private void initpc(final double[] diagb, final int idiagb, final double[] emat, final int iemat, final int n,
			final double[] w, final int lw, final int modet, final boolean upd1, final double yksk, final double gsk,
			final double yrsr, final boolean lreset) {
		initp3(diagb, idiagb, emat, iemat, n, lreset, yksk, yrsr, w, lhyk, w, lsk, w, lyk, w, lsr, w, lyr, modet, upd1);
	}

	private static void initp3(final double[] diagb, final int idiagb, final double[] emat, final int iemat,
			final int n, final boolean lreset, final double yksk, final double yrsr, final double[] bsk, final int ibsk,
			final double[] sk, final int isk, final double[] yk, final int iyk, final double[] sr, final int isr,
			final double[] yr, final int iyr, final int modet, final boolean upd1) {

		int i;
		double sds, srds, yrsk, td, d1, dn;

		if (upd1) {

			dcopy(n, diagb, idiagb, emat, iemat);
		} else if (lreset) {

			for (i = 1; i <= n; ++i) {
				bsk[i - 1 + ibsk - 1] = diagb[i - 1 + idiagb - 1] * sk[i - 1 + isk - 1];
			}
			sds = BlasMath.ddotm(n, sk, isk, bsk, ibsk);
			for (i = 1; i <= n; ++i) {
				td = diagb[i - 1 + idiagb - 1];
				emat[i - 1 + iemat - 1] = td - td * td * sk[i - 1 + isk - 1] * sk[i - 1 + isk - 1] / sds
						+ yk[i - 1 + iyk - 1] * yk[i - 1 + iyk - 1] / yksk;
			}
		} else {

			for (i = 1; i <= n; ++i) {
				bsk[i - 1 + ibsk - 1] = diagb[i - 1 + idiagb - 1] * sr[i - 1 + isr - 1];
			}
			sds = BlasMath.ddotm(n, sr, isr, bsk, ibsk);
			srds = BlasMath.ddotm(n, sk, isk, bsk, ibsk);
			yrsk = BlasMath.ddotm(n, yr, iyr, sk, isk);
			for (i = 1; i <= n; ++i) {
				td = diagb[i - 1 + idiagb - 1];
				bsk[i - 1 + ibsk - 1] = td * sk[i - 1 + isk - 1] - bsk[i - 1 + ibsk - 1] * srds / sds
						+ yr[i - 1 + iyr - 1] * yrsk / yrsr;
				emat[i - 1 + iemat - 1] = td - td * td * sr[i - 1 + isr - 1] * sr[i - 1 + isr - 1] / sds
						+ yr[i - 1 + iyr - 1] * yr[i - 1 + iyr - 1] / yrsr;
			}
			sds = BlasMath.ddotm(n, sk, isk, bsk, ibsk);
			for (i = 1; i <= n; ++i) {
				emat[i - 1 + iemat - 1] = emat[i - 1 + iemat - 1] - bsk[i - 1 + ibsk - 1] * bsk[i - 1 + ibsk - 1] / sds
						+ yk[i - 1 + iyk - 1] * yk[i - 1 + iyk - 1] / yksk;
			}
		}

		if (modet < 1) {
			return;
		}
		d1 = emat[1 - 1 + iemat - 1];
		dn = emat[1 - 1 + iemat - 1];
		for (i = 1; i <= n; ++i) {
			if (emat[i - 1 + iemat - 1] < d1) {
				d1 = emat[i - 1 + iemat - 1];
			}
			if (emat[i - 1 + iemat - 1] > dn) {
				dn = emat[i - 1 + iemat - 1];
			}
		}
	}

	private void setpar(final int n) {

		// SET UP PARAMETERS FOR THE OPTIMIZATION ROUTINE
		// COMMON/SUBSCR/ LSUB,LWTEST
		lgv = (1 - 1) * n + 1;
		lz1 = (2 - 1) * n + 1;
		lzk = (3 - 1) * n + 1;
		lv = (4 - 1) * n + 1;
		lsk = (5 - 1) * n + 1;
		lyk = (6 - 1) * n + 1;
		ldiagb = (7 - 1) * n + 1;
		lsr = (8 - 1) * n + 1;
		lyr = (9 - 1) * n + 1;
		loldg = (10 - 1) * n + 1;
		lhg = (11 - 1) * n + 1;
		lhyk = (12 - 1) * n + 1;
		lpk = (13 - 1) * n + 1;
		lemat = (14 - 1) * n + 1;
		lwtest = lemat + n - 1;
	}

	// LINE SEARCH ALGORITHMS OF GILL AND MURRAY
	private void linder(final int n, final Sfun sfun, final double small, final double epsmch, final double[] reltol,
			final double[] abstol, final double tnytol, final double eta, final double sftbnd, final double xbnd,
			final double[] p, final int ip, final double gtp, final double[] x, final double[] f, final double[] alpha,
			final double[] g, final int[] nftotl, final int[] iflag, final double[] w, final int lw) {

		int i, l, lg, lx, numf, itcnt;
		final int[] ientry = new int[1], itest = new int[1];
		double big, fpresn, rmu, rtsmll, ualpha;
		final double[] u = new double[1], fu = new double[1], gu = new double[1], fmin = new double[1],
				gmin = new double[1], xmin = new double[1], xw = new double[1], gw = new double[1], fw = new double[1],
				a = new double[1], b = new double[1], oldf = new double[1], b1 = new double[1], scxbnd = new double[1],
				e = new double[1], step = new double[1], factor = new double[1], gtest1 = new double[1],
				gtest2 = new double[1], tol = new double[1];
		final boolean[] braktd = new boolean[1];

		// ALLOCATE THE ADDRESSES FOR LOCAL WORKSPACE
		lx = 1;
		lg = lx + n;
		rtsmll = Math.sqrt(small);
		big = 1.0 / small;
		itcnt = 0;

		// SET THE ESTIMATED RELATIVE PRECISION IN F(X)
		fpresn = 10.0 * epsmch;
		numf = 0;
		u[0] = alpha[0];
		fu[0] = f[0];
		fmin[0] = f[0];
		gu[0] = gtp;
		rmu = 1.0e-4;

		// FIRST ENTRY SETS UP THE INITIAL INTERVAL OF UNCERTAINTY
		ientry[0] = 1;

		while (true) {

			// TEST FOR TOO MANY ITERATIONS
			++itcnt;
			iflag[0] = 1;
			if (itcnt > 20) {
				return;
			}
			iflag[0] = 0;
			getptc(big, small, rtsmll, reltol, abstol, tnytol, fpresn, eta, rmu, xbnd, u, fu, gu, xmin, fmin, gmin, xw,
					fw, gw, a, b, oldf, b1, scxbnd, e, step, factor, braktd, gtest1, gtest2, tol, ientry, itest);

			// IF ITEST=1, THE ALGORITHM REQUIRES THE FUNCTION VALUE TO BE
			// CALCULATED
			if (itest[0] != 1) {
				break;
			}
			ualpha = xmin[0] + u[0];
			l = lx;
			for (i = 1; i <= n; ++i) {
				w[l - 1] = x[i - 1] + ualpha * p[i - 1 + ip - 1];
				++l;
			}
			fu[0] = sfun.apply(n, w, lx, w, lg);
			++myEvals;
			++myGEvals;
			++numf;
			gu[0] = BlasMath.ddotm(n, w, lg, p, ip);

			// THE GRADIENT VECTOR CORRESPONDING TO THE BEST POINT IS
			// OVERWRITTEN IF FU IS LESS THAN FMIN AND FU IS SUFFICIENTLY
			// LOWER THAN F AT THE ORIGIN
			if (fu[0] <= fmin[0] && fu[0] <= oldf[0] - ualpha * gtest1[0]) {
				dcopy(n, w, lg, g, 1);
			}
		}

		// IF ITEST=2 OR 3 A LOWER POINT COULD NOT BE FOUND
		nftotl[0] = numf;
		iflag[0] = 1;
		if (itest[0] != 0) {
			return;
		}

		// IF ITEST=0 A SUCCESSFUL SEARCH HAS BEEN MADE
		iflag[0] = 0;
		f[0] = fmin[0];
		alpha[0] = xmin[0];
		BlasMath.daxpym(n, alpha[0], p, ip, x, 1);
	}

	private static void getptc(final double big, final double small, final double rtsmall, final double[] reltol,
			final double[] abstol, final double tnytol, final double fpresn, final double eta, final double rmu,
			final double xbnd, final double[] u, final double[] fu, final double[] gu, final double[] xmin,
			final double[] fmin, final double[] gmin, final double[] xw, final double[] fw, final double[] gw,
			final double[] a, final double b[], final double[] oldf, final double[] b1, final double[] scxbnd,
			final double[] e, final double[] step, final double[] factor, final boolean[] braktd, final double[] gtest1,
			final double[] gtest2, final double[] tol, final int[] ientry, final int[] itest) {

		boolean convrg;
		double abgmin, abgw, absr, a1, chordm, chordu, denom, d1, d2, p, q, r, s, scale, sumsq, twotol, xmidpt;
		double zero = 0.0, point1 = 0.1, half = 0.5, one = 1.0, three = 3.0, five = 5.0, eleven = 11.0;

		// BRANCH TO APPROPRIATE SECTION OF CODE DEPENDING ON THE
		// VALUE OF IENTRY
		if (ientry[0] == 1) {

			// CHECK INPUT PARAMETERS
			itest[0] = 2;
			if (u[0] <= zero || xbnd <= tnytol || gu[0] > zero) {
				return;
			}
			itest[0] = 1;
			if (xbnd < abstol[0]) {
				abstol[0] = xbnd;
			}
			tol[0] = abstol[0];
			// twotol = tol[0] + tol[0]; // appears to be unncessary

			// A AND B DEFINE THE INTERVAL OF UNCERTAINTY, X AND XW ARE POINTS
			// WITH LOWEST AND SECOND LOWEST FUNCTION VALUES SO FAR OBTAINED.
			// INITIALIZE A,SMIN,XW AT ORIGIN AND CORRESPONDING VALUES OF
			// FUNCTION AND PROJECTION OF THE GRADIENT ALONG DIRECTION OF SEARCH
			// AT VALUES FOR LATEST ESTIMATE AT MINIMUM
			a[0] = xw[0] = xmin[0] = zero;
			oldf[0] = fmin[0] = fw[0] = fu[0];
			gw[0] = gmin[0] = gu[0];
			step[0] = u[0];
			factor[0] = five;

			// THE MINIMUM HAS NOT YET BEEN BRACKETED
			braktd[0] = false;

			// SET UP XBND AS A BOUND ON THE STEP TO BE TAKEN. (XBND IS NOT COMPUTED
			// EXPLICITLY BUT SCXBND IS ITS SCALED VALUE.) SET THE UPPER BOUND
			// ON THE INTERVAL OF UNCERTAINTY INITIALLY TO XBND + TOL(XBND)
			scxbnd[0] = xbnd;
			b[0] = scxbnd[0] + reltol[0] * Math.abs(scxbnd[0]) + abstol[0];
			e[0] = b[0] + b[0];
			b1[0] = b[0];

			// COMPUTE THE CONSTANTS REQUIRED FOR THE TWO CONVERGENCE CRITERIA
			gtest1[0] = -rmu * gu[0];
			gtest2[0] = -eta * gu[0];

			// SET IENTRY TO INDICATE THAT THIS IS THE FIRST ITERATION
			ientry[0] = 2;
		} else {

			// UPDATE A,B,XW, AND XMIN
			if (fu[0] > fmin[0]) {

				// IF FUNCTION VALUE INCREASED, ORIGIN REMAINS UNCHANGED
				// BUT NEW POINT MAY NOW QUALIFY AS W
				if (u[0] < zero) {
					a[0] = u[0];
				} else {
					b[0] = u[0];
					braktd[0] = true;
				}
				xw[0] = u[0];
				fw[0] = fu[0];
				gw[0] = gu[0];
			} else {

				// IF FUNCTION VALUE NOT INCREASED, NEW POINT BECOMES NEXT
				// ORIGIN AND OTHER POINTS ARE SCALED ACCORDINGLY
				chordu = oldf[0] - (xmin[0] + u[0]) * gtest1[0];
				if (fu[0] > chordu) {

					// THE NEW FUNCTION VALUE DOES NOT SATISFY THE SUFFICIENT
					// DECREASE
					// CRITERION. PREPARE TO MOVE THE UPPER BOUND TO THIS POINT AND
					// FORCE THE INTERPOLATION SCHEME TO EITHER BISECT THE INTERVAL
					// OF
					// UNCERTAINTY OR TAKE THE LINEAR INTERPOLATION STEP WHICH
					// ESTIMATES
					// THE ROOT OF F(ALPHA)=CHORD(ALPHA)
					chordm = oldf[0] - xmin[0] * gtest1[0];
					gu[0] = -gmin[0];
					denom = chordm - fmin[0];
					if (Math.abs(denom) < 1.0e-15) {
						denom = 1.0e-15;
						if (chordm - fmin[0] < 0.0) {
							denom = -denom;
						}
					}
					if (xmin[0] != zero) {
						gu[0] = gmin[0] * (chordu - fu[0]) / denom;
					}
					fu[0] = half * u[0] * (gmin[0] + gu[0]) + fmin[0];
					if (fu[0] < fmin[0]) {
						fu[0] = fmin[0];
					}

					// IF FUNCTION VALUE INCREASED, ORIGIN REMAINS UNCHANGED
					// BUT NEW POINT MAY NOW QUALIFY AS W
					if (u[0] < zero) {
						a[0] = u[0];
					} else {
						b[0] = u[0];
						braktd[0] = true;
					}
					xw[0] = u[0];
					fw[0] = fu[0];
					gw[0] = gu[0];
				} else {
					fw[0] = fmin[0];
					fmin[0] = fu[0];
					gw[0] = gmin[0];
					gmin[0] = gu[0];
					xmin[0] += u[0];
					a[0] -= u[0];
					b[0] -= u[0];
					xw[0] = -u[0];
					scxbnd[0] -= u[0];
					if (gu[0] <= zero) {
						a[0] = zero;
					} else {
						b[0] = zero;
						braktd[0] = true;
					}
					tol[0] = Math.abs(xmin[0]) * reltol[0] + abstol[0];
				}
			}
			twotol = tol[0] + tol[0];
			xmidpt = half * (a[0] + b[0]);

			// CHECK TERMINATION CRITERIA
			convrg = Math.abs(xmidpt) <= twotol - half * (b[0] - a[0]) || Math.abs(gmin[0]) <= gtest2[0]
					&& fmin[0] < oldf[0] && (Math.abs(xmin[0] - xbnd) > tol[0] || !braktd[0]);
			if (convrg) {
				itest[0] = 0;
				if (xmin[0] != zero) {
					return;
				}

				// IF THE FUNCTION HAS NOT BEEN REDUCED, CHECK TO SEE THAT THE
				// RELATIVE
				// CHANGE IN F(X) IS CONSISTENT WITH THE ESTIMATE OF THE DELTA-
				// UNIMODALITY CONSTANT, TOL. IF THE CHANGE IN F(X) IS LARGER THAN
				// EXPECTED, REDUCE THE VALUE OF TOL
				itest[0] = 3;
				if (Math.abs(oldf[0] - fw[0]) <= fpresn * (one + Math.abs(oldf[0]))) {
					return;
				}
				tol[0] *= point1;
				if (tol[0] < tnytol) {
					return;
				}
				reltol[0] *= point1;
				abstol[0] *= point1;
				twotol *= point1;
			}

			// CONTINUE WITH THE COMPUTATION OF A TRIAL STEP LENGTH
			r = q = s = zero;
			if (Math.abs(e[0]) > tol[0]) {

				// FIT CUBIC THROUGH XMIN AND XW
				r = three * (fmin[0] - fw[0]) / xw[0] + gmin[0] + gw[0];
				absr = Math.abs(r);
				q = absr;
				boolean do140;
				if (gw[0] == zero || gmin[0] == zero) {
					do140 = true;
				} else {

					// COMPUTE THE SQUARE ROOT OF (R*R - GMIN*GW) IN A WAY
					// WHICH AVOIDS UNDERFLOW AND OVERFLOW
					abgw = Math.abs(gw[0]);
					abgmin = Math.abs(gmin[0]);
					s = Math.sqrt(abgmin) * Math.sqrt(abgw);
					if ((gw[0] / abgw) * gmin[0] > zero) {

						// COMPUTE THE SQUARE ROOT OF R*R - S*S
						q = Math.sqrt(Math.abs(r + s)) * Math.sqrt(Math.abs(r - s));
						if (r >= s || r <= (-s)) {
							do140 = true;
						} else {
							r = q = zero;
							do140 = false;
						}
					} else {

						// COMPUTE THE SQUARE ROOT OF R*R + S*S
						sumsq = one;
						p = zero;
						if (absr >= s) {

							// THERE IS A POSSIBILITY OF UNDERFLOW
							if (absr > rtsmall) {
								p = absr * rtsmall;
							}
							if (s >= p) {
								sumsq = one + (s / absr) * (s / absr);
							}
							scale = absr;
						} else {

							// THERE IS A POSSIBILITY OF OVERFLOW
							if (s > rtsmall) {
								p = s * rtsmall;
							}
							if (absr >= p) {
								sumsq = one + (absr / s) * (absr / s);
							}
							scale = s;
						}
						sumsq = Math.sqrt(sumsq);
						q = big;
						if (scale < big / sumsq) {
							q = scale * sumsq;
						}
						do140 = true;
					}
				}

				if (do140) {

					// COMPUTE THE MINIMUM OF FITTED CUBIC
					if (xw[0] < zero) {
						q = -q;
					}
					s = xw[0] * (gmin[0] - r - q);
					q = gw[0] - gmin[0] + q + q;
					if (q > zero) {
						s = -s;
					}
					if (q <= zero) {
						q = -q;
					}
					r = e[0];
					if (b1[0] != step[0] || braktd[0]) {
						e[0] = step[0];
					}
				}
			}

			// CONSTRUCT AN ARTIFICIAL BOUND ON THE ESTIMATED STEPLENGTH
			a1 = a[0];
			b1[0] = b[0];
			step[0] = xmidpt;
			if (braktd[0]) {

				// IF THE MINIMUM IS BRACKETED BY 0 AND XW THE STEP MUST LIE
				// WITHIN (A,B)
				if (!(a[0] != zero || xw[0] >= zero) || !(b[0] != zero || xw[0] <= zero)) {

					// IF THE MINIMUM IS NOT BRACKETED BY 0 AND XW THE STEP MUST LIE
					// WITHIN (A1,B1)
					d1 = xw[0];
					d2 = a[0];
					if (a[0] == zero) {
						d2 = b[0];
					}
					u[0] = -d1 / d2;
					step[0] = five * d2 * (point1 + one / u[0]) / eleven;
					if (u[0] < one) {
						step[0] = half * d2 * Math.sqrt(u[0]);
					}
					if (step[0] <= zero) {
						a1 = step[0];
					}
					if (step[0] > zero) {
						b1[0] = step[0];
					}
				}
			} else {
				step[0] = -factor[0] * xw[0];
				if (step[0] > scxbnd[0]) {
					step[0] = scxbnd[0];
				}
				if (step[0] != scxbnd[0]) {
					factor[0] *= five;
				}
				if (step[0] <= zero) {
					a1 = step[0];
				}
				if (step[0] > zero) {
					b1[0] = step[0];
				}
			}

			// REJECT THE STEP OBTAINED BY INTERPOLATION IF IT LIES OUTSIDE THE
			// REQUIRED INTERVAL OR IT IS GREATER THAN HALF THE STEP OBTAINED
			// DURING THE LAST-BUT-ONE ITERATION
			if (Math.abs(s) <= Math.abs(half * q * r) || s <= q * a1 || s >= q * b1[0]) {
				e[0] = b[0] - a[0];
			} else {

				// A CUBIC INTERPOLATION STEP
				step[0] = s / q;

				// THE FUNCTION MUST NOT BE EVALUTATED TOO CLOSE TO A OR B
				if (step[0] - a[0] < twotol || b[0] - step[0] < twotol) {
					if (xmidpt > zero) {
						step[0] = tol[0];
					} else {
						step[0] = -tol[0];
					}
				}
			}
		}

		// IF THE STEP IS TOO LARGE, REPLACE BY THE SCALED BOUND (SO AS TO
		// COMPUTE THE NEW POINT ON THE BOUNDARY
		if (step[0] >= scxbnd[0]) {
			step[0] = scxbnd[0];

			// MOVE SXBD TO THE LEFT SO THAT SBND + TOL(XBND) = XBND
			scxbnd[0] -= (reltol[0] * Math.abs(xbnd) + abstol[0]) / (one + reltol[0]);
		}
		u[0] = step[0];
		if (Math.abs(step[0]) < tol[0] && step[0] < zero) {
			u[0] = -tol[0];
		}
		if (Math.abs(step[0]) < tol[0] && step[0] >= zero) {
			u[0] = tol[0];
		}
		itest[0] = 1;
	}

	private static void dcopy(final int n, final double[] dx, final int idx, final double[] dy, final int idy) {

		// USE NATIVE REPLACEMENT FOR COPYING ARRAYS
		System.arraycopy(dx, idx - 1, dy, idy - 1, n);
	}
}
