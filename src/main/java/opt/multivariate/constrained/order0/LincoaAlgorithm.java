/*
LINCOA---a LINearly Constrained Optimization Algorithm.
Copyright (C) 2013 M. J. D. Powell (University of Cambridge)

This package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License
 https://www.gnu.org/copyleft/lesser.html
for more details.

Michael J. D. Powell <mjdp@cam.ac.uk>
University of Cambridge
Cambridge, UK.
*/
package opt.multivariate.constrained.order0;

import java.util.Arrays;
import java.util.function.Function;

import opt.OptimizerSolution;
import utils.BlasMath;
import utils.RealMath;

/**
 * A translation of the algorithm LINCOA for minimizing a non-linear function
 * subject to linear inequality constraints. Originally written by M. J. D.
 * Powell.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Code found at: https://zhangzk.net/software.html
 * 
 * [2] Powell, Michael JD. "On fast trust region methods for quadratic models
 * with linear constraints." Mathematical Programming Computation 7.3 (2015):
 * 237-267.
 */
public final class LincoaAlgorithm {

	private final Function<Integer, Integer> mySize;
	private final double myTol, myRho0;
	private final int myMaxEvals;

	/**
	 *
	 * @param tolerance
	 * @param initialRadius
	 * @param maxEvals
	 * @param size
	 */
	public LincoaAlgorithm(final double tolerance, final double initialRadius, final int maxEvals,
			final Function<Integer, Integer> size) {
		myTol = tolerance;
		myRho0 = initialRadius;
		myMaxEvals = maxEvals;
		mySize = size;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialRadius
	 * @param maxEvals
	 */
	public LincoaAlgorithm(final double tolerance, final double initialRadius, final int maxEvals) {
		this(tolerance, initialRadius, maxEvals, d -> 2 * d + 1);
	}

	/**
	 *
	 * @param func
	 * @param a
	 * @param b
	 * @param guess
	 * @return
	 */
	public final OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> func,
			final double[][] a, final double[] b, final double[] guess) {

		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		final int m = b.length;
		final int[] fev = new int[1];

		// TODO: check convergence
		lincoa(func, n, mySize.apply(n), m, a, n, b, x, myRho0, myTol, 0, myMaxEvals, fev);
		return new OptimizerSolution<>(x, fev[0], 0, false);
	}

	private static void lincoa(final Function<? super double[], Double> func, final int n, final int npt, final int m,
			final double[][] a, final int ia, final double[] b, final double[] x, final double rhobeg,
			final double rhoend, final int iprint, final int maxfun, final int[] fev) {

		double zero, smallx, sum, temp;
		int i, j, np, nptm, iamat, ib, iflag, iw;

		// Check that N, NPT and MAXFUN are acceptable.
		zero = 0.0;
		smallx = 1.0e-6 * rhoend;
		np = n + 1;
		nptm = npt - np;
		if (n <= 1) {
			return;
		}
		if (npt < n + 2 || npt > (n + 2) * np / 2) {
			return;
		}
		if (maxfun <= npt) {
			return;
		}

		// Normalize the constraints, and copy the resultant constraint matrix
		// and right hand sides into working space, after increasing the right
		// hand sides if necessary so that the starting point is feasible.
		final double[] wb = new double[m];
		final double[][] wamat = new double[m][n];

		iamat = Math.max(m + 3 * n, Math.max(2 * m + n, 2 * npt)) + 1;
		ib = iamat + m * n;
		iflag = 0;
		if (m > 0) {
			iw = iamat - 1;
			for (j = 1; j <= m; ++j) {
				sum = temp = zero;
				for (i = 1; i <= n; ++i) {
					sum += a[j - 1][i - 1] * x[i - 1];
					temp += a[j - 1][i - 1] * a[j - 1][i - 1];
				}
				if (temp == zero) {
					return;
				}
				temp = Math.sqrt(temp);
				if (sum - b[j - 1] > smallx * temp) {
					iflag = 1;
				}
				wb[j - 1] = Math.max(b[j - 1], sum) / temp;
				for (i = 1; i <= n; ++i) {
					++iw;
					wamat[j - 1][i - 1] = a[j - 1][i - 1] / temp;
				}
			}
		}

		// The above settings provide a partition of W for subroutine LINCOB.
		final double[] xbase = new double[n], fval = new double[npt], xsav = new double[n], xopt = new double[n],
				gopt = new double[n], hq = new double[n * np / 2], pq = new double[npt], step = new double[n],
				sp = new double[npt + npt], xnew = new double[n], rescon = new double[m], rfac = new double[n * np / 2],
				pqw = new double[npt + n], w = new double[iamat];
		final double[][] xpt = new double[npt][n], bmat = new double[npt + n][n], zmat = new double[npt][nptm],
				qfac = new double[n][n];
		final int[] iact = new int[n];
		lincob(func, n, npt, m, wamat, wb, x, rhobeg, rhoend, iprint, maxfun, xbase, xpt, fval, xsav, xopt, gopt, hq,
				pq, bmat, zmat, npt + n, step, sp, xnew, iact, rescon, qfac, rfac, pqw, w, fev);
	}

	private static void lincob(final Function<? super double[], Double> func, final int n, final int npt, final int m,
			final double[][] amat, final double[] b, final double[] x, final double rhobeg, final double rhoend,
			final int iprint, final double maxfun, final double[] xbase, final double[][] xpt, final double[] fval,
			final double[] xsav, final double[] xopt, final double[] gopt, final double[] hq, final double[] pq,
			final double[][] bmat, final double[][] zmat, final int ndim, final double[] step, final double[] sp,
			final double[] xnew, final int[] iact, final double[] rescon, final double[][] qfac, final double[] rfac,
			final double[] pqw, final double[] w, final int[] fev) {

		double half, one, tenth, zero, fopt, delta, rho, fsave = 0.0, xoptsq, qoptsq, sum, temp, sumz, delsav = 0.0,
				del, xdiff, diff, vquad = 0.0, f = 0.0, vqalt, dffalt = 0.0, ratio = 0.0, ssq, distsq;
		int i, ip, ih, j, k, np, nh, nptm, nf, itest, nvala, nvalb, ksave = 0;
		final int[] kopt = new int[1], idz = new int[1], knew = new int[1], nact = new int[1], ifeas = new int[1];
		final double[] snorm = new double[1];

		// Set some constants.
		half = 0.5;
		one = 1.0;
		tenth = 0.1;
		zero = 0.0;
		np = n + 1;
		nh = (n * np) / 2;
		nptm = npt - np;

		// Set the elements of XBASE, XPT, FVAL, XSAV, XOPT, GOPT, HQ, PQ, BMAT,
		// ZMAT and SP for the first iteration. An important feature is that,
		// if the interpolation point XPT(K,.) is not feasible, where K is any
		// integer from [1,NPT], then a change is made to XPT(K,.) if necessary
		// so that the constraint violation is at least 0.2*RHOBEG. Also KOPT
		// is set so that XPT(KOPT,.) is the initial trust region centre.
		prelim(func, n, npt, m, amat, b, x, rhobeg, iprint, xbase, xpt, fval, xsav, xopt, gopt, kopt, hq, pq, bmat,
				zmat, idz, ndim, sp, rescon, step, pqw, w, fev);

		// Begin the iterative procedure.
		nf = npt;
		fopt = fval[kopt[0] - 1];
		rho = rhobeg;
		delta = rho;
		ifeas[0] = nact[0] = 0;
		itest = 3;
		knew[0] = nvala = nvalb = 0;
		int flag = 20;

		while (true) {

			if (flag == 20) {

				// Shift XBASE if XOPT may be too far from XBASE. First make the
				// changes
				// to BMAT that do not depend on ZMAT.
				fsave = fopt;
				xoptsq = BlasMath.ddotm(n, xopt, 1, xopt, 1);
				if (xoptsq >= 1.0e4 * delta * delta) {
					qoptsq = 0.25 * xoptsq;
					for (k = 1; k <= npt; k++) {
						sum = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
						sum -= half * xoptsq;
						w[npt + k - 1] = sum;
						sp[k - 1] = zero;
						for (i = 1; i <= n; i++) {
							xpt[k - 1][i - 1] -= half * xopt[i - 1];
							step[i - 1] = bmat[k - 1][i - 1];
							w[i - 1] = sum * xpt[k - 1][i - 1] + qoptsq * xopt[i - 1];
							ip = npt + i;
							for (j = 1; j <= i; j++) {
								bmat[ip - 1][j - 1] += step[i - 1] * w[j - 1] + w[i - 1] * step[j - 1];
							}
						}
					}

					// Then the revisions of BMAT that depend on ZMAT are calculated.
					for (k = 1; k <= nptm; k++) {
						sumz = zero;
						for (i = 1; i <= npt; i++) {
							sumz += zmat[i - 1][k - 1];
							w[i - 1] = w[npt + i - 1] * zmat[i - 1][k - 1];
						}
						for (j = 1; j <= n; j++) {
							sum = qoptsq * sumz * xopt[j - 1];
							for (i = 1; i <= npt; i++) {
								sum += w[i - 1] * xpt[i - 1][j - 1];
							}
							step[j - 1] = sum;
							if (k < idz[0]) {
								sum = -sum;
							}
							for (i = 1; i <= npt; i++) {
								bmat[i - 1][j - 1] += sum * zmat[i - 1][k - 1];
							}
						}
						for (i = 1; i <= n; i++) {
							ip = i + npt;
							temp = step[i - 1];
							if (k < idz[0]) {
								temp = -temp;
							}
							BlasMath.daxpym(i, temp, step, 1, bmat[ip - 1], 1);
						}
					}

					// Update the right hand sides of the constraints.
					if (m > 0) {
						for (j = 1; j <= m; j++) {
							temp = BlasMath.ddotm(n, amat[j - 1], 1, xopt, 1);
							b[j - 1] -= temp;
						}
					}

					// The following instructions complete the shift of XBASE,
					// including the
					// changes to the parameters of the quadratic model.
					ih = 0;
					for (j = 1; j <= n; j++) {
						w[j - 1] = zero;
						for (k = 1; k <= npt; k++) {
							w[j - 1] += pq[k - 1] * xpt[k - 1][j - 1];
							xpt[k - 1][j - 1] -= half * xopt[j - 1];
						}
						for (i = 1; i <= j; i++) {
							++ih;
							hq[ih - 1] += w[i - 1] * xopt[j - 1] + xopt[i - 1] * w[j - 1];
							bmat[npt + i - 1][j - 1] = bmat[npt + j - 1][i - 1];
						}
					}
					for (j = 1; j <= n; j++) {
						xbase[j - 1] += xopt[j - 1];
						xopt[j - 1] = zero;
						xpt[kopt[0] - 1][j - 1] = zero;
					}
				}

				// In the case KNEW=0, generate the next trust region step by calling
				// TRSTEP, where SNORM is the current trust region radius initially.
				// The final value of SNORM is the length of the calculated step,
				// except that SNORM is zero on return if the projected gradient is
				// unsuitable for starting the conjugate gradient iterations.
				delsav = delta;
				ksave = knew[0];
				if (knew[0] == 0) {
					snorm[0] = delta;
					System.arraycopy(gopt, 0, xnew, 0, n);
					trstep(n, npt, m, amat, b, xpt, hq, pq, nact, iact, rescon, qfac, rfac, snorm, step, xnew, w, w,
							m + 1, pqw, pqw, np, w, m + np);

					// A trust region step is applied whenever its length, namely
					// SNORM, is at
					// least HALF*DELTA. It is also applied if its length is at least
					// 0.1999
					// times DELTA and if a line search of TRSTEP has caused a change
					// to the
					// active set. Otherwise there is a branch below to label 530 or
					// 560.
					temp = half * delta;
					if (xnew[1 - 1] >= half) {
						temp = 0.1999 * delta;
					}
					if (snorm[0] <= temp) {
						delta *= half;
						if (delta <= 1.4 * rho) {
							delta = rho;
						}
						++nvala;
						++nvalb;
						temp = snorm[0] / rho;
						if (delsav > rho) {
							temp = one;
						}
						if (temp >= half) {
							nvala = 0;
						}
						if (temp >= tenth) {
							nvalb = 0;
						}
						if (delsav > rho || (nvala < 5 && nvalb < 3)) {
							flag = 530;
						} else if (snorm[0] > zero) {
							ksave = -1;
							nvala = nvalb = 0;
						} else {
							flag = 560;
						}
					} else {
						nvala = nvalb = 0;
					}
				} else {

					// Alternatively, KNEW is positive. Then the model step is
					// calculated
					// within a trust region of radius DEL, after setting the
					// gradient at
					// XBASE and the second derivative parameters of the KNEW-th
					// Lagrange
					// function in W(1) to W(N) and in PQW(1) to PQW(NPT),
					// respectively.
					del = Math.max(tenth * delta, rho);
					System.arraycopy(bmat[knew[0] - 1], 0, w, 0, n);
					Arrays.fill(pqw, 0, npt, zero);
					for (j = 1; j <= nptm; j++) {
						temp = zmat[knew[0] - 1][j - 1];
						if (j < idz[0]) {
							temp = -temp;
						}
						for (k = 1; k <= npt; k++) {
							pqw[k - 1] += temp * zmat[k - 1][j - 1];
						}
					}
					qmstep(n, npt, m, amat, b, xpt, xopt, nact[0], iact, rescon, qfac, kopt[0], knew[0], del, step, w,
							pqw, w, np, w, np + m, ifeas);
				}

				if (flag == 20) {

					// Set VQUAD to the change to the quadratic model when the move
					// STEP is
					// made from XOPT. If STEP is a trust region step, then VQUAD
					// should be
					// negative. If it is nonnegative due to rounding errors in this
					// case,
					// there is a branch to label 530 to try to improve the model.
					vquad = zero;
					ih = 0;
					for (j = 1; j <= n; j++) {
						vquad += step[j - 1] * gopt[j - 1];
						for (i = 1; i <= j; i++) {
							++ih;
							temp = step[i - 1] * step[j - 1];
							if (i == j) {
								temp *= half;
							}
							vquad += temp * hq[ih - 1];
						}
					}
					for (k = 1; k <= npt; k++) {
						temp = zero;
						for (j = 1; j <= n; j++) {
							temp += xpt[k - 1][j - 1] * step[j - 1];
							sp[npt + k - 1] = temp;
						}
						vquad += half * pq[k - 1] * temp * temp;
					}
					if ((ksave == 0) && (vquad >= zero)) {
						flag = 530;
					} else {
						flag = 220;
					}
				}
			}

			if (flag == 220) {

				// Calculate the next value of the objective function. The difference
				// between the actual new value of F and the value predicted by the
				// model is recorded in DIFF.
				++nf;
				if (nf > maxfun) {
					--nf;
					break;
				}
				xdiff = zero;
				for (i = 1; i <= n; i++) {
					xnew[i - 1] = xopt[i - 1] + step[i - 1];
					x[i - 1] = xbase[i - 1] + xnew[i - 1];
					xdiff += (x[i - 1] - xsav[i - 1]) * (x[i - 1] - xsav[i - 1]);
				}
				xdiff = Math.sqrt(xdiff);
				if (ksave == -1) {
					xdiff = rho;
				}
				if ((xdiff <= tenth * rho) || (xdiff >= delta + delta)) {
					ifeas[0] = 0;
					break;
				}
				if (ksave <= 0) {
					ifeas[0] = 1;
				}
				// f = (double) ifeas[0];
				f = func.apply(x);
				if (ksave == -1) {
					break;
				}
				diff = f - fopt - vquad;

				// If X is feasible, then set DFFALT to the difference between the
				// new
				// value of F and the value predicted by the alternative model.
				if ((ifeas[0] == 1) && (itest < 3)) {
					for (k = 1; k <= npt; k++) {
						pqw[k - 1] = zero;
						w[k - 1] = fval[k - 1] - fval[kopt[0] - 1];
					}
					for (j = 1; j <= nptm; j++) {
						sum = zero;
						for (i = 1; i <= npt; i++) {
							sum += w[i - 1] * zmat[i - 1][j - 1];
						}
						if (j < idz[0]) {
							sum = -sum;
						}
						for (k = 1; k <= npt; k++) {
							pqw[k - 1] += sum * zmat[k - 1][j - 1];
						}
					}
					vqalt = zero;
					for (k = 1; k <= npt; k++) {
						sum = BlasMath.ddotm(n, bmat[k - 1], 1, step, 1);
						vqalt += sum * w[k - 1];
						vqalt += pqw[k - 1] * sp[npt + k - 1] * (half * sp[npt + k - 1] + sp[k - 1]);
					}
					dffalt = f - fopt - vqalt;
				}
				if (itest == 3) {
					dffalt = diff;
					itest = 0;
				}

				// Pick the next value of DELTA after a trust region step.
				if (ksave == 0) {
					ratio = (f - fopt) / vquad;
					if (ratio <= tenth) {
						delta *= half;
					} else if (ratio <= 0.7) {
						delta = Math.max(half * delta, snorm[0]);
					} else {
						temp = Math.sqrt(2.0) * delta;
						delta = Math.max(half * delta, snorm[0] + snorm[0]);
						delta = Math.min(delta, temp);
					}
					if (delta <= 1.4 * rho) {
						delta = rho;
					}
				}

				// Update BMAT, ZMAT and IDZ, so that the KNEW-th interpolation point
				// can be moved. If STEP is a trust region step, then KNEW is zero at
				// present, but a positive value is picked by subroutine UPDATE.
				update(n, npt, xpt, bmat, zmat, idz, ndim, sp, step, kopt[0], knew, pqw, w);
				if (knew[0] == 0) {
					break;
				}

				// If ITEST is increased to 3, then the next quadratic model is the
				// one whose second derivative matrix is least subject to the new
				// interpolation conditions. Otherwise the new model is constructed
				// by the symmetric Broyden method in the usual way.
				if (ifeas[0] == 1) {
					++itest;
					if (Math.abs(dffalt) >= tenth * Math.abs(diff)) {
						itest = 0;
					}
				}

				// Update the second derivatives of the model by the symmetric
				// Broyden
				// method, using PQW for the second derivative parameters of the new
				// KNEW-th Lagrange function. The contribution from the old parameter
				// PQ(KNEW) is included in the second derivative matrix HQ. W is used
				// later for the gradient of the new KNEW-th Lagrange function.
				if (itest < 3) {
					Arrays.fill(pqw, 0, npt, zero);
					for (j = 1; j <= nptm; j++) {
						temp = zmat[knew[0] - 1][j - 1];
						if (temp != zero) {
							if (j < idz[0]) {
								temp = -temp;
							}
							for (k = 1; k <= npt; k++) {
								pqw[k - 1] += temp * zmat[k - 1][j - 1];
							}
						}
					}
					ih = 0;
					for (i = 1; i <= n; i++) {
						w[i - 1] = bmat[knew[0] - 1][i - 1];
						temp = pq[knew[0] - 1] * xpt[knew[0] - 1][i - 1];
						for (j = 1; j <= i; j++) {
							++ih;
							hq[ih - 1] += temp * xpt[knew[0] - 1][j - 1];
						}
					}
					pq[knew[0] - 1] = zero;
					BlasMath.daxpym(npt, diff, pqw, 1, pq, 1);
				}

				// Include the new interpolation point with the corresponding updates
				// of
				// SP. Also make the changes of the symmetric Broyden method to GOPT
				// at
				// the old XOPT if ITEST is less than 3.
				fval[knew[0] - 1] = f;
				sp[knew[0] - 1] += sp[npt + kopt[0] - 1];
				System.arraycopy(xnew, 0, xpt[knew[0] - 1], 0, n);
				ssq = BlasMath.ddotm(n, step, 1, step, 1);
				sp[npt + knew[0] - 1] = sp[npt + kopt[0] - 1] + ssq;
				if (itest < 3) {
					for (k = 1; k <= npt; k++) {
						temp = pqw[k - 1] * sp[k - 1];
						BlasMath.daxpym(n, temp, xpt[k - 1], 1, w, 1);
					}
					BlasMath.daxpym(n, diff, w, 1, gopt, 1);
				}

				// Update FOPT, XSAV, XOPT, KOPT, RESCON and SP if the new F is the
				// least calculated value so far with a feasible vector of variables.
				if ((f < fopt) && (ifeas[0] == 1)) {
					fopt = f;
					System.arraycopy(x, 0, xsav, 0, n);
					System.arraycopy(xnew, 0, xopt, 0, n);
					kopt[0] = knew[0];
					snorm[0] = Math.sqrt(ssq);
					for (j = 1; j <= m; j++) {
						if (rescon[j - 1] >= delta + snorm[0]) {
							rescon[j - 1] = snorm[0] - rescon[j - 1];
						} else {
							rescon[j - 1] += snorm[0];
							if (rescon[j - 1] + delta > zero) {
								temp = b[j - 1];
								temp -= BlasMath.ddotm(n, xopt, 1, amat[j - 1], 1);
								temp = Math.max(temp, zero);
								if (temp >= delta) {
									temp = -temp;
								}
								rescon[j - 1] = temp;
							}
						}
					}
					BlasMath.daxpym(npt, 1.0, sp, npt, sp, 1);
					if (itest < 3) {
						ih = 0;
						for (j = 1; j <= n; j++) {
							for (i = 1; i <= j; i++) {
								++ih;
								if (i < j) {
									gopt[j - 1] += hq[ih - 1] * step[i - 1];
								}
								gopt[i - 1] += hq[ih - 1] * step[j - 1];
							}
						}
						for (k = 1; k <= npt; k++) {
							temp = pq[k - 1] * sp[npt + k - 1];
							BlasMath.daxpym(n, temp, xpt[k - 1], 1, gopt, 1);
						}
					}
				}

				// Replace the current model by the least Frobenius norm interpolant
				// if
				// this interpolant gives substantial reductions in the predictions
				// of values of F at feasible points.
				if (itest == 3) {
					for (k = 1; k <= npt; k++) {
						pq[k - 1] = zero;
						w[k - 1] = fval[k - 1] - fval[kopt[0] - 1];
					}
					for (j = 1; j <= nptm; j++) {
						sum = zero;
						for (i = 1; i <= npt; i++) {
							sum += w[i - 1] * zmat[i - 1][j - 1];
						}
						if (j < idz[0]) {
							sum = -sum;
						}
						for (k = 1; k <= npt; k++) {
							pq[k - 1] += sum * zmat[k - 1][j - 1];
						}
					}
					for (j = 1; j <= n; j++) {
						gopt[j - 1] = zero;
						for (i = 1; i <= npt; i++) {
							gopt[j - 1] += w[i - 1] * bmat[i - 1][j - 1];
						}
					}
					for (k = 1; k <= npt; k++) {
						temp = pq[k - 1] * sp[k - 1];
						BlasMath.daxpym(n, temp, xpt[k - 1], 1, gopt, 1);
					}
					Arrays.fill(hq, 0, nh, zero);
				}

				// If a trust region step has provided a sufficient decrease in F,
				// then
				// branch for another trust region calculation. Every iteration that
				// takes a model step is followed by an attempt to take a trust
				// region
				// step.
				knew[0] = 0;
				if (ksave > 0 || ratio >= tenth) {
					flag = 20;
					continue;
				} else {
					flag = 530;
				}
			}

			if (flag == 530) {

				// Alternatively, find out if the interpolation points are close
				// enough
				// to the best point so far.
				distsq = Math.max(delta * delta, 4.0 * rho * rho);
				for (k = 1; k <= npt; k++) {
					sum = zero;
					for (j = 1; j <= n; j++) {
						sum += (xpt[k - 1][j - 1] - xopt[j - 1]) * (xpt[k - 1][j - 1] - xopt[j - 1]);
					}
					if (sum > distsq) {
						knew[0] = k;
						distsq = sum;
					}
				}

				// If KNEW is positive, then branch back for the next iteration,
				// which
				// will generate a "model step". Otherwise, if the current iteration
				// has reduced F, or if DELTA was above its lower bound when the last
				// trust region step was calculated, then try a "trust region" step
				// instead.
				if (knew[0] > 0) {
					flag = 20;
					continue;
				}
				knew[0] = 0;
				if (fopt < fsave || delsav > rho) {
					flag = 20;
					continue;
				} else {
					flag = 560;
				}
			}

			if (flag == 560) {

				// The calculations with the current value of RHO are complete.
				// Pick the next value of RHO.
				if (rho > rhoend) {
					delta = half * rho;
					if (rho > 250.0 * rhoend) {
						rho *= tenth;
					} else if (rho <= 16.0 * rhoend) {
						rho = rhoend;
					} else {
						rho = Math.sqrt(rho * rhoend);
					}
					delta = Math.max(delta, rho);
					knew[0] = nvala = nvalb = 0;
					flag = 20;
				} else if (ksave == -1) {

					// Return from the calculation, after branching to label 220 for
					// another
					// Newton-Raphson step if it has not been tried before.
					flag = 220;
				} else {
					break;
				}
			}
		}

		if (fopt <= f || ifeas[0] == 0) {
			System.arraycopy(xsav, 0, x, 0, n);
			f = fopt;
		}
		w[1 - 1] = f;
		w[2 - 1] = (double) nf + half;
	}

	private static void trstep(final int n, final int npt, final int m, final double[][] amat, final double[] b,
			final double[][] xpt, final double[] hq, final double[] pq, final int[] nact, final int[] iact,
			final double[] rescon, final double[][] qfac, final double[] rfac, final double[] snorm,
			final double[] step, final double[] g, final double[] resnew, final double[] resact, final int iresact,
			final double[] d, final double[] dw, final int idw, final double[] w, final int iw) {

		double half, zero, tiny, ctest, snsq, one, ss, scale, reduct, resmax, gamma = 0.0, temp, sum, rhs, ds, dd, ad,
				adw, alpbd = 0.0, alpha, dg, alpht, dgd, alphm, beta, wgd;
		int i, ih, icount = 0, ir, j, jsav, k, ncall;

		// Set some numbers for the conjugate gradient iterations.
		half = 0.5;
		one = 1.0;
		tiny = 1e-60;
		zero = 0.0;
		ctest = 0.01;
		snsq = snorm[0] * snorm[0];

		// Set the initial elements of RESNEW, RESACT and STEP.
		if (m > 0) {
			System.arraycopy(rescon, 0, resnew, 0, m);
			for (j = 1; j <= m; j++) {
				if (rescon[j - 1] >= snorm[0]) {
					resnew[j - 1] = -one;
				} else if (rescon[j - 1] >= zero) {
					resnew[j - 1] = Math.max(resnew[j - 1], tiny);
				}

			}
			if (nact[0] > 0) {
				for (k = 1; k <= nact[0]; k++) {
					resact[k - 1 + iresact - 1] = rescon[iact[k - 1] - 1];
					resnew[iact[k - 1] - 1] = zero;
				}
			}
		}
		Arrays.fill(step, 0, n, zero);
		ss = reduct = zero;
		ncall = 0;

		boolean goto40 = true;
		while (true) {

			if (goto40) {

				// GETACT picks the active set for the current STEP. It also sets DW
				// to
				// the vector closest to -G that is orthogonal to the normals of the
				// active constraints. DW is scaled to have length 0.2*SNORM, as then
				// a move of DW from STEP is allowed by the linear constraints.
				++ncall;
				getact(n, m, amat, b, nact, iact, qfac, rfac, snorm[0], resnew, 1, resact, iresact, g, dw, idw, w,
						1 + iw - 1, w, n + 1 + iw - 1);
				if (w[n + 1 - 1 + iw - 1] == zero) {
					break;
				}
				scale = 0.2 * snorm[0] / Math.sqrt(w[n + 1 - 1 + iw - 1]);
				BlasMath.dscalm(n, scale, dw, idw);

				// If the modulus of the residual of an active constraint is
				// substantial,
				// then set D to the shortest move from STEP to the boundaries of the
				// active constraints.
				resmax = zero;
				if (nact[0] > 0) {
					for (k = 1; k <= nact[0]; ++k) {
						resmax = Math.max(resmax, resact[k - 1 + iresact - 1]);
					}
				}
				gamma = zero;
				if (resmax > 1.0e-4 * snorm[0]) {

					ir = 0;
					for (k = 1; k <= nact[0]; k++) {
						temp = resact[k - 1 + iresact - 1];
						if (k >= 2) {
							for (i = 1; i <= k - 1; ++i) {
								++ir;
								temp -= rfac[ir - 1] * w[i - 1 + iw - 1];
							}
						}
						++ir;
						w[k - 1 + iw - 1] = temp / rfac[ir - 1];
					}
					for (i = 1; i <= n; i++) {
						d[i - 1] = BlasMath.ddotm(nact[0], w, iw, qfac[i - 1], 1);
					}

					// The vector D that has just been calculated is also the
					// shortest move
					// from STEP+DW to the boundaries of the active constraints. Set
					// GAMMA
					// to the greatest steplength of this move that satisfies the
					// trust
					// region bound.
					rhs = snsq;
					ds = dd = zero;
					for (i = 1; i <= n; i++) {
						sum = step[i - 1] + dw[i - 1 + idw - 1];
						rhs -= sum * sum;
						ds += d[i - 1] * sum;
						dd += d[i - 1] * d[i - 1];
					}
					if (rhs > zero) {
						temp = Math.sqrt(ds * ds + dd * rhs);
						if (ds <= zero) {
							gamma = (temp - ds) / dd;
						} else {
							gamma = rhs / (temp + ds);
						}
					}

					// Reduce the steplength GAMMA if necessary so that the move
					// along D
					// also satisfies the linear constraints.
					j = 0;
					while (true) {
						if (gamma > zero) {
							++j;
							if (resnew[j - 1] > zero) {
								ad = adw = zero;
								for (i = 1; i <= n; i++) {
									ad += amat[j - 1][i - 1] * d[i - 1];
									adw += amat[j - 1][i - 1] * dw[i - 1 + idw - 1];
								}
								if (ad > zero) {
									temp = Math.max((resnew[j - 1] - adw) / ad, zero);
									gamma = Math.min(gamma, temp);
								}
							}
							if (j >= m) {
								break;
							}
						}
					}
					gamma = Math.min(gamma, one);
				}

				// Set the next direction for seeking a reduction in the model
				// function
				// subject to the trust region bound and the linear constraints.
				if (gamma <= zero) {
					System.arraycopy(dw, idw - 1, d, 0, n);
					icount = nact[0];
				} else {
					BlasMath.daxpy1(n, gamma, d, 1, dw, idw, d, 1);
					icount = nact[0] - 1;
				}
				alpbd = one;
			}

			// Set ALPHA to the steplength from STEP along D to the trust region
			// boundary. Return if the first derivative term of this step is
			// sufficiently small or if no further progress is possible.
			++icount;
			rhs = snsq - ss;
			if (rhs <= zero) {
				break;
			}
			dg = ds = dd = zero;
			for (i = 1; i <= n; i++) {
				dg += d[i - 1] * g[i - 1];
				ds += d[i - 1] * step[i - 1];
				dd += d[i - 1] * d[i - 1];
			}
			if (dg >= zero) {
				break;
			}
			temp = Math.sqrt(rhs * dd + ds * ds);
			if (ds <= zero) {
				alpha = (temp - ds) / dd;
			} else {
				alpha = rhs / (temp + ds);
			}
			if (-alpha * dg <= ctest * reduct) {
				break;
			}

			// Set DW to the change in gradient along D.
			ih = 0;
			for (j = 1; j <= n; j++) {
				dw[j - 1 + idw - 1] = zero;
				for (i = 1; i <= j; i++) {
					++ih;
					if (i < j) {
						dw[j - 1 + idw - 1] += hq[ih - 1] * d[i - 1];
					}
					dw[i - 1 + idw - 1] += hq[ih - 1] * d[j - 1];
				}
			}
			for (k = 1; k <= npt; k++) {
				temp = BlasMath.ddotm(n, xpt[k - 1], 1, d, 1);
				temp *= pq[k - 1];
				BlasMath.daxpym(n, temp, xpt[k - 1], 1, dw, idw);
			}

			// Set DGD to the curvature of the model along D. Then reduce ALPHA if
			// necessary to the value that minimizes the model.
			dgd = BlasMath.ddotm(n, d, 1, dw, idw);
			alpht = alpha;
			if (dg + alpha * dgd > zero) {
				alpha = -dg / dgd;
			}

			// Make a further reduction in ALPHA if necessary to preserve
			// feasibility,
			// and put some scalar products of D with constraint gradients in W.
			alphm = alpha;
			jsav = 0;
			if (m > 0) {
				for (j = 1; j <= m; j++) {
					ad = zero;
					if (resnew[j - 1] > zero) {
						ad += BlasMath.ddotm(n, amat[j - 1], 1, d, 1);
						if (alpha * ad > resnew[j - 1]) {
							alpha = resnew[j - 1] / ad;
							jsav = j;
						}
					}
					w[j - 1 + iw - 1] = ad;
				}
			}
			alpha = Math.max(alpha, alpbd);
			alpha = Math.min(alpha, alphm);
			if (icount == nact[0]) {
				alpha = Math.min(alpha, one);
			}

			// Update STEP, G, RESNEW, RESACT and REDUCT.
			ss = zero;
			for (i = 1; i <= n; i++) {
				step[i - 1] += alpha * d[i - 1];
				ss += step[i - 1] * step[i - 1];
				g[i - 1] += alpha * dw[i - 1 + idw - 1];
			}
			if (m > 0) {
				for (j = 1; j <= m; j++) {
					if (resnew[j - 1] > zero) {
						resnew[j - 1] = Math.max(resnew[j - 1] - alpha * w[j - 1 + iw - 1], tiny);
					}
				}
			}
			if ((icount == nact[0]) && (nact[0] > 0)) {
				BlasMath.dscalm(nact[0], one - gamma, resact, iresact);
			}
			reduct -= alpha * (dg + half * alpha * dgd);

			// Test for termination. Branch to label 40 if there is a new active
			// constraint and if the distance from STEP to the trust region
			// boundary is at least 0.2*SNORM.
			if (alpha == alpht) {
				break;
			}
			temp = -alphm * (dg + half * alphm * dgd);
			if (temp <= ctest * reduct) {
				break;
			} else if (jsav > 0) {
				if (ss <= 0.64 * snsq) {
					goto40 = true;
				} else {
					break;
				}
			} else if (icount == n) {
				break;
			} else {

				// Calculate the next search direction, which is conjugate to the
				// previous one except in the case ICOUNT=NACT.
				if (nact[0] > 0) {
					for (j = nact[0] + 1; j <= n; ++j) {
						w[j - 1 + iw - 1] = zero;
						for (i = 1; i <= n; i++) {
							w[j - 1 + iw - 1] += g[i - 1] * qfac[i - 1][j - 1];
						}
					}
					for (i = 1; i <= n; i++) {
						temp = zero;
						for (j = nact[0] + 1; j <= n; ++j) {
							temp += qfac[i - 1][j - 1] * w[j - 1 + iw - 1];
						}
						w[n + i - 1 + iw - 1] = temp;
					}
				} else {
					System.arraycopy(g, 0, w, n + iw - 1, n);
				}
				if (icount == nact[0]) {
					beta = zero;
				} else {
					wgd = BlasMath.ddotm(n, w, n + iw, dw, idw);
					beta = wgd / dgd;
				}
				for (i = 1; i <= n; i++) {
					d[i - 1] = -w[n + i - 1 + iw - 1] + beta * d[i - 1];
				}
				alpbd = zero;
				goto40 = false;
			}
		}

		// Return from the subroutine.
		snorm[0] = zero;
		if (reduct > zero) {
			snorm[0] = Math.sqrt(ss);
		}
		g[1 - 1] = zero;
		if (ncall > 1) {
			g[1 - 1] = one;
		}
	}

	private static void getact(final int n, final int m, final double[][] amat, final double[] b, final int[] nact,
			final int[] iact, final double[][] qfac, final double[] rfac, final double snorm, final double[] resnew,
			final int iresnew, final double[] resact, final int iresact, final double[] g, final double[] dw,
			final int idw, final double[] vlam, final int ivlam, final double[] w, final int iw) {

		double one, tiny, zero, tdel, ddsav, temp, dd, dnorm, test, violmx = 0.0, sum, ctol = 0.0, rdiag, sprod, sinv,
				cosv, vmult, cval, sval;
		int i, j, jw, jc, jcp, jdiag, idiag, iflag = 0, ic = 0, k, l, nactp;

		// Set some constants and a temporary VLAM.
		one = 1.0;
		tiny = 1.0e-60;
		zero = 0.0;
		tdel = 0.2 * snorm;
		ddsav = zero;
		for (i = 1; i <= n; ++i) {
			ddsav += g[i - 1] * g[i - 1];
			vlam[i - 1 + ivlam - 1] = zero;
		}
		ddsav = ddsav + ddsav;

		// Set the initial QFAC to the identity matrix in the case NACT=0.
		int gotoflag = 40;
		if (nact[0] == 0) {
			for (i = 1; i <= n; ++i) {
				Arrays.fill(qfac[i - 1], 0, n, zero);
				qfac[i - 1][i - 1] = one;
			}
			gotoflag = 100;
		} else {

			// Remove any constraints from the initial active set whose residuals
			// exceed TDEL.
			iflag = 1;
			ic = nact[0];
		}

		while (true) {

			if (gotoflag == 40) {
				while (true) {
					if (resact[ic - 1 + iresact - 1] > tdel) {
						gotoflag = 800;
						break;
					}
					--ic;
					if (ic <= 0) {

						// Remove any constraints from the initial active set whose
						// Lagrange
						// multipliers are nonnegative, and set the surviving
						// multipliers.
						iflag = 2;
						gotoflag = 60;
						break;
					}
				}
			}

			if (gotoflag == 60) {

				if (nact[0] != 0) {
					ic = nact[0];
					while (true) {
						temp = zero;
						for (i = 1; i <= n; i++) {
							temp += qfac[i - 1][ic - 1] * g[i - 1];
						}
						idiag = (ic * ic + ic) / 2;
						if (ic < nact[0]) {
							jw = idiag + ic;
							for (j = ic + 1; j <= nact[0]; ++j) {
								temp -= rfac[jw - 1] * vlam[j - 1 + ivlam - 1];
								jw += j;
							}
						}
						if (temp >= zero) {
							gotoflag = 800;
							break;
						}
						vlam[ic - 1 + ivlam - 1] = temp / rfac[idiag - 1];
						--ic;
						if (ic <= 0) {
							gotoflag = 100;
							break;
						}
					}
				} else {
					gotoflag = 100;
				}
			}

			if (gotoflag == 100) {

				// Set the new search direction D. Terminate if the 2-norm of D is
				// zero
				// or does not decrease, or if NACT=N holds. The situation NACT=N
				// occurs for sufficiently large SNORM if the origin is in the convex
				// hull of the constraint gradients.
				if (nact[0] == n) {
					dd = zero;
					w[1 - 1 + iw - 1] = dd;
					return;
				}
				for (j = nact[0] + 1; j <= n; ++j) {
					w[j - 1 + iw - 1] = zero;
					for (i = 1; i <= n; i++) {
						w[j - 1 + iw - 1] += qfac[i - 1][j - 1] * g[i - 1];
					}
				}
				dd = zero;
				for (i = 1; i <= n; i++) {
					dw[i - 1 + idw - 1] = zero;
					for (j = nact[0] + 1; j <= n; ++j) {
						dw[i - 1 + idw - 1] -= w[j - 1 + iw - 1] * qfac[i - 1][j - 1];
					}
					dd += dw[i - 1 + idw - 1] * dw[i - 1 + idw - 1];
				}
				if (dd >= ddsav) {
					dd = zero;
					w[1 - 1 + iw - 1] = dd;
					return;
				}
				if (dd == zero) {
					w[1 - 1 + iw - 1] = dd;
					return;
				}
				ddsav = dd;
				dnorm = Math.sqrt(dd);

				// Pick the next integer L or terminate, a positive value of L being
				// the index of the most violated constraint. The purpose of CTOL
				// below is to estimate whether a positive value of VIOLMX may be
				// due to computer rounding errors.
				l = 0;
				if (m > 0) {
					test = dnorm / snorm;
					violmx = zero;
					for (j = 1; j <= m; j++) {
						if ((resnew[j - 1 + iresnew - 1] > zero) && (resnew[j - 1 + iresnew - 1] <= tdel)) {
							sum = BlasMath.ddotm(n, amat[j - 1], 1, dw, idw);
							if (sum > test * resnew[j - 1 + iresnew - 1]) {
								if (sum > violmx) {
									l = j;
									violmx = sum;
								}
							}
						}
					}
					ctol = zero;
					temp = 0.01 * dnorm;
					if ((violmx > zero) && (violmx < temp)) {
						if (nact[0] > 0) {
							for (k = 1; k <= nact[0]; k++) {
								j = iact[k - 1];
								sum = BlasMath.ddotm(n, dw, idw, amat[j - 1], 1);
								ctol = Math.max(ctol, Math.abs(sum));
							}
						}
					}
				}
				w[1 - 1 + iw - 1] = one;
				if (l == 0 || violmx <= 10.0 * ctol) {
					w[1 - 1 + iw - 1] = dd;
					return;
				}

				// Apply Givens rotations to the last (N-NACT) columns of QFAC so
				// that
				// the first (NACT+1) columns of QFAC are the ones required for the
				// addition of the L-th constraint, and add the appropriate column
				// to RFAC.
				nactp = nact[0] + 1;
				idiag = (nactp * nactp - nactp) / 2;
				rdiag = zero;
				for (j = n; j >= 1; --j) {
					sprod = zero;
					for (i = 1; i <= n; i++) {
						sprod += qfac[i - 1][j - 1] * amat[l - 1][i - 1];
					}
					if (j <= nact[0]) {
						rfac[idiag + j - 1] = sprod;
					} else if (Math.abs(rdiag) <= 1.0e-20 * Math.abs(sprod)) {
						rdiag = sprod;
					} else {
						temp = Math.sqrt(sprod * sprod + rdiag * rdiag);
						cosv = sprod / temp;
						sinv = rdiag / temp;
						rdiag = temp;
						for (i = 1; i <= n; i++) {
							temp = cosv * qfac[i - 1][j - 1] + sinv * qfac[i - 1][j + 1 - 1];
							qfac[i - 1][j + 1 - 1] = -sinv * qfac[i - 1][j - 1] + cosv * qfac[i - 1][j + 1 - 1];
							qfac[i - 1][j - 1] = temp;
						}
					}
				}

				if (rdiag < zero) {
					for (i = 1; i <= n; i++) {
						qfac[i - 1][nactp - 1] = -qfac[i - 1][nactp - 1];
					}
				}
				rfac[idiag + nactp - 1] = Math.abs(rdiag);
				nact[0] = nactp;
				iact[nact[0] - 1] = l;
				resact[nact[0] - 1 + iresact - 1] = resnew[l - 1 + iresnew - 1];
				vlam[nact[0] - 1 + ivlam - 1] = zero;
				resnew[l - 1 + iresnew - 1] = zero;
				gotoflag = 220;
			}

			if (gotoflag == 220) {

				// Set the components of the vector VMU in W.
				w[nact[0] - 1 + iw - 1] = one / RealMath.pow(rfac[(nact[0] * nact[0] + nact[0]) / 2 - 1], 2);
				if (nact[0] > 1) {
					for (i = nact[0] - 1; i >= 1; --i) {
						idiag = (i * i + i) / 2;
						jw = idiag + i;
						sum = zero;
						for (j = i + 1; j <= nact[0]; ++j) {
							sum -= rfac[jw - 1] * w[j - 1 + iw - 1];
							jw += j;
						}
						w[i - 1 + iw - 1] = sum / rfac[idiag - 1];
					}
				}

				// Calculate the multiple of VMU to subtract from VLAM, and update
				// VLAM.
				vmult = violmx;
				ic = 0;
				j = 1;
				while (j < nact[0]) {
					if (vlam[j - 1 + ivlam - 1] >= vmult * w[j - 1 + iw - 1]) {
						ic = j;
						vmult = vlam[j - 1 + ivlam - 1] / w[j - 1 + iw - 1];
					}
					++j;
				}
				BlasMath.daxpym(nact[0], -vmult, w, iw, vlam, ivlam);
				if (ic > 0) {
					vlam[ic - 1 + ivlam - 1] = zero;
				}
				violmx = Math.max(violmx - vmult, zero);
				if (ic == 0) {
					violmx = zero;
				}

				// Reduce the active set if necessary, so that all components of the
				// new VLAM are negative, with resetting of the residuals of the
				// constraints that become inactive.
				iflag = 3;
				ic = nact[0];
				gotoflag = 270;
			}

			if (gotoflag == 270) {

				while (true) {
					if (vlam[ic - 1 + ivlam - 1] >= zero) {
						resnew[iact[ic - 1] - 1 + iresnew - 1] = Math.max(resact[ic - 1 + iresact - 1], tiny);
						gotoflag = 800;
						break;
					}
					--ic;
					if (ic <= 0) {

						// Calculate the next VMU if VIOLMX is positive. Return if
						// NACT=N holds,
						// as then the active constraints imply D=0. Otherwise, go to
						// label
						// 100, to calculate the new D and to test for termination.
						if (violmx > zero) {
							gotoflag = 220;
							break;
						} else if (nact[0] < n) {
							gotoflag = 100;
							break;
						} else {
							dd = zero;
							w[1 - 1 + iw - 1] = dd;
							return;
						}
					}
				}
			}

			if (gotoflag == 800) {

				// These instructions rearrange the active constraints so that the
				// new
				// value of IACT(NACT) is the old value of IACT(IC). A sequence of
				// Givens rotations is applied to the current QFAC and RFAC. Then
				// NACT
				// is reduced by one.
				resnew[iact[ic - 1] - 1 + iresnew - 1] = Math.max(resact[ic - 1 + iresact - 1], tiny);
				jc = ic;

				while (jc < nact[0]) {
					jcp = jc + 1;
					idiag = jc * jcp / 2;
					jw = idiag + jcp;
					temp = RealMath.hypot(rfac[jw - 1 - 1], rfac[jw - 1]);
					cval = rfac[jw - 1] / temp;
					sval = rfac[jw - 1 - 1] / temp;
					rfac[jw - 1 - 1] = sval * rfac[idiag - 1];
					rfac[jw - 1] = cval * rfac[idiag - 1];
					rfac[idiag - 1] = temp;
					if (jcp < nact[0]) {
						for (j = jcp + 1; j <= nact[0]; ++j) {
							temp = sval * rfac[jw + jc - 1] + cval * rfac[jw + jcp - 1];
							rfac[jw + jcp - 1] = cval * rfac[jw + jc - 1] - sval * rfac[jw + jcp - 1];
							rfac[jw + jc - 1] = temp;
							jw += j;
						}
					}
					jdiag = idiag - jc;
					for (i = 1; i <= n; i++) {
						if (i < jc) {
							temp = rfac[idiag + i - 1];
							rfac[idiag + i - 1] = rfac[jdiag + i - 1];
							rfac[jdiag + i - 1] = temp;
						}
						temp = sval * qfac[i - 1][jc - 1] + cval * qfac[i - 1][jcp - 1];
						qfac[i - 1][jcp - 1] = cval * qfac[i - 1][jc - 1] - sval * qfac[i - 1][jcp - 1];
						qfac[i - 1][jc - 1] = temp;
					}
					iact[jc - 1] = iact[jcp - 1];
					resact[jc - 1 + iresact - 1] = resact[jcp - 1 + iresact - 1];
					vlam[jc - 1 + ivlam - 1] = vlam[jcp - 1 + ivlam - 1];
					jc = jcp;
				}

				--nact[0];
				if (iflag < 0) {
					--ic;
					if (ic > 0) {
						gotoflag = 40;
					} else {
						iflag = 2;
						gotoflag = 60;
					}
				} else if (iflag == 0) {
					gotoflag = 60;
				} else {
					--ic;
					if (ic > 0) {
						gotoflag = 270;
					} else if (violmx > zero) {
						gotoflag = 220;
					} else if (nact[0] < n) {
						gotoflag = 100;
					} else {
						dd = zero;
						w[1 - 1 + iw - 1] = dd;
						return;
					}
				}
			}
		}
	}

	private static void qmstep(final int n, final int npt, final int m, final double[][] amat, final double[] b,
			final double[][] xpt, final double[] xopt, final int nact, final int[] iact, final double[] rescon,
			final double[][] qfac, final int kopt, final int knew, final double del, final double[] step,
			final double[] gl, final double[] pqw, final double[] rstat, final int irstat, final double[] w,
			final int iw, final int[] ifeas) {

		double half, one, tenth, zero, test, temp, ss, sp, stp, vlag, vbig, stpsav = 0.0, gg, vgrad, ghg, vnew, ww,
				bigv, ctol, sum, resmax;
		int i, j, jsav = 0, k, ksav = 0, iflag;

		// Set some constants.
		half = 0.5;
		one = 1.0;
		tenth = 0.1;
		zero = 0.0;
		test = 0.2 * del;

		// Replace GL by the gradient of LFUNC at the trust region centre, and
		// set the elements of RSTAT.
		for (k = 1; k <= npt; k++) {
			temp = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
			temp *= pqw[k - 1];
			BlasMath.daxpym(n, temp, xpt[k - 1], 1, gl, 1);
		}
		if (m > 0) {
			for (j = 1; j <= m; j++) {
				rstat[j - 1 + irstat - 1] = one;
				if (Math.abs(rescon[j - 1]) >= del) {
					rstat[j - 1 + irstat - 1] = -one;
				}
			}
			for (k = 1; k <= nact; k++) {
				rstat[iact[k - 1] - 1 + irstat - 1] = zero;
			}
		}

		// Find the greatest modulus of LFUNC on a line through XOPT and
		// another interpolation point within the trust region.
		iflag = 0;
		vbig = zero;
		for (k = 1; k <= npt; k++) {
			if (k == kopt) {
				continue;
			}
			ss = zero;
			sp = zero;
			for (i = 1; i <= n; i++) {
				temp = xpt[k - 1][i - 1] - xopt[i - 1];
				ss += temp * temp;
				sp += gl[i - 1] * temp;
			}
			stp = -del / Math.sqrt(ss);
			if (k == knew) {
				if (sp * (sp - one) < zero) {
					stp = -stp;
				}
				vlag = Math.abs(stp * sp) + stp * stp * Math.abs(sp - one);
			} else {
				vlag = Math.abs(stp * (one - stp) * sp);
			}
			if (vlag > vbig) {
				ksav = k;
				stpsav = stp;
				vbig = vlag;
			}
		}

		// Set STEP to the move that gives the greatest modulus calculated above.
		// This move may be replaced by a steepest ascent step from XOPT.
		gg = zero;
		for (i = 1; i <= n; i++) {
			gg += gl[i - 1] * gl[i - 1];
			step[i - 1] = stpsav * (xpt[ksav - 1][i - 1] - xopt[i - 1]);
		}
		vgrad = del * Math.sqrt(gg);
		if (vgrad > tenth * vbig) {

			// Make the replacement if it provides a larger value of VBIG.
			ghg = zero;
			for (k = 1; k <= npt; k++) {
				temp = BlasMath.ddotm(n, xpt[k - 1], 1, gl, 1);
				ghg += pqw[k - 1] * temp * temp;
			}
			vnew = vgrad + Math.abs(half * del * del * ghg / gg);
			if (vnew > vbig) {
				vbig = vnew;
				stp = del / Math.sqrt(gg);
				if (ghg < zero) {
					stp = -stp;
				}
				BlasMath.dscal1(n, stp, gl, 1, step, 1);
			}
			if (!(nact == 0) && !(nact == n)) {

				// Overwrite GL by its projection. Then set VNEW to the greatest
				// value of |LFUNC| on the projected gradient from XOPT subject to
				// the trust region bound. If VNEW is sufficiently large, then STEP
				// may be changed to a move along the projected gradient.
				for (k = nact + 1; k <= n; ++k) {
					w[k - 1 + iw - 1] = zero;
					for (i = 1; i <= n; i++) {
						w[k - 1 + iw - 1] += gl[i - 1] * qfac[i - 1][k - 1];
					}
				}
				gg = zero;
				for (i = 1; i <= n; i++) {
					gl[i - 1] = zero;
					for (k = nact + 1; k <= n; ++k) {
						gl[i - 1] += qfac[i - 1][k - 1] * w[k - 1 + iw - 1];
					}
					gg += gl[i - 1] * gl[i - 1];
				}
				vgrad = del * Math.sqrt(gg);
				if (vgrad > tenth * vbig) {
					ghg = zero;
					for (k = 1; k <= npt; k++) {
						temp = BlasMath.ddotm(n, xpt[k - 1], 1, gl, 1);
						ghg += pqw[k - 1] * temp * temp;
					}
					vnew = vgrad + Math.abs(half * del * del * ghg / gg);

					// Set W to the possible move along the projected gradient.
					stp = del / Math.sqrt(gg);
					if (ghg < zero) {
						stp = -stp;
					}
					ww = zero;
					for (i = 1; i <= n; i++) {
						w[i - 1 + iw - 1] = stp * gl[i - 1];
						ww += w[i - 1 + iw - 1] * w[i - 1 + iw - 1];
					}

					// Set STEP to W if W gives a sufficiently large value of the
					// modulus
					// of the Lagrange function, and if W either preserves
					// feasibility
					// or gives a constraint violation of at least 0.2*DEL. The
					// purpose
					// of CTOL below is to provide a check on feasibility that
					// includes
					// a tolerance for contributions from computer rounding errors.
					if (vnew / vbig >= 0.2) {
						ifeas[0] = 1;
						bigv = 0.0;
						j = 0;
						while (true) {
							++j;
							if (j <= m) {
								if (rstat[j - 1 + irstat - 1] == one) {
									temp = -rescon[j - 1];
									temp += BlasMath.ddotm(n, w, iw, amat[j - 1], 1);
									bigv = Math.max(bigv, temp);
								}
								if (bigv < test) {
									continue;
								}
								ifeas[0] = 0;
							}
							break;
						}
						ctol = zero;
						temp = 0.01 * Math.sqrt(ww);
						if ((bigv > zero) && (bigv < temp)) {
							for (k = 1; k <= nact; k++) {
								j = iact[k - 1];
								sum = BlasMath.ddotm(n, w, iw, amat[j - 1], 1);
								ctol = Math.max(ctol, Math.abs(sum));
							}
						}
						if ((bigv <= 10.0 * ctol) || (bigv >= test)) {
							System.arraycopy(w, iw - 1, step, 0, n);
							return;
						}
					}
				}
			}
		}

		// Calculate the greatest constraint violation at XOPT+STEP with STEP at
		// its original value. Modify STEP if this violation is unacceptable.
		ifeas[0] = 1;
		bigv = resmax = 0.0;
		j = 0;
		while (true) {
			++j;
			if (j <= m) {
				if (rstat[j - 1 + irstat - 1] < zero) {
					continue;
				}
				temp = -rescon[j - 1];
				temp += BlasMath.ddotm(n, step, 1, amat[j - 1], 1);
				resmax = Math.max(resmax, temp);
				if (temp < test) {
					if (temp <= bigv) {
						continue;
					}
					bigv = temp;
					jsav = j;
					ifeas[0] = -1;
					continue;
				}
				ifeas[0] = 0;
			}
			if (ifeas[0] == -1) {
				BlasMath.daxpym(n, test - bigv, amat[jsav - 1], 1, step, 1);
				ifeas[0] = 0;
			}
			break;
		}
		// Return the calculated STEP and the value of IFEAS.
	}

	private static void update(final int n, final int npt, final double[][] xpt, final double[][] bmat,
			final double[][] zmat, final int[] idz, final int ndim, final double[] sp, final double[] step,
			final int kopt, final int[] knew, final double[] vlag, final double[] w) {

		double half, zero, one, sum, bsum, beta, dx, ssq, denmax, hdiag, temp, tempa, tempb = 0.0, denabs, distsq,
				alpha, tau, tausq, denom, sqrtdn, scala, scalb;
		int nptm, i, j, ja, jb, jp, jl, k, iflag;

		// Set some constants.
		half = 0.5;
		one = 1.0;
		zero = 0.0;
		nptm = npt - n - 1;

		// Calculate VLAG and BETA for the current choice of STEP. The first NPT
		// elements of VLAG are set to the values of the Lagrange functions at
		// XPT(KOPT,.)+STEP(.). The first NPT components of W_check are held
		// in W, where W_check is defined in a paper on the updating method.
		for (k = 1; k <= npt; k++) {
			w[k - 1] = sp[npt + k - 1] * (half * sp[npt + k - 1] + sp[k - 1]);
			sum = BlasMath.ddotm(n, bmat[k - 1], 1, step, 1);
			vlag[k - 1] = sum;
		}
		beta = zero;
		for (k = 1; k <= nptm; k++) {
			sum = zero;
			for (i = 1; i <= npt; i++) {
				sum += zmat[i - 1][k - 1] * w[i - 1];
			}
			if (k < idz[0]) {
				beta += sum * sum;
				sum = -sum;
			} else {
				beta -= sum * sum;
			}
			for (i = 1; i <= npt; i++) {
				vlag[i - 1] += sum * zmat[i - 1][k - 1];
			}
		}
		bsum = dx = ssq = zero;
		for (j = 1; j <= n; j++) {
			sum = zero;
			for (i = 1; i <= npt; i++) {
				sum += w[i - 1] * bmat[i - 1][j - 1];
			}
			bsum += sum * step[j - 1];
			jp = npt + j;
			sum += BlasMath.ddotm(n, bmat[jp - 1], 1, step, 1);
			vlag[jp - 1] = sum;
			bsum += sum * step[j - 1];
			dx += step[j - 1] * xpt[kopt - 1][j - 1];
			ssq += step[j - 1] * step[j - 1];
		}
		beta = dx * dx + ssq * (sp[kopt - 1] + dx + dx + half * ssq) + beta - bsum;
		vlag[kopt - 1] += one;

		// If KNEW is zero initially, then pick the index of the interpolation
		// point to be deleted, by maximizing the absolute value of the
		// denominator of the updating formula times a weighting factor.
		if (knew[0] == 0) {
			denmax = zero;
			for (k = 1; k <= npt; k++) {
				hdiag = zero;
				for (j = 1; j <= nptm; j++) {
					temp = one;
					if (j < idz[0]) {
						temp = -one;
					}
					hdiag += temp * zmat[k - 1][j - 1] * zmat[k - 1][j - 1];
				}
				denabs = Math.abs(beta * hdiag + vlag[k - 1] * vlag[k - 1]);
				distsq = zero;
				for (j = 1; j <= n; j++) {
					distsq += (xpt[k - 1][j - 1] - xpt[kopt - 1][j - 1]) * (xpt[k - 1][j - 1] - xpt[kopt - 1][j - 1]);
				}
				temp = denabs * distsq * distsq;
				if (temp > denmax) {
					denmax = temp;
					knew[0] = k;
				}
			}
		}

		// Apply the rotations that put zeros in the KNEW-th row of ZMAT.
		jl = 1;
		if (nptm >= 2) {
			for (j = 2; j <= nptm; j++) {
				if (j == idz[0]) {
					jl = idz[0];
				} else if (zmat[knew[0] - 1][j - 1] != zero) {
					temp = RealMath.hypot(zmat[knew[0] - 1][jl - 1], zmat[knew[0] - 1][j - 1]);
					tempa = zmat[knew[0] - 1][jl - 1] / temp;
					tempb = zmat[knew[0] - 1][j - 1] / temp;
					for (i = 1; i <= npt; i++) {
						temp = tempa * zmat[i - 1][jl - 1] + tempb * zmat[i - 1][j - 1];
						zmat[i - 1][j - 1] = tempa * zmat[i - 1][j - 1] - tempb * zmat[i - 1][jl - 1];
						zmat[i - 1][jl - 1] = temp;
					}
					zmat[knew[0] - 1][j - 1] = zero;
				}
			}
		}

		// Put the first NPT components of the KNEW-th column of the Z Z^T matrix
		// into W, and calculate the parameters of the updating formula.
		tempa = zmat[knew[0] - 1][1 - 1];
		if (idz[0] >= 2) {
			tempa = -tempa;
		}
		if (jl > 1) {
			tempb = zmat[knew[0] - 1][jl - 1];
		}
		for (i = 1; i <= npt; i++) {
			w[i - 1] = tempa * zmat[i - 1][1 - 1];
			if (jl > 1) {
				w[i - 1] += tempb * zmat[i - 1][jl - 1];
			}
		}
		alpha = w[knew[0] - 1];
		tau = vlag[knew[0] - 1];
		tausq = tau * tau;
		denom = alpha * beta + tausq;
		vlag[knew[0] - 1] -= one;
		if (denom == zero) {
			knew[0] = 0;
			return;
		}
		sqrtdn = Math.sqrt(Math.abs(denom));

		// Complete the updating of ZMAT when there is only one nonzero element
		// in the KNEW-th row of the new matrix ZMAT. IFLAG is set to one when
		// the value of IDZ is going to be reduced.
		iflag = 0;
		if (jl == 1) {
			tempa = tau / sqrtdn;
			tempb = zmat[knew[0] - 1][1 - 1] / sqrtdn;
			for (i = 1; i <= npt; i++) {
				zmat[i - 1][1 - 1] = tempa * zmat[i - 1][1 - 1] - tempb * vlag[i - 1];
			}
			if (denom < zero) {
				if (idz[0] == 1) {
					idz[0] = 2;
				} else {
					iflag = 1;
				}
			}
		} else {

			// Complete the updating of ZMAT in the alternative case.
			ja = 1;
			if (beta >= zero) {
				ja = jl;
			}
			jb = jl + 1 - ja;
			temp = zmat[knew[0] - 1][jb - 1] / denom;
			tempa = temp * beta;
			tempb = temp * tau;
			temp = zmat[knew[0] - 1][ja - 1];
			scala = one / Math.sqrt(Math.abs(beta) * temp * temp + tausq);
			scalb = scala * sqrtdn;
			for (i = 1; i <= npt; i++) {
				zmat[i - 1][ja - 1] = scala * (tau * zmat[i - 1][ja - 1] - temp * vlag[i - 1]);
				zmat[i - 1][jb - 1] = scalb * (zmat[i - 1][jb - 1] - tempa * w[i - 1] - tempb * vlag[i - 1]);
			}
			if (denom <= zero) {
				if (beta < zero) {
					++idz[0];
				} else {
					iflag = 1;
				}
			}
		}

		// Reduce IDZ when the diagonal part of the ZMAT times Diag(DZ) times
		// ZMAT^T factorization gains another positive element. Then exchange
		// the first and IDZ-th columns of ZMAT.
		if (iflag == 1) {
			--idz[0];
			for (i = 1; i <= npt; i++) {
				temp = zmat[i - 1][1 - 1];
				zmat[i - 1][1 - 1] = zmat[i - 1][idz[0] - 1];
				zmat[i - 1][idz[0] - 1] = temp;
			}
		}

		// Finally, update the matrix BMAT.
		for (j = 1; j <= n; j++) {
			jp = npt + j;
			w[jp - 1] = bmat[knew[0] - 1][j - 1];
			tempa = (alpha * vlag[jp - 1] - tau * w[jp - 1]) / denom;
			tempb = (-beta * w[jp - 1] - tau * vlag[jp - 1]) / denom;
			for (i = 1; i <= jp; i++) {
				bmat[i - 1][j - 1] += tempa * vlag[i - 1] + tempb * w[i - 1];
				if (i > npt) {
					bmat[jp - 1][i - npt - 1] = bmat[i - 1][j - 1];
				}
			}
		}
	}

	private static void prelim(final Function<? super double[], Double> func, final int n, final int npt, final int m,
			final double[][] amat, final double[] b, final double[] x, final double rhobeg, final int iprint,
			final double[] xbase, final double[][] xpt, final double[] fval, final double[] xsav, final double[] xopt,
			final double[] gopt, final int[] kopt, final double[] hq, final double[] pq, final double[][] bmat,
			final double[][] zmat, final int[] idz, final int ndim, final double[] sp, final double[] rescon,
			final double[] step, final double[] pqw, final double[] w, final int[] fev) {

		double half, one, zero, rhosq, recip, reciq, test, temp, feas, bigv, resid, f;
		int nptm, kbase, i, j, jp, k, itemp, ipt, jpt, jsav = 0;
		final int[] nf = new int[1];

		// Set some constants.
		half = 0.5;
		one = 1.0;
		zero = 0.0;
		nptm = npt - n - 1;
		rhosq = rhobeg * rhobeg;
		recip = one / rhosq;
		reciq = Math.sqrt(half) / rhosq;
		test = 0.2 * rhobeg;
		idz[0] = kbase = 1;

		// Set the initial elements of XPT, BMAT, SP and ZMAT to zero.
		System.arraycopy(x, 0, xbase, 0, n);
		for (j = 1; j <= n; ++j) {
			for (k = 1; k <= npt; ++k) {
				xpt[k - 1][j - 1] = zero;
			}
			for (i = 1; i <= ndim; ++i) {
				bmat[i - 1][j - 1] = zero;
			}
		}
		for (k = 1; k <= npt; ++k) {
			sp[k - 1] = zero;
			Arrays.fill(zmat[k - 1], 0, npt - n - 1, zero);
		}

		// Set the nonzero coordinates of XPT(K,.), K=1,2,...,min[2*N+1,NPT]
		// but they may be altered later to make a constraint violation
		// sufficiently large. The initial nonzero elements of BMAT and of
		// the first min[N,NPT-N-1] columns of ZMAT are set also.
		for (j = 1; j <= n; ++j) {
			xpt[j + 1 - 1][j - 1] = rhobeg;
			if (j < npt - n) {
				jp = n + j + 1;
				xpt[jp - 1][j - 1] = -rhobeg;
				bmat[j + 1 - 1][j - 1] = half / rhobeg;
				bmat[jp - 1][j - 1] = -half / rhobeg;
				zmat[1 - 1][j - 1] = -reciq - reciq;
				zmat[j + 1 - 1][j - 1] = reciq;
				zmat[jp - 1][j - 1] = reciq;
			} else {
				bmat[1 - 1][j - 1] = -one / rhobeg;
				bmat[j + 1 - 1][j - 1] = one / rhobeg;
				bmat[npt + j - 1][j - 1] = half * rhosq;
			}
		}

		// Set the remaining initial nonzero elements of XPT and ZMAT when the
		// number of interpolation points exceeds 2*N+1.
		if (npt > 2 * n + 1) {
			for (k = n + 1; k <= npt - n - 1; ++k) {
				itemp = (k - 1) / n;
				ipt = k - itemp * n;
				jpt = ipt + itemp;
				if (jpt > n) {
					jpt -= n;
				}
				xpt[n + k + 1 - 1][ipt - 1] = rhobeg;
				xpt[n + k + 1 - 1][jpt - 1] = rhobeg;
				zmat[1 - 1][k - 1] = recip;
				zmat[ipt + 1 - 1][k - 1] = -recip;
				zmat[jpt + 1 - 1][k - 1] = -recip;
				zmat[n + k + 1 - 1][k - 1] = recip;
			}
		}

		// Update the constraint right hand sides to allow for the shift XBASE.
		if (m > 0) {
			for (j = 1; j <= m; j++) {
				temp = BlasMath.ddotm(n, amat[j - 1], 1, xbase, 1);
				b[j - 1] -= temp;
			}
		}

		// Go through the initial points, shifting every infeasible point if
		// necessary so that its constraint violation is at least 0.2*RHOBEG.
		for (nf[0] = 1; nf[0] <= npt; ++nf[0]) {
			feas = one;
			bigv = zero;
			j = 0;
			while (true) {
				++j;
				if ((j <= m) && (nf[0] >= 2)) {
					resid = -b[j - 1];
					resid += BlasMath.ddotm(n, xpt[nf[0] - 1], 1, amat[j - 1], 1);
					if (resid <= bigv) {
						continue;
					}
					bigv = resid;
					jsav = j;
					if (resid <= test) {
						feas = -one;
						continue;
					}
					feas = zero;
				}
				break;
			}

			if (feas < zero) {
				for (i = 1; i <= n; i++) {
					step[i - 1] = xpt[nf[0] - 1][i - 1] + (test - bigv) * amat[jsav - 1][i - 1];
				}
				for (k = 1; k <= npt; k++) {
					sp[npt + k - 1] = BlasMath.ddotm(n, xpt[k - 1], 1, step, 1);
				}
				update(n, npt, xpt, bmat, zmat, idz, ndim, sp, step, kbase, nf, pqw, w);
				System.arraycopy(step, 0, xpt[nf[0] - 1], 0, n);
			}

			// Calculate the objective function at the current interpolation point,
			// and set KOPT to the index of the first trust region centre.
			BlasMath.dxpy1(n, xbase, 1, xpt[nf[0] - 1], 1, x, 1);
			// f = feas;
			f = func.apply(x);
			++fev[0];
			if (nf[0] == 1) {
				kopt[0] = 1;
			} else if (f < fval[kopt[0] - 1] && feas > zero) {
				kopt[0] = nf[0];
			}
			fval[nf[0] - 1] = f;
		}

		// Set PQ for the first quadratic model.
		for (j = 1; j <= nptm; j++) {
			w[j - 1] = zero;
			for (k = 1; k <= npt; k++) {
				w[j - 1] += zmat[k - 1][j - 1] * fval[k - 1];
			}
		}
		for (k = 1; k <= npt; k++) {
			pq[k - 1] = BlasMath.ddotm(nptm, zmat[k - 1], 1, w, 1);
		}

		// Set XOPT, SP, GOPT and HQ for the first quadratic model.
		System.arraycopy(xpt[kopt[0] - 1], 0, xopt, 0, n);
		for (j = 1; j <= n; j++) {
			xsav[j - 1] = xbase[j - 1] + xopt[j - 1];
			gopt[j - 1] = zero;
		}
		for (k = 1; k <= npt; k++) {
			sp[k - 1] = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
			temp = pq[k - 1] * sp[k - 1];
			for (j = 1; j <= n; j++) {
				gopt[j - 1] += fval[k - 1] * bmat[k - 1][j - 1] + temp * xpt[k - 1][j - 1];
			}
		}
		Arrays.fill(hq, 0, (n * n + n) / 2, zero);

		// Set the initial elements of RESCON.
		for (j = 1; j <= m; j++) {
			temp = b[j - 1];
			temp -= BlasMath.ddotm(n, xopt, 1, amat[j - 1], 1);
			temp = Math.max(temp, zero);
			if (temp >= rhobeg) {
				temp = -temp;
			}
			rescon[j - 1] = temp;
		}
	}
}
