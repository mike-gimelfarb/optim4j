/*
BOBYQA---Bound Optimization BY Quadratic Approximation.
Copyright (C) 2009 M. J. D. Powell (University of Cambridge)

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
package opt.multivariate.unconstrained.order0.quad;

import java.util.Arrays;
import java.util.function.Function;

import opt.OptimizerSolution;
import opt.multivariate.GradientFreeOptimizer;
import utils.BlasMath;
import utils.IntMath;
import utils.RealMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] Powell, Michael JD. "The BOBYQA algorithm for bound constrained
 * optimization without derivatives." Cambridge NA Report NA2009/06, University
 * of Cambridge, Cambridge (2009): 26-46.
 */
public final class BobyqaAlgorithm extends GradientFreeOptimizer {

	private final Function<? super Integer, Integer> mySize;
	private final double myRho0;
	private final int myMaxFEvals;

	/**
	 *
	 * @param tolerance
	 * @param initialRadius
	 * @param maxEvaluations
	 * @param sizeFunction
	 */
	public BobyqaAlgorithm(final double tolerance, final double initialRadius, final int maxEvaluations,
			final Function<? super Integer, Integer> sizeFunction) {
		super(tolerance);
		mySize = sizeFunction;
		myRho0 = initialRadius;
		myMaxFEvals = maxEvaluations;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialStep
	 * @param maxEvaluations
	 */
	public BobyqaAlgorithm(final double tolerance, final double initialStep, final int maxEvaluations) {
		this(tolerance, initialStep, maxEvaluations, d -> 2 * d + 1);
	}

	@Override
	public final void initialize(final Function<? super double[], Double> function, final double[] guess) {
		// nothing to do here
	}

	@Override
	public final void iterate() {
		// nothing to do here
	}

	@Override
	public final OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> func,
			final double[] guess) {

		// prepare variables
		final int n = guess.length;
		final double[] lower = new double[n];
		final double[] upper = new double[n];
		for (int i = 0; i < n; ++i) {
			lower[i] = 1.0e-60;
			upper[i] = 1.0e+60;
		}

		// call main subroutine
		return optimize(func, guess, lower, upper);
	}

	/**
	 *
	 * @param func
	 * @param guess
	 * @param lower
	 * @param upper
	 * @return
	 */
	public final OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> func,
			final double[] guess, final double[] lower, final double[] upper) {

		// prepare variables
		final int d = guess.length;
		final int npt = mySize.apply(d);
		final int[] nf = new int[1];

		// call main subroutine
		// TODO: check convergence
		final double[] result = bobyqa(func, guess, lower, upper, npt, myRho0, myTol, myMaxFEvals, nf);
		return new OptimizerSolution<>(result, nf[0], 0, false);
	}

	private static double[] bobyqa(final Function<? super double[], Double> func, final double[] guess,
			final double[] xl, final double[] xu, final int npt, final double rhobeg, final double rhoend,
			final int maxfun, final int[] nf) {

		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		nf[0] = 0;

		// call main subroutine
		bobyqa(func, n, npt, x, xl, xu, rhobeg, rhoend, 0, maxfun, nf);
		return x;
	}

	private static void bobyqa(final Function<? super double[], Double> func, final int n, final int npt,
			final double[] x, final double[] xl, final double[] xu, final double rhobeg, final double rhoend,
			final int iprint, final int maxfun, final int[] nf) {

		final double[] sl = new double[n], su = new double[n], xbase = new double[n], fval = new double[npt],
				xopt = new double[n], gopt = new double[n], hq = new double[n * (n + 1) / 2], pq = new double[npt],
				xnew = new double[n], xalt = new double[n], d = new double[n], vlag = new double[npt + n],
				w = new double[3 * (npt + n)];
		final double[][] xpt = new double[npt][n], bmat = new double[npt + n][n], zmat = new double[npt][npt - (n + 1)];

		// Return if the value of NPT is unacceptable.
		final int np = n + 1;
		if (npt < n + 2 || npt > (n + 2) * np / 2) {
			return;
		}

		final double zero = 0.0;
		for (int j = 1; j <= n; ++j) {
			final double temp = xu[j - 1] - xl[j - 1];
			if (temp < rhobeg + rhobeg) {
				return;
			}
			sl[j - 1] = xl[j - 1] - x[j - 1];
			su[j - 1] = xu[j - 1] - x[j - 1];
			if (sl[j - 1] >= -rhobeg) {
				if (sl[j - 1] >= zero) {
					x[j - 1] = xl[j - 1];
					sl[j - 1] = zero;
					su[j - 1] = temp;
				} else {
					x[j - 1] = xl[j - 1] + rhobeg;
					sl[j - 1] = -rhobeg;
					su[j - 1] = Math.max(xu[j - 1] - x[j - 1], rhobeg);
				}
			} else if (su[j - 1] <= rhobeg) {
				if (su[j - 1] <= zero) {
					x[j - 1] = xu[j - 1];
					sl[j - 1] = -temp;
					su[j - 1] = zero;
				} else {
					x[j - 1] = xu[j - 1] - rhobeg;
					sl[j - 1] = Math.min(xl[j - 1] - x[j - 1], -rhobeg);
					su[j - 1] = rhobeg;
				}
			}
		}

		// Make the call of BOBYQB.
		bobyqb(func, n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, xbase, xpt, fval, xopt, gopt, hq, pq, bmat,
				zmat, npt + n, sl, su, xnew, xalt, d, vlag, w, nf);
	}

	private static void bobyqb(final Function<? super double[], Double> func, final int n, final int npt,
			final double[] x, final double[] xl, final double[] xu, final double rhobeg, final double rhoend,
			final int iprint, final int maxfun, final double[] xbase, final double[][] xpt, final double[] fval,
			final double[] xopt, final double[] gopt, final double[] hq, final double[] pq, final double[][] bmat,
			final double[][] zmat, final int ndim, final double[] sl, final double[] su, final double[] xnew,
			final double[] xalt, final double[] d, final double[] vlag, final double[] w, final int[] nf) {

		final int[] kopt = new int[1], knew = new int[1];
		final double[] dsq = new double[1], crvmin = new double[1], alpha = new double[1], cauchy = new double[1],
				beta = new double[1], denom = new double[1];
		final double[][] ptsaux = new double[2][n];
		double xoptsq, fsave, rho, delta, diff, diffa, diffb, diffc = 0.0, temp, dnorm = 0.0, distsq = 0.0, errbig,
				frhosq, bdtest, bdtol, curv, sumpq, fracsq, sum, sumz, sumw, adelt = 0.0, suma, sumb, bsum, dx, delsq,
				scaden, den, hdiag, biglsq, f, fopt, vquad, densav, ratio = 0.0, pqold, gqsq, gisq, dist, temp2;
		int i, ih, ip, itest, j, jj, jp, k, ksav, kbase, nresc, ntrits, nfsav;

		// Set some constants.
		final double half = 0.5, one = 1.0, ten = 10.0, tenth = 0.1, two = 2.0, zero = 0.0;
		final int np = n + 1;
		final int nptm = npt - np;
		final int nh = (n * np) / 2;

		// The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
		// BMAT and ZMAT for the first iteration, with the corresponding values of
		// of NF and KOPT, which are the number of calls of CALFUN so far and the
		// index of the interpolation point at the trust region centre. Then the
		// initial XOPT is set too. The branch to label 720 occurs if MAXFUN is
		// less than NPT. GOPT will be updated if KOPT is different from KBASE.
		prelim(func, n, npt, x, xl, xu, rhobeg, iprint, maxfun, xbase, xpt, fval, gopt, hq, pq, bmat, zmat, ndim, sl,
				su, nf, kopt);
		System.arraycopy(xpt[kopt[0] - 1], 0, xopt, 0, n);
		xoptsq = BlasMath.ddotm(n, xopt, 1, xopt, 1);
		fsave = fval[1 - 1];
		if (nf[0] < npt) {
			if (fval[kopt[0] - 1] <= fsave) {
				for (i = 1; i <= n; ++i) {
					x[i - 1] = Math.min(Math.max(xl[i - 1], xbase[i - 1] + xopt[i - 1]), xu[i - 1]);
					if (xopt[i - 1] == sl[i - 1]) {
						x[i - 1] = xl[i - 1];
					}
					if (xopt[i - 1] == su[i - 1]) {
						x[i - 1] = xu[i - 1];
					}
				}
			}
			return;
		}
		kbase = 1;

		// Complete the settings that are required for the iterative procedure.
		rho = rhobeg;
		delta = rho;
		nresc = nf[0];
		ntrits = 0;
		diffa = diffb = zero;
		itest = 0;
		nfsav = nf[0];
		int gotoflag = 20;

		while (true) {

			if (gotoflag == 20) {

				// Update GOPT if necessary before the first iteration and after each
				// call of RESCUE that makes a call of CALFUN.
				if (kopt[0] != kbase) {
					ih = 0;
					for (j = 1; j <= n; j++) {
						for (i = 1; i <= j; i++) {
							++ih;
							if (i < j) {
								gopt[j - 1] += hq[ih - 1] * xopt[i - 1];
							}
							gopt[i - 1] += hq[ih - 1] * xopt[j - 1];
						}
					}
					if (nf[0] > npt) {
						for (k = 1; k <= npt; k++) {
							temp = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
							temp *= pq[k - 1];
							BlasMath.daxpym(n, temp, xpt[k - 1], 1, gopt, 1);
						}
					}
				}
				gotoflag = 60;
			}

			if (gotoflag == 60) {

				// Generate the next point in the trust region that provides a small
				// value
				// of the quadratic model subject to the constraints on the
				// variables...
				trsbox(n, npt, xpt, xopt, gopt, hq, pq, sl, su, delta, xnew, d, w, 1, w, np, w, np + n, w, np + 2 * n,
						w, np + 3 * n, dsq, crvmin);
				dnorm = Math.min(delta, Math.sqrt(dsq[0]));
				if (dnorm < half * rho) {
					ntrits = -1;
					distsq = (ten * rho) * (ten * rho);
					if (nf[0] <= nfsav + 2) {
						gotoflag = 650;
					} else {

						// The following choice between labels 650 and 680 depends on
						// whether or
						// not our work with the current RHO seems to be complete...
						errbig = Math.max(diffa, Math.max(diffb, diffc));
						frhosq = 0.125 * rho * rho;
						if (crvmin[0] > zero && errbig > frhosq * crvmin[0]) {
							gotoflag = 650;
						} else {
							bdtol = errbig / rho;
							boolean skipto650 = false;
							for (j = 1; j <= n; ++j) {
								bdtest = bdtol;
								if (xnew[j - 1] == sl[j - 1]) {
									bdtest = w[j - 1];
								}
								if (xnew[j - 1] == su[j - 1]) {
									bdtest = -w[j - 1];
								}
								if (bdtest < bdtol) {
									curv = hq[(j + j * j) / 2 - 1];
									for (k = 1; k <= npt; ++k) {
										curv += pq[k - 1] * xpt[k - 1][j - 1] * xpt[k - 1][j - 1];
									}
									bdtest += half * curv * rho;
									if (bdtest < bdtol) {
										skipto650 = true;
										break;
									}
								}
							}
							if (skipto650) {
								gotoflag = 650;
							} else {
								gotoflag = 680;
							}
						}
					}
				} else {
					++ntrits;
					gotoflag = 90;
				}
			}

			if (gotoflag == 90) {

				// Severe cancellation is likely to occur if XOPT is too far from
				// XBASE.
				// If the following test holds, then XBASE is shifted so that XOPT
				// becomes
				// zero. The appropriate changes are made to BMAT and to the second
				// derivatives of the current model, beginning with the changes to
				// BMAT
				// that do not depend on ZMAT. VLAG is used temporarily for working
				// space.
				if (dsq[0] <= 1.0e-3 * xoptsq) {

					fracsq = 0.25 * xoptsq;
					sumpq = zero;
					for (k = 1; k <= npt; k++) {
						sumpq += pq[k - 1];
						sum = -half * xoptsq;
						sum += BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
						w[npt + k - 1] = sum;
						temp = fracsq - half * sum;
						for (i = 1; i <= n; i++) {
							w[i - 1] = bmat[k - 1][i - 1];
							vlag[i - 1] = sum * xpt[k - 1][i - 1] + temp * xopt[i - 1];
							ip = npt + i;
							for (j = 1; j <= i; j++) {
								bmat[ip - 1][j - 1] += w[i - 1] * vlag[j - 1] + vlag[i - 1] * w[j - 1];
							}
						}
					}

					// Then the revisions of BMAT that depend on ZMAT are calculated.
					for (jj = 1; jj <= nptm; jj++) {
						sumz = sumw = zero;
						for (k = 1; k <= npt; k++) {
							sumz += zmat[k - 1][jj - 1];
							vlag[k - 1] = w[npt + k - 1] * zmat[k - 1][jj - 1];
							sumw += vlag[k - 1];
						}
						for (j = 1; j <= n; j++) {
							sum = (fracsq * sumz - half * sumw) * xopt[j - 1];
							for (k = 1; k <= npt; k++) {
								sum += vlag[k - 1] * xpt[k - 1][j - 1];
							}
							w[j - 1] = sum;
							for (k = 1; k <= npt; k++) {
								bmat[k - 1][j - 1] += sum * zmat[k - 1][jj - 1];
							}
						}
						for (i = 1; i <= n; i++) {
							ip = i + npt;
							temp = w[i - 1];
							BlasMath.daxpym(i, temp, w, 1, bmat[ip - 1], 1);
						}
					}

					// The following instructions complete the shift, including the
					// changes
					// to the second derivative parameters of the quadratic model.
					ih = 0;
					for (j = 1; j <= n; j++) {
						w[j - 1] = -half * sumpq * xopt[j - 1];
						for (k = 1; k <= npt; k++) {
							w[j - 1] += pq[k - 1] * xpt[k - 1][j - 1];
							xpt[k - 1][j - 1] -= xopt[j - 1];
						}
						for (i = 1; i <= j; i++) {
							++ih;
							hq[ih - 1] += w[i - 1] * xopt[j - 1] + xopt[i - 1] * w[j - 1];
							bmat[npt + i - 1][j - 1] = bmat[npt + j - 1][i - 1];
						}
					}
					for (i = 1; i <= n; i++) {
						xbase[i - 1] += xopt[i - 1];
						xnew[i - 1] -= xopt[i - 1];
						sl[i - 1] -= xopt[i - 1];
						su[i - 1] -= xopt[i - 1];
						xopt[i - 1] = zero;
					}
					xoptsq = zero;
				}
				if (ntrits == 0) {
					gotoflag = 210;
				} else {
					gotoflag = 230;
				}
			}

			if (gotoflag == 190) {

				// XBASE is also moved to XOPT by a call of RESCUE. This calculation
				// is
				// more expensive than the previous shift, because new matrices BMAT
				// and
				// ZMAT are generated from scratch, which may include the replacement
				// of
				// interpolation points whose positions seem to be causing near
				// linear
				// dependence in the interpolation conditions....
				nfsav = nf[0];
				kbase = kopt[0];
				for (int i1 = 1; i1 <= 2; ++i1) {
					for (int i2 = 1; i2 <= n; ++i2) {
						ptsaux[i1 - 1][i2 - 1] = w[2 * (i2 - 1) + i1 - 1];
					}
				}
				rescue(func, n, npt, xl, xu, iprint, maxfun, xbase, xpt, fval, xopt, gopt, hq, pq, bmat, zmat, ndim, sl,
						su, nf, delta, kopt, vlag, ptsaux, w, n + np, w, ndim + np);
				for (int i1 = 1; i1 <= 2; ++i1) {
					for (int i2 = 1; i2 <= n; ++i2) {
						w[2 * (i2 - 1) + i1 - 1] = ptsaux[i1 - 1][i2 - 1];
					}
				}

				// XOPT is updated now in case the branch below to label 720 is
				// taken.
				// Any updating of GOPT occurs after the branch below to label 20,
				// which
				// leads to a trust region iteration as does the branch to label 60.
				xoptsq = zero;
				if (kopt[0] != kbase) {
					System.arraycopy(xpt[kopt[0] - 1], 0, xopt, 0, n);
					xoptsq += BlasMath.ddotm(n, xopt, 1, xopt, 1);
				}
				if (nf[0] < 0) {
					nf[0] = maxfun;
					break;
				} else {
					nresc = nf[0];
					if (nfsav < nf[0]) {
						nfsav = nf[0];
						gotoflag = 20;
						continue;
					} else if (ntrits > 0) {
						gotoflag = 60;
						continue;
					} else {
						gotoflag = 210;
					}
				}
			}

			if (gotoflag == 210) {

				// Pick two alternative vectors of variables, relative to XBASE, that
				// are suitable as new positions of the KNEW-th interpolation
				// point....
				altmov(n, npt, xpt, xopt, bmat, zmat, ndim, sl, su, kopt, knew, adelt, xnew, xalt, alpha, cauchy, w, 1,
						w, np, w, ndim + 1);
				for (i = 1; i <= n; ++i) {
					d[i - 1] = xnew[i - 1] - xopt[i - 1];
				}
				gotoflag = 230;
			}

			if (gotoflag == 230) {

				// Calculate VLAG and BETA for the current choice of D. The scalar
				// product of D with XPT(K,.) is going to be held in W(NPT+K) for
				// use when VQUAD is calculated.
				for (k = 1; k <= npt; k++) {
					suma = sumb = sum = zero;
					for (j = 1; j <= n; j++) {
						suma += xpt[k - 1][j - 1] * d[j - 1];
						sumb += xpt[k - 1][j - 1] * xopt[j - 1];
						sum += bmat[k - 1][j - 1] * d[j - 1];
					}
					w[k - 1] = suma * (half * suma + sumb);
					vlag[k - 1] = sum;
					w[npt + k - 1] = suma;
				}
				beta[0] = zero;
				for (jj = 1; jj <= nptm; jj++) {
					sum = zero;
					for (k = 1; k <= npt; k++) {
						sum += zmat[k - 1][jj - 1] * w[k - 1];
					}
					beta[0] -= sum * sum;
					for (k = 1; k <= npt; k++) {
						vlag[k - 1] += sum * zmat[k - 1][jj - 1];
					}
				}
				dsq[0] = bsum = dx = zero;
				for (j = 1; j <= n; j++) {
					dsq[0] += d[j - 1] * d[j - 1];
					sum = zero;
					for (k = 1; k <= npt; k++) {
						sum += w[k - 1] * bmat[k - 1][j - 1];
					}
					bsum += sum * d[j - 1];
					jp = npt + j;
					sum += BlasMath.ddotm(n, bmat[jp - 1], 1, d, 1);
					vlag[jp - 1] = sum;
					bsum += sum * d[j - 1];
					dx += d[j - 1] * xopt[j - 1];
				}
				beta[0] = dx * dx + dsq[0] * (xoptsq + dx + dx + half * dsq[0]) + beta[0] - bsum;
				vlag[kopt[0] - 1] += one;

				// If NTRITS is zero, the denominator may be increased by replacing
				// the step D of ALTMOV by a Cauchy step. Then RESCUE may be called
				// if
				// rounding errors have damaged the chosen denominator.
				if (ntrits == 0) {
					denom[0] = vlag[knew[0] - 1] * vlag[knew[0] - 1] + alpha[0] * beta[0];
					if (denom[0] < cauchy[0] && cauchy[0] > zero) {
						System.arraycopy(xalt, 0, xnew, 0, n);
						for (i = 1; i <= n; ++i) {
							d[i - 1] = xnew[i - 1] - xopt[i - 1];
						}
						cauchy[0] = zero;
						gotoflag = 230;
						continue;
					} else if (denom[0] <= half * vlag[knew[0] - 1] * vlag[knew[0] - 1]) {
						if (nf[0] > nresc) {
							gotoflag = 190;
							continue;
						} else {
							break;
						}
					} else {
						gotoflag = 360;
					}
				} else {

					// Alternatively, if NTRITS is positive, then set KNEW to the
					// index of
					// the next interpolation point to be deleted to make room for a
					// trust
					// region step. Again RESCUE may be called if rounding errors
					// have damaged
					// the chosen denominator, which is the reason for attempting to
					// select
					// KNEW before calculating the next value of the objective
					// function.
					delsq = delta * delta;
					scaden = biglsq = zero;
					knew[0] = 0;
					for (k = 1; k <= npt; k++) {
						if (k == kopt[0]) {
							continue;
						}
						hdiag = BlasMath.ddotm(nptm, zmat[k - 1], 1, zmat[k - 1], 1);
						den = beta[0] * hdiag + vlag[k - 1] * vlag[k - 1];
						distsq = zero;
						for (j = 1; j <= n; j++) {
							temp = xpt[k - 1][j - 1] - xopt[j - 1];
							distsq += (temp * temp);
						}
						temp = Math.max(one, (distsq / delsq) * (distsq / delsq));
						if (temp * den > scaden) {
							scaden = temp * den;
							knew[0] = k;
							denom[0] = den;
						}
						biglsq = Math.max(biglsq, temp * vlag[k - 1] * vlag[k - 1]);
					}
					if (scaden <= half * biglsq) {
						if (nf[0] > nresc) {
							gotoflag = 190;
							continue;
						} else {
							break;
						}
					} else {
						gotoflag = 360;
					}
				}
			}

			if (gotoflag == 360) {

				// Put the variables for the next calculation of the objective
				// function
				// in XNEW, with any adjustments for the bounds.
				//
				// Calculate the value of the objective function at XBASE+XNEW,
				// unless
				// the limit on the number of calculations of F has been reached.
				for (i = 1; i <= n; i++) {
					x[i - 1] = Math.min(Math.max(xl[i - 1], xbase[i - 1] + xnew[i - 1]), xu[i - 1]);
					if (xnew[i - 1] == sl[i - 1]) {
						x[i - 1] = xl[i - 1];
					}
					if (xnew[i - 1] == su[i - 1]) {
						x[i - 1] = xu[i - 1];
					}
				}
				if (nf[0] >= maxfun) {
					break;
				}
				++nf[0];
				f = func.apply(x);
				if (ntrits == -1) {
					fsave = f;
					break;
				}

				// Use the quadratic model to predict the change in F due to the step
				// D,
				// and set DIFF to the error of this prediction.
				fopt = fval[kopt[0] - 1];
				vquad = zero;
				ih = 0;
				for (j = 1; j <= n; j++) {
					vquad += d[j - 1] * gopt[j - 1];
					for (i = 1; i <= j; i++) {
						++ih;
						temp = d[i - 1] * d[j - 1];
						if (i == j) {
							temp *= half;
						}
						vquad += hq[ih - 1] * temp;
					}
				}
				for (k = 1; k <= npt; k++) {
					vquad += half * pq[k - 1] * w[npt + k - 1] * w[npt + k - 1];
				}
				diff = f - fopt - vquad;
				diffc = diffb;
				diffb = diffa;
				diffa = Math.abs(diff);
				if (dnorm > rho) {
					nfsav = nf[0];
				}

				// Pick the next value of DELTA after a trust region step.
				if (ntrits > 0) {
					if (vquad >= zero) {
						break;
					}
					ratio = (f - fopt) / vquad;
					if (ratio <= tenth) {
						delta = Math.min(half * delta, dnorm);
					} else if (ratio <= 0.7) {
						delta = Math.max(half * delta, dnorm);
					} else {
						delta = Math.max(half * delta, dnorm + dnorm);
					}
					if (delta <= 1.5 * rho) {
						delta = rho;
					}

					// Recalculate KNEW and DENOM if the new F is less than FOPT.
					if (f < fopt) {
						ksav = knew[0];
						densav = denom[0];
						delsq = delta * delta;
						scaden = biglsq = zero;
						knew[0] = 0;
						for (k = 1; k <= npt; k++) {
							hdiag = BlasMath.ddotm(nptm, zmat[k - 1], 1, zmat[k - 1], 1);
							den = beta[0] * hdiag + vlag[k - 1] * vlag[k - 1];
							distsq = zero;
							for (j = 1; j <= n; j++) {
								temp = xpt[k - 1][j - 1] - xnew[j - 1];
								distsq += (temp * temp);
							}
							temp = Math.max(one, (distsq / delsq) * (distsq / delsq));
							if (temp * den > scaden) {
								scaden = temp * den;
								knew[0] = k;
								denom[0] = den;
							}
							biglsq = Math.max(biglsq, temp * vlag[k - 1] * vlag[k - 1]);
						}
						if (scaden <= half * biglsq) {
							knew[0] = ksav;
							denom[0] = densav;
						}
					}
				}

				// Update BMAT and ZMAT, so that the KNEW-th interpolation point can
				// be
				// moved. Also update the second derivative terms of the model.
				update(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew[0], w, 1);
				ih = 0;
				pqold = pq[knew[0] - 1];
				pq[knew[0] - 1] = zero;
				for (i = 1; i <= n; i++) {
					temp = pqold * xpt[knew[0] - 1][i - 1];
					for (j = 1; j <= i; j++) {
						++ih;
						hq[ih - 1] += temp * xpt[knew[0] - 1][j - 1];
					}
				}
				for (jj = 1; jj <= nptm; jj++) {
					temp = diff * zmat[knew[0] - 1][jj - 1];
					for (k = 1; k <= npt; k++) {
						pq[k - 1] += temp * zmat[k - 1][jj - 1];
					}
				}

				// Include the new interpolation point, and make the changes to GOPT
				// at
				// the old XOPT that are caused by the updating of the quadratic
				// model.
				fval[knew[0] - 1] = f;
				System.arraycopy(xnew, 0, xpt[knew[0] - 1], 0, n);
				System.arraycopy(bmat[knew[0] - 1], 0, w, 0, n);
				for (k = 1; k <= npt; k++) {
					suma = BlasMath.ddotm(nptm, zmat[knew[0] - 1], 1, zmat[k - 1], 1);
					sumb = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
					temp = suma * sumb;
					BlasMath.daxpym(n, temp, xpt[k - 1], 1, w, 1);
				}
				BlasMath.daxpym(n, diff, w, 1, gopt, 1);

				// Update XOPT, GOPT and KOPT if the new calculated F is less than
				// FOPT.
				if (f < fopt) {
					kopt[0] = knew[0];
					ih = 0;
					System.arraycopy(xnew, 0, xopt, 0, n);
					xoptsq = BlasMath.ddotm(n, xopt, 1, xopt, 1);
					for (j = 1; j <= n; j++) {
						for (i = 1; i <= j; i++) {
							++ih;
							if (i < j) {
								gopt[j - 1] += hq[ih - 1] * d[i - 1];
							}
							gopt[i - 1] += hq[ih - 1] * d[j - 1];
						}
					}
					for (k = 1; k <= npt; k++) {
						temp = BlasMath.ddotm(n, xpt[k - 1], 1, d, 1);
						temp *= pq[k - 1];
						BlasMath.daxpym(n, temp, xpt[k - 1], 1, gopt, 1);
					}
				}

				// Calculate the parameters of the least Frobenius norm interpolant
				// to
				// the current data, the gradient of this interpolant at XOPT being
				// put
				// into VLAG(NPT+I), I=1,2,...,N.
				if (ntrits > 0) {

					for (k = 1; k <= npt; k++) {
						vlag[k - 1] = fval[k - 1] - fval[kopt[0] - 1];
						w[k - 1] = zero;
					}
					for (j = 1; j <= nptm; j++) {
						sum = zero;
						for (k = 1; k <= npt; k++) {
							sum += zmat[k - 1][j - 1] * vlag[k - 1];
						}
						for (k = 1; k <= npt; k++) {
							w[k - 1] += sum * zmat[k - 1][j - 1];
						}
					}
					for (k = 1; k <= npt; k++) {
						sum = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
						w[k + npt - 1] = w[k - 1];
						w[k - 1] *= sum;
					}
					gqsq = gisq = zero;
					for (i = 1; i <= n; i++) {
						sum = zero;
						for (k = 1; k <= npt; k++) {
							sum += bmat[k - 1][i - 1] * vlag[k - 1] + xpt[k - 1][i - 1] * w[k - 1];
						}
						if (xopt[i - 1] == sl[i - 1]) {
							temp = Math.min(zero, gopt[i - 1]);
							gqsq += (temp * temp);
							temp = Math.min(zero, sum);
							gisq += (temp * temp);
						} else if (xopt[i - 1] == su[i - 1]) {
							temp = Math.max(zero, gopt[i - 1]);
							gqsq += (temp * temp);
							temp = Math.max(zero, sum);
							gisq += (temp * temp);
						} else {
							gqsq += gopt[i - 1] * gopt[i - 1];
							gisq += sum * sum;
						}
						vlag[npt + i - 1] = sum;
					}

					// Test whether to replace the new quadratic model by the least
					// Frobenius
					// norm interpolant, making the replacement if the test is
					// satisfied.
					++itest;
					if (gqsq < ten * gisq) {
						itest = 0;
					}
					if (itest >= 3) {
						for (i = 1; i <= Math.max(npt, nh); ++i) {
							if (i <= n) {
								gopt[i - 1] = vlag[npt + i - 1];
							}
							if (i <= npt) {
								pq[i - 1] = w[npt + i - 1];
							}
							if (i <= nh) {
								hq[i - 1] = zero;
							}
							itest = 0;
						}
					}
				}

				// If a trust region step has provided a sufficient decrease in F,
				// then
				// branch for another trust region calculation. The case NTRITS=0
				// occurs
				// when the new interpolation point was reached by an alternative
				// step.
				if (ntrits == 0 || f <= fopt + tenth * vquad) {
					gotoflag = 60;
					continue;
				} else {

					// Alternatively, find out if the interpolation points are close
					// enough
					// to the best point so far.
					distsq = Math.max((two * delta) * (two * delta), (ten * rho) * (ten * rho));
					gotoflag = 650;
				}
			}

			if (gotoflag == 650) {

				knew[0] = 0;
				for (k = 1; k <= npt; ++k) {
					sum = zero;
					for (j = 1; j <= n; ++j) {
						sum += (xpt[k - 1][j - 1] - xopt[j - 1]) * (xpt[k - 1][j - 1] - xopt[j - 1]);
					}
					if (sum > distsq) {
						knew[0] = k;
						distsq = sum;
					}
				}

				// If KNEW is positive, then ALTMOV finds alternative new positions
				// for
				// the KNEW-th interpolation point within distance ADELT of XOPT...
				if (knew[0] > 0) {
					dist = Math.sqrt(distsq);
					if (ntrits == -1) {
						delta = Math.min(tenth * delta, half * dist);
						if (delta <= 1.5 * rho) {
							delta = rho;
						}
					}
					ntrits = 0;
					adelt = Math.max(Math.min(tenth * dist, delta), rho);
					dsq[0] = adelt * adelt;
					gotoflag = 90;
					continue;
				} else if (ntrits == -1) {
					gotoflag = 680;
				} else if (ratio > zero || Math.max(delta, dnorm) > rho) {
					gotoflag = 60;
					continue;
				} else {
					gotoflag = 680;
				}
			}

			if (gotoflag == 680) {

				// The calculations with the current value of RHO are complete. Pick
				// the
				// next values of RHO and DELTA.
				if (rho > rhoend) {
					delta = half * rho;
					ratio = rho / rhoend;
					if (ratio <= 16.0) {
						rho = rhoend;
					} else if (ratio <= 250.0) {
						rho = Math.sqrt(ratio) * rhoend;
					} else {
						rho *= tenth;
					}
					delta = Math.max(delta, rho);
					ntrits = 0;
					nfsav = nf[0];
					gotoflag = 60;
					continue;
				}

				// Return from the calculation, after another Newton-Raphson step, if
				// it is too short to have been tried before.
				if (ntrits == -1) {
					gotoflag = 360;
				} else {
					break;
				}
			}
		}

		if (fval[kopt[0] - 1] <= fsave) {
			for (i = 1; i <= n; ++i) {
				x[i - 1] = Math.min(Math.max(xl[i - 1], xbase[i - 1] + xopt[i - 1]), xu[i - 1]);
				if (xopt[i - 1] == sl[i - 1]) {
					x[i - 1] = xl[i - 1];
				}
				if (xopt[i - 1] == su[i - 1]) {
					x[i - 1] = xu[i - 1];
				}
			}
			// f = fval[kopt[0] - 1];
		}
	}

	private static void rescue(final Function<? super double[], Double> func, final int n, final int npt,
			final double[] xl, final double[] xu, final int iprint, final int maxfun, final double[] xbase,
			final double[][] xpt, final double[] fval, final double[] xopt, final double[] gopt, final double[] hq,
			final double[] pq, final double[][] bmat, final double[][] zmat, final int ndim, final double[] sl,
			final double[] su, final int[] nf, final double delta, final int[] kopt, final double[] vlag,
			final double[][] ptsaux, final double[] ptsid, final int ipt, final double[] w, final int iiw) {

		int i, ih, iw, ip, iq, ihq, ihp = 0, j, jp, jpn, k, kpt, kold, knew, nrem;
		double sumpq, winc, distsq, temp, fbase, dsqmin, sum, bsum, vlmxsq, hdiag, den, xp = 0.0, xq = 0.0, vquad, f,
				diff;
		final double[] beta = new double[1], denom = new double[1], w1 = new double[n];

		// Set some constants.
		final double half = 0.5, one = 1.0, zero = 0.0;
		final int np = n + 1;
		final double sfrac = half / np;
		final int nptm = npt - np;

		// Shift the interpolation points so that XOPT becomes the origin, and set
		// the elements of ZMAT to zero.
		sumpq = winc = zero;
		for (k = 1; k <= npt; k++) {
			BlasMath.daxpym(n, -1.0, xopt, 1, xpt[k - 1], 1);
			distsq = BlasMath.ddotm(n, xpt[k - 1], 1, xpt[k - 1], 1);
			sumpq += pq[k - 1];
			w[ndim + k - 1 + iiw - 1] = distsq;
			winc = Math.max(winc, distsq);
			Arrays.fill(zmat[k - 1], 0, nptm, zero);
		}

		// Update HQ so that HQ and PQ define the second derivatives of the model
		// after XBASE has been shifted to the trust region centre.
		ih = 0;
		for (j = 1; j <= n; j++) {
			w[j - 1 + iiw - 1] = half * sumpq * xopt[j - 1];
			for (k = 1; k <= npt; k++) {
				w[j - 1 + iiw - 1] += pq[k - 1] * xpt[k - 1][j - 1];
			}
			for (i = 1; i <= j; i++) {
				++ih;
				hq[ih - 1] += w[i - 1 + iiw - 1] * xopt[j - 1] + w[j - 1 + iiw - 1] * xopt[i - 1];
			}
		}

		// Shift XBASE, SL, SU and XOPT. Set the elements of BMAT to zero, and
		// also set the elements of PTSAUX.
		for (j = 1; j <= n; j++) {
			xbase[j - 1] += xopt[j - 1];
			sl[j - 1] -= xopt[j - 1];
			su[j - 1] -= xopt[j - 1];
			xopt[j - 1] = zero;
			ptsaux[1 - 1][j - 1] = Math.min(delta, su[j - 1]);
			ptsaux[2 - 1][j - 1] = Math.max(-delta, sl[j - 1]);
			if (ptsaux[1 - 1][j - 1] + ptsaux[2 - 1][j - 1] < zero) {
				temp = ptsaux[1 - 1][j - 1];
				ptsaux[1 - 1][j - 1] = ptsaux[2 - 1][j - 1];
				ptsaux[2 - 1][j - 1] = temp;
			}
			if (Math.abs(ptsaux[2 - 1][j - 1]) < half * Math.abs(ptsaux[1 - 1][j - 1])) {
				ptsaux[2 - 1][j - 1] = half * ptsaux[1 - 1][j - 1];
			}
			for (i = 1; i <= ndim; i++) {
				bmat[i - 1][j - 1] = zero;
			}
		}
		fbase = fval[kopt[0] - 1];

		// Set the identifiers of the artificial interpolation points that are
		// along a coordinate direction from XOPT, and set the corresponding
		// nonzero elements of BMAT and ZMAT.
		ptsid[1 - 1 + ipt - 1] = sfrac;
		for (j = 1; j <= n; j++) {
			jp = j + 1;
			jpn = jp + n;
			ptsid[jp - 1 + ipt - 1] = (double) j + sfrac;
			if (jpn <= npt) {
				ptsid[jpn - 1 + ipt - 1] = (double) j / np + sfrac;
				temp = one / (ptsaux[1 - 1][j - 1] - ptsaux[2 - 1][j - 1]);
				bmat[jp - 1][j - 1] = -temp + one / ptsaux[1 - 1][j - 1];
				bmat[jpn - 1][j - 1] = temp + one / ptsaux[2 - 1][j - 1];
				bmat[1 - 1][j - 1] = -bmat[jp - 1][j - 1] - bmat[jpn - 1][j - 1];
				zmat[1 - 1][j - 1] = Math.sqrt(2.0) / Math.abs(ptsaux[1 - 1][j - 1] * ptsaux[2 - 1][j - 1]);
				zmat[jp - 1][j - 1] = zmat[1 - 1][j - 1] * ptsaux[2 - 1][j - 1] * temp;
				zmat[jpn - 1][j - 1] = -zmat[1 - 1][j - 1] * ptsaux[1 - 1][j - 1] * temp;
			} else {
				bmat[1 - 1][j - 1] = -one / ptsaux[1 - 1][j - 1];
				bmat[jp - 1][j - 1] = one / ptsaux[1 - 1][j - 1];
				bmat[j + npt - 1][j - 1] = -half * ptsaux[1 - 1][j - 1] * ptsaux[1 - 1][j - 1];
			}
		}

		// Set any remaining identifiers with their nonzero elements of ZMAT.
		if (npt >= n + np) {
			for (k = 2 * np; k <= npt; ++k) {
				iw = (int) (((double) (k - np) - half) / n);
				ip = k - np - iw * n;
				iq = ip + iw;
				if (iq > n) {
					iq -= n;
				}
				ptsid[k - 1 + ipt - 1] = ip + ((double) iq) / np + sfrac;
				temp = one / (ptsaux[1 - 1][ip - 1] * ptsaux[1 - 1][iq - 1]);
				zmat[1 - 1][k - np - 1] = temp;
				zmat[ip + 1 - 1][k - np - 1] = -temp;
				zmat[iq + 1 - 1][k - np - 1] = -temp;
				zmat[k - 1][k - np - 1] = temp;
			}
		}
		nrem = npt;
		kold = 1;
		knew = kopt[0];

		boolean goto80 = true;
		while (true) {

			if (goto80) {

				// Reorder the provisional points in the way that exchanges
				// PTSID(KOLD)
				// with PTSID(KNEW).
				double[] tmp = Arrays.copyOf(bmat[kold - 1], n);
				System.arraycopy(bmat[knew - 1], 0, bmat[kold - 1], 0, n);
				System.arraycopy(tmp, 0, bmat[knew - 1], 0, n);
				tmp = Arrays.copyOf(zmat[kold - 1], nptm);
				System.arraycopy(zmat[knew - 1], 0, zmat[kold - 1], 0, nptm);
				System.arraycopy(tmp, 0, zmat[knew - 1], 0, nptm);
				ptsid[kold - 1 + ipt - 1] = ptsid[knew - 1 + ipt - 1];
				ptsid[knew - 1 + ipt - 1] = zero;
				w[ndim + knew - 1 + iiw - 1] = zero;
				--nrem;
				if (knew != kopt[0]) {
					temp = vlag[kold - 1];
					vlag[kold - 1] = vlag[knew - 1];
					vlag[knew - 1] = temp;

					// Update the BMAT and ZMAT matrices so that the status of the
					// KNEW-th
					// interpolation point can be changed from provisional to
					// original.
					update(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w, iiw);
					if (nrem == 0) {
						return;
					}
					for (k = 1; k <= npt; ++k) {
						w[ndim + k - 1 + iiw - 1] = Math.abs(w[ndim + k - 1 + iiw - 1]);
					}
				}
			}

			// Pick the index KNEW of an original interpolation point that has not
			// yet replaced one of the provisional interpolation points, giving
			// attention to the closeness to XOPT and to previous tries with KNEW.
			dsqmin = zero;
			for (k = 1; k <= npt; k++) {
				if (w[ndim + k - 1 + iiw - 1] > zero) {
					if (dsqmin == zero || w[ndim + k - 1 + iiw - 1] < dsqmin) {
						knew = k;
						dsqmin = w[ndim + k - 1 + iiw - 1];
					}
				}
			}
			if (dsqmin == zero) {
				break;
			}

			// Form the W-vector of the chosen original interpolation point.
			System.arraycopy(xpt[knew - 1], 0, w, npt + iiw - 1, n);
			for (k = 1; k <= npt; k++) {
				sum = zero;
				if (k == kopt[0]) {
				} else if (ptsid[k - 1 + ipt - 1] == zero) {
					sum += BlasMath.ddotm(n, w, npt + iiw, xpt[k - 1], 1);
				} else {
					ip = (int) ptsid[k - 1 + ipt - 1];
					if (ip > 0) {
						sum = w[npt + ip - 1 + iiw - 1] * ptsaux[1 - 1][ip - 1];
					}
					iq = (int) ((double) np * ptsid[k - 1 + ipt - 1] - (ip * np));
					if (iq > 0) {
						iw = 1;
						if (ip == 0) {
							iw = 2;
						}
						sum += w[npt + iq - 1 + iiw - 1] * ptsaux[iw - 1][iq - 1];
					}
				}
				w[k - 1 + iiw - 1] = half * sum * sum;
			}

			// Calculate VLAG and BETA for the required updating of the H matrix if
			// XPT(KNEW,.) is reinstated in the set of interpolation points.
			for (k = 1; k <= npt; k++) {
				sum = BlasMath.ddotm(n, bmat[k - 1], 1, w, npt + iiw);
				vlag[k - 1] = sum;
			}
			beta[0] = zero;
			for (j = 1; j <= nptm; j++) {
				sum = zero;
				for (k = 1; k <= npt; k++) {
					sum += zmat[k - 1][j - 1] * w[k - 1 + iiw - 1];
				}
				beta[0] -= sum * sum;
				for (k = 1; k <= npt; k++) {
					vlag[k - 1] += sum * zmat[k - 1][j - 1];
				}
			}
			bsum = distsq = zero;
			for (j = 1; j <= n; j++) {
				sum = zero;
				for (k = 1; k <= npt; k++) {
					sum += bmat[k - 1][j - 1] * w[k - 1 + iiw - 1];
				}
				jp = j + npt;
				bsum += sum * w[jp - 1 + iiw - 1];
				for (ip = npt + 1; ip <= ndim; ++ip) {
					sum += bmat[ip - 1][j - 1] * w[ip - 1 + iiw - 1];
				}
				bsum += sum * w[jp - 1 + iiw - 1];
				vlag[jp - 1] = sum;
				distsq += xpt[knew - 1][j - 1] * xpt[knew - 1][j - 1];
			}
			beta[0] = half * distsq * distsq + beta[0] - bsum;
			vlag[kopt[0] - 1] += one;

			// KOLD is set to the index of the provisional interpolation point that
			// is
			// going to be deleted to make way for the KNEW-th original interpolation
			// point. The choice of KOLD is governed by the avoidance of a small
			// value
			// of the denominator in the updating calculation of UPDATE.
			denom[0] = vlmxsq = zero;
			for (k = 1; k <= npt; k++) {
				if (ptsid[k - 1 + ipt - 1] != zero) {
					hdiag = BlasMath.ddotm(nptm, zmat[k - 1], 1, zmat[k - 1], 1);
					den = beta[0] * hdiag + vlag[k - 1] * vlag[k - 1];
					if (den > denom[0]) {
						kold = k;
						denom[0] = den;
					}
				}
				vlmxsq = Math.max(vlmxsq, vlag[k - 1] * vlag[k - 1]);
			}
			if (denom[0] <= 1.0e-2 * vlmxsq) {
				w[ndim + knew - 1 + iiw - 1] = -w[ndim + knew - 1 + iiw - 1] - winc;
				goto80 = false;
			} else {
				goto80 = true;
			}
		}

		// When label 260 is reached, all the final positions of the interpolation
		// points have been chosen although any changes have not been included yet
		// in XPT...
		for (kpt = 1; kpt <= npt; ++kpt) {
			if (ptsid[kpt - 1 + ipt - 1] == zero) {
				continue;
			}
			if (nf[0] >= maxfun) {
				nf[0] = -1;
				return;
			}
			ih = 0;
			for (j = 1; j <= n; j++) {
				w[j - 1 + iiw - 1] = xpt[kpt - 1][j - 1];
				xpt[kpt - 1][j - 1] = zero;
				temp = pq[kpt - 1] * w[j - 1 + iiw - 1];
				for (i = 1; i <= j; i++) {
					++ih;
					hq[ih - 1] += temp * w[i - 1 + iiw - 1];
				}
			}
			pq[kpt - 1] = zero;
			ip = (int) ptsid[kpt - 1 + ipt - 1];
			iq = (int) ((double) np * ptsid[kpt - 1 + ipt - 1] - (double) (ip * np));
			if (ip > 0) {
				xp = ptsaux[1 - 1][ip - 1];
				xpt[kpt - 1][ip - 1] = xp;
			}
			if (iq > 0) {
				xq = ptsaux[1 - 1][iq - 1];
				if (ip == 0) {
					xq = ptsaux[2 - 1][iq - 1];
				}
				xpt[kpt - 1][iq - 1] = xq;
			}

			// Set VQUAD to the value of the current model at the new point.
			vquad = fbase;
			if (ip > 0) {
				ihp = (ip + ip * ip) / 2;
				vquad += xp * (gopt[ip - 1] + half * xp * hq[ihp - 1]);
			}
			if (iq > 0) {
				ihq = (iq + iq * iq) / 2;
				vquad += xq * (gopt[iq - 1] + half * xq * hq[ihq - 1]);
				if (ip > 0) {
					iw = Math.max(ihp, ihq) - IntMath.abs(ip - iq);
					vquad += xp * xq * hq[iw - 1];
				}
			}
			for (k = 1; k <= npt; k++) {
				temp = zero;
				if (ip > 0) {
					temp += xp * xpt[k - 1][ip - 1];
				}
				if (iq > 0) {
					temp += xq * xpt[k - 1][iq - 1];
				}
				vquad += half * pq[k - 1] * temp * temp;
			}

			// Calculate F at the new interpolation point, and set DIFF to the factor
			// that is going to multiply the KPT-th Lagrange function when the model
			// is updated to provide interpolation to the new function value.
			for (i = 1; i <= n; i++) {
				final double temp1 = xbase[i - 1] + xpt[kpt - 1][i - 1];
				w[i - 1 + iiw - 1] = Math.min(Math.max(xl[i - 1], temp1), xu[i - 1]);
				if (xpt[kpt - 1][i - 1] == sl[i - 1]) {
					w[i - 1 + iiw - 1] = xl[i - 1];
				}
				if (xpt[kpt - 1][i - 1] == su[i - 1]) {
					w[i - 1 + iiw - 1] = xu[i - 1];
				}
			}
			++nf[0];
			System.arraycopy(w, iiw - 1, w1, 0, n);
			f = func.apply(w1);
			fval[kpt - 1] = f;
			if (f < fval[kopt[0] - 1]) {
				kopt[0] = kpt;
			}
			diff = f - vquad;

			// Update the quadratic model. The RETURN from the subroutine occurs when
			// all the new interpolation points are included in the model.
			BlasMath.daxpym(n, diff, bmat[kpt - 1], 1, gopt, 1);
			for (k = 1; k <= npt; k++) {
				sum = BlasMath.ddotm(nptm, zmat[k - 1], 1, zmat[kpt - 1], 1);
				temp = diff * sum;
				if (ptsid[k - 1 + ipt - 1] == zero) {
					pq[k - 1] += temp;
				} else {
					ip = (int) ptsid[k - 1 + ipt - 1];
					iq = (int) ((double) np * ptsid[k - 1 + ipt - 1] - (double) (ip * np));
					ihq = (iq * iq + iq) / 2;
					if (ip == 0) {
						hq[ihq - 1] += temp * ptsaux[2 - 1][iq - 1] * ptsaux[2 - 1][iq - 1];
					} else {
						ihp = (ip * ip + ip) / 2;
						hq[ihp - 1] += temp * ptsaux[1 - 1][ip - 1] * ptsaux[1 - 1][ip - 1];
						if (iq > 0) {
							hq[ihq - 1] += temp * ptsaux[1 - 1][iq - 1] * ptsaux[1 - 1][iq - 1];
							iw = Math.max(ihp, ihq) - IntMath.abs(iq - ip);
							hq[iw - 1] += temp * ptsaux[1 - 1][ip - 1] * ptsaux[1 - 1][iq - 1];
						}
					}
				}
			}
			ptsid[kpt - 1 + ipt - 1] = zero;
		}
	}

	private static void trsbox(final int n, final int npt, final double[][] xpt, final double[] xopt,
			final double[] gopt, final double[] hq, final double[] pq, final double[] sl, final double[] su,
			final double delta, final double[] xnew, final double[] d, final double[] gnew, final int ignew,
			final double[] xbdi, final int ixbdi, final double[] s, final int is, final double[] hs, final int ihs,
			final double[] hred, final int ihred, final double[] dsq, final double[] crvmin) {

		int i, ih, iu, isav, iact = 0, iterc, itcsav = 0, itermax = 0, nact, k, j;
		double sqstp, delsq, qred, beta, stepsq = 0.0, gredsq = 0.0, resid, ds, shs, temp, blen, stplen, xsum, sdec,
				ggsav = 0.0, dredsq = 0.0, dredg = 0.0, sredg = 0.0, tempa, tempb, angbd = 0.0, ratio, ssq, xsav = 0.0,
				dhs, dhd, redmax, redsav, angt = 0.0, sth, rednew, rdprev = 0.0, rdnext = 0.0, cth;

		// Set some constants.
		final double half = 0.5, one = 1.0, onemin = -1.0, zero = 0.0;

		// The sign of GOPT(I) gives the sign of the change to the I-th variable
		// that will reduce Q from its value at XOPT...
		iterc = nact = 0;
		sqstp = zero;
		for (i = 1; i <= n; i++) {
			xbdi[i - 1 + ixbdi - 1] = zero;
			if (xopt[i - 1] <= sl[i - 1]) {
				if (gopt[i - 1] >= zero) {
					xbdi[i - 1 + ixbdi - 1] = onemin;
				}
			} else if (xopt[i - 1] >= su[i - 1]) {
				if (gopt[i - 1] <= zero) {
					xbdi[i - 1 + ixbdi - 1] = one;
				}
			}
			if (xbdi[i - 1 + ixbdi - 1] != zero) {
				++nact;
			}
			d[i - 1] = zero;
		}
		System.arraycopy(gopt, 0, gnew, ignew - 1, n);
		delsq = delta * delta;
		qred = zero;
		crvmin[0] = onemin;
		beta = zero;

		int gotoflag = 30;
		while (true) {

			if (gotoflag == 30) {

				// Set the next search direction of the conjugate gradient method...
				stepsq = zero;
				for (i = 1; i <= n; i++) {
					if (xbdi[i - 1 + ixbdi - 1] != zero) {
						s[i - 1 + is - 1] = zero;
					} else if (beta == zero) {
						s[i - 1 + is - 1] = -gnew[i - 1 + ignew - 1];
					} else {
						s[i - 1 + is - 1] = beta * s[i - 1 + is - 1] - gnew[i - 1 + ignew - 1];
					}
					stepsq += s[i - 1 + is - 1] * s[i - 1 + is - 1];
				}
				if (stepsq == zero) {
					gotoflag = 190;
				} else {
					if (beta == zero) {
						gredsq = stepsq;
						itermax = iterc + n - nact;
					}
					if (gredsq * delsq <= 1.0e-4 * qred * qred) {
						gotoflag = 190;
					} else {
						gotoflag = 210;
					}
				}
			}

			if (gotoflag == 50) {

				// Multiply the search direction by the second derivative matrix of Q
				// and
				// calculate some scalars for the choice of steplength. Then set BLEN
				// to
				// the length of the the step to the trust region boundary and STPLEN
				// to
				// the steplength, ignoring the simple bounds.
				resid = delsq;
				ds = shs = zero;
				for (i = 1; i <= n; i++) {
					if (xbdi[i - 1 + ixbdi - 1] == zero) {
						resid -= d[i - 1] * d[i - 1];
						ds += s[i - 1 + is - 1] * d[i - 1];
						shs += s[i - 1 + is - 1] * hs[i - 1 + ihs - 1];
					}
				}
				if (resid <= zero) {
					crvmin[0] = zero;
					gotoflag = 100;
				} else {

					temp = Math.sqrt(stepsq * resid + ds * ds);
					if (ds < zero) {
						blen = (temp - ds) / stepsq;
					} else {
						blen = resid / (temp + ds);
					}
					stplen = blen;
					if (shs > zero) {
						stplen = Math.min(blen, gredsq / shs);
					}

					// Reduce STPLEN if necessary in order to preserve the simple
					// bounds,
					// letting IACT be the index of the new constrained variable.
					iact = 0;
					for (i = 1; i <= n; i++) {
						if (s[i - 1 + is - 1] != zero) {
							xsum = xopt[i - 1] + d[i - 1];
							if (s[i - 1 + is - 1] > zero) {
								temp = (su[i - 1] - xsum) / s[i - 1 + is - 1];
							} else {
								temp = (sl[i - 1] - xsum) / s[i - 1 + is - 1];
							}
							if (temp < stplen) {
								stplen = temp;
								iact = i;
							}
						}
					}

					// Update CRVMIN, GNEW and D. Set SDEC to the decrease that
					// occurs in Q.
					sdec = zero;
					if (stplen > zero) {
						++iterc;
						temp = shs / stepsq;
						if (iact == 0 && temp > zero) {
							crvmin[0] = Math.min(crvmin[0], temp);
							if (crvmin[0] == onemin) {
								crvmin[0] = temp;
							}
						}
						ggsav = gredsq;
						gredsq = zero;
						for (i = 1; i <= n; i++) {
							gnew[i - 1 + ignew - 1] += stplen * hs[i - 1 + ihs - 1];
							if (xbdi[i - 1 + ixbdi - 1] == zero) {
								gredsq += gnew[i - 1 + ignew - 1] * gnew[i - 1 + ignew - 1];
							}
							d[i - 1] += stplen * s[i - 1 + is - 1];
						}
						sdec = Math.max(stplen * (ggsav - half * stplen * shs), zero);
						qred += sdec;
					}

					// Restart the conjugate gradient method if it has hit a new
					// bound.
					if (iact > 0) {
						++nact;
						xbdi[iact - 1 + ixbdi - 1] = one;
						if (s[iact - 1 + is - 1] < zero) {
							xbdi[iact - 1 + ixbdi - 1] = onemin;
						}
						delsq -= d[iact - 1] * d[iact - 1];
						if (delsq <= zero) {
							crvmin[0] = zero;
							gotoflag = 100;
						} else {
							beta = zero;
							gotoflag = 30;
							continue;
						}
					} else if (stplen < blen) {

						// If STPLEN is less than BLEN, then either apply another
						// conjugate
						// gradient iteration or RETURN.
						if (iterc == itermax || sdec <= 0.01 * qred) {
							break;
						} else {
							beta = gredsq / ggsav;
							gotoflag = 30;
							continue;
						}
					} else {
						crvmin[0] = zero;
						gotoflag = 100;
					}
				}
			}

			if (gotoflag == 100) {

				// Prepare for the alternative iteration by calculating some scalars
				// and
				// by multiplying the reduced D by the second derivative matrix of Q.
				if (nact >= n - 1) {
					break;
				} else {
					dredsq = dredg = gredsq = zero;
					for (i = 1; i <= n; i++) {
						if (xbdi[i - 1 + ixbdi - 1] == zero) {
							dredsq += d[i - 1] * d[i - 1];
							dredg += d[i - 1] * gnew[i - 1 + ignew - 1];
							gredsq += gnew[i - 1 + ignew - 1] * gnew[i - 1 + ignew - 1];
							s[i - 1 + is - 1] = d[i - 1];
						} else {
							s[i - 1 + is - 1] = zero;
						}
					}
					itcsav = iterc;
					gotoflag = 210;
				}
			}

			if (gotoflag == 120) {

				// Let the search direction S be a linear combination of the reduced
				// D
				// and the reduced G that is orthogonal to the reduced D.
				++iterc;
				temp = gredsq * dredsq - dredg * dredg;
				if (temp <= 1.0e-4 * qred * qred) {
					break;
				}
				temp = Math.sqrt(temp);
				for (i = 1; i <= n; i++) {
					if (xbdi[i - 1 + ixbdi - 1] == zero) {
						s[i - 1 + is - 1] = (dredg * d[i - 1] - dredsq * gnew[i - 1 + ignew - 1]) / temp;
					} else {
						s[i - 1 + is - 1] = zero;
					}
				}
				sredg = -temp;

				// By considering the simple bounds on the variables, calculate an
				// upper
				// bound on the tangent of half the angle of the alternative
				// iteration,
				// namely ANGBD, except that, if already a free variable has reached
				// a
				// bound, there is a branch back to label 100 after fixing that
				// variable.
				angbd = one;
				iact = 0;
				boolean skipto100 = false;
				for (i = 1; i <= n; i++) {
					if (xbdi[i - 1 + ixbdi - 1] == zero) {
						tempa = xopt[i - 1] + d[i - 1] - sl[i - 1];
						tempb = su[i - 1] - xopt[i - 1] - d[i - 1];
						if (tempa <= zero) {
							++nact;
							xbdi[i - 1 + ixbdi - 1] = onemin;
							skipto100 = true;
							break;
						} else if (tempb <= zero) {
							++nact;
							xbdi[i - 1 + ixbdi - 1] = one;
							skipto100 = true;
							break;
						}
						ratio = one;
						ssq = d[i - 1] * d[i - 1] + s[i - 1 + is - 1] * s[i - 1 + is - 1];
						temp = ssq - (xopt[i - 1] - sl[i - 1]) * (xopt[i - 1] - sl[i - 1]);
						if (temp > zero) {
							temp = Math.sqrt(temp) - s[i - 1 + is - 1];
							if (angbd * temp > tempa) {
								angbd = tempa / temp;
								iact = i;
								xsav = onemin;
							}
						}
						temp = ssq - (su[i - 1] - xopt[i - 1]) * (su[i - 1] - xopt[i - 1]);
						if (temp > zero) {
							temp = Math.sqrt(temp) + s[i - 1 + is - 1];
							if (angbd * temp > tempb) {
								angbd = tempb / temp;
								iact = i;
								xsav = one;
							}
						}
					}
				}
				if (skipto100) {
					gotoflag = 100;
					continue;
				} else {
					gotoflag = 210;
				}
			}

			if (gotoflag == 150) {

				// Calculate HHD and some curvatures for the alternative iteration.
				shs = dhs = dhd = zero;
				for (i = 1; i <= n; i++) {
					if (xbdi[i - 1 + ixbdi - 1] == zero) {
						shs += s[i - 1 + is - 1] * hs[i - 1 + ihs - 1];
						dhs += d[i - 1] * hs[i - 1 + ihs - 1];
						dhd += d[i - 1] * hred[i - 1 + ihred - 1];
					}
				}

				// Seek the greatest reduction in Q for a range of equally spaced
				// values
				// of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle
				// of
				// the alternative iteration.
				redmax = zero;
				isav = 0;
				redsav = zero;
				iu = (int) (17.0 * angbd + 3.1);
				for (i = 1; i <= iu; i++) {
					angt = angbd * i / iu;
					sth = (angt + angt) / (one + angt * angt);
					temp = shs + angt * (angt * dhd - dhs - dhs);
					rednew = sth * (angt * dredg - sredg - half * sth * temp);
					if (rednew > redmax) {
						redmax = rednew;
						isav = i;
						rdprev = redsav;
					} else if (i == isav + 1) {
						rdnext = rednew;
					}
					redsav = rednew;
				}

				// Return if the reduction is zero. Otherwise, set the sine and
				// cosine
				// of the angle of the alternative iteration, and calculate SDEC.
				if (isav == 0) {
					break;
				}
				if (isav < iu) {
					temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
					angt = angbd * (isav + half * temp) / iu;
				}
				cth = (one - angt * angt) / (one + angt * angt);
				sth = (angt + angt) / (one + angt * angt);
				temp = shs + angt * (angt * dhd - dhs - dhs);
				sdec = sth * (angt * dredg - sredg - half * sth * temp);
				if (sdec <= zero) {
					break;
				}

				// Update GNEW, D and HRED. If the angle of the alternative iteration
				// is restricted by a bound on a free variable, that variable is
				// fixed
				// at the bound.
				dredg = gredsq = zero;
				for (i = 1; i <= n; i++) {
					gnew[i - 1 + ignew - 1] += (cth - one) * hred[i - 1 + ihred - 1] + sth * hs[i - 1 + ihs - 1];
					if (xbdi[i - 1 + ixbdi - 1] == zero) {
						d[i - 1] = cth * d[i - 1] + sth * s[i - 1 + is - 1];
						dredg += d[i - 1] * gnew[i - 1 + ignew - 1];
						gredsq += gnew[i - 1 + ignew - 1] * gnew[i - 1 + ignew - 1];
					}
					hred[i - 1 + ihred - 1] = cth * hred[i - 1 + ihred - 1] + sth * hs[i - 1 + ihs - 1];
				}
				qred += sdec;

				// If SDEC is sufficiently small, then RETURN after setting XNEW to
				// XOPT+D, giving careful attention to the bounds.
				if (iact > 0 && isav == iu) {
					++nact;
					xbdi[iact - 1 + ixbdi - 1] = xsav;
					gotoflag = 100;
					continue;
				} else if (sdec > 0.01 * qred) {
					gotoflag = 120;
					continue;
				} else {
					break;
				}
			}

			if (gotoflag == 210) {

				// The following instructions multiply the current S-vector by the
				// second
				// derivative matrix of the quadratic model, putting the product in
				// HS.
				ih = 0;
				for (j = 1; j <= n; j++) {
					hs[j - 1 + ihs - 1] = zero;
					for (i = 1; i <= j; i++) {
						++ih;
						if (i < j) {
							hs[j - 1 + ihs - 1] += hq[ih - 1] * s[i - 1 + is - 1];
						}
						hs[i - 1 + ihs - 1] += hq[ih - 1] * s[j - 1 + is - 1];
					}
				}
				for (k = 1; k <= npt; k++) {
					if (pq[k - 1] != zero) {
						temp = BlasMath.ddotm(n, xpt[k - 1], 1, s, is);
						temp *= pq[k - 1];
						BlasMath.daxpym(n, temp, xpt[k - 1], 1, hs, ihs);
					}
				}
				if (crvmin[0] != zero) {
					gotoflag = 50;
				} else if (iterc > itcsav) {
					gotoflag = 150;
				} else {
					System.arraycopy(hs, ihs - 1, hred, ihred - 1, n);
					gotoflag = 120;
				}
			}
		}

		dsq[0] = zero;
		for (i = 1; i <= n; i++) {
			xnew[i - 1] = Math.max(Math.min(xopt[i - 1] + d[i - 1], su[i - 1]), sl[i - 1]);
			if (xbdi[i - 1 + ixbdi - 1] == onemin) {
				xnew[i - 1] = sl[i - 1];
			}
			if (xbdi[i - 1 + ixbdi - 1] == one) {
				xnew[i - 1] = su[i - 1];
			}
			d[i - 1] = xnew[i - 1] - xopt[i - 1];
			dsq[0] += d[i - 1] * d[i - 1];
		}
	}

	private static void altmov(final int n, final int npt, final double[][] xpt, final double[] xopt,
			final double[][] bmat, final double[][] zmat, final int ndim, final double[] sl, final double[] su,
			final int[] kopt, final int[] knew, final double adelt, final double[] xnew, final double[] xalt,
			final double[] alpha, final double[] cauchy, final double[] glag, final int iglag, final double[] hcol,
			final int ihcol, final double[] w, final int iw) {

		final double half = 0.5, one = 1.0, zero = 0.0, cons = one + Math.sqrt(2.0);
		int i, ilbd, iubd, isbd, ibdsav = 0, iflag, j, k, ksav = 0;
		double temp, ha, presav, dderiv, distsq, subd, slbd, sumin, diff, step = 0.0, vlag, tempa, tempb, tempd, predsq,
				stpsav = 0.0, bigstp, ggfree, wfixsq, wsqsav, gw, scale, curv, csave = 0.0;

		Arrays.fill(hcol, ihcol - 1, npt + ihcol - 1, zero);
		for (j = 1; j <= npt - n - 1; ++j) {
			temp = zmat[knew[0] - 1][j - 1];
			for (k = 1; k <= npt; k++) {
				hcol[k - 1 + ihcol - 1] += temp * zmat[k - 1][j - 1];
			}
		}
		alpha[0] = hcol[knew[0] - 1 + ihcol - 1];
		ha = half * alpha[0];

		// Calculate the gradient of the KNEW-th Lagrange function at XOPT.
		System.arraycopy(bmat[knew[0] - 1], 0, glag, iglag - 1, n);
		for (k = 1; k <= npt; k++) {
			temp = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
			temp *= hcol[k - 1 + ihcol - 1];
			BlasMath.daxpym(n, temp, xpt[k - 1], 1, glag, iglag);
		}

		// Search for a large denominator along the straight lines through XOPT
		// and another interpolation point.
		presav = zero;
		for (k = 1; k <= npt; ++k) {
			if (k == kopt[0]) {
				continue;
			}
			dderiv = distsq = zero;
			for (i = 1; i <= n; ++i) {
				temp = xpt[k - 1][i - 1] - xopt[i - 1];
				dderiv += glag[i - 1 + iglag - 1] * temp;
				distsq += (temp * temp);
			}
			subd = adelt / Math.sqrt(distsq);
			slbd = -subd;
			ilbd = iubd = 0;
			sumin = Math.min(one, subd);

			// Revise SLBD and SUBD if necessary because of the bounds in SL and SU.
			for (i = 1; i <= n; i++) {
				temp = xpt[k - 1][i - 1] - xopt[i - 1];
				if (temp > zero) {
					if (slbd * temp < sl[i - 1] - xopt[i - 1]) {
						slbd = (sl[i - 1] - xopt[i - 1]) / temp;
						ilbd = -i;
					}
					if (subd * temp > su[i - 1] - xopt[i - 1]) {
						subd = Math.max(sumin, (su[i - 1] - xopt[i - 1]) / temp);
						iubd = i;
					}
				} else if (temp < zero) {
					if (slbd * temp > su[i - 1] - xopt[i - 1]) {
						slbd = (su[i - 1] - xopt[i - 1]) / temp;
						ilbd = i;
					}
					if (subd * temp < sl[i - 1] - xopt[i - 1]) {
						subd = Math.max(sumin, (sl[i - 1] - xopt[i - 1]) / temp);
						iubd = -i;
					}
				}
			}

			// Seek a large modulus of the KNEW-th Lagrange function when the index
			// of the other interpolation point on the line through XOPT is KNEW.
			if (k == knew[0]) {
				diff = dderiv - one;
				step = slbd;
				vlag = slbd * (dderiv - slbd * diff);
				isbd = ilbd;
				temp = subd * (dderiv - subd * diff);
				if (Math.abs(temp) > Math.abs(vlag)) {
					step = subd;
					vlag = temp;
					isbd = iubd;
				}
				tempd = half * dderiv;
				tempa = tempd - diff * slbd;
				tempb = tempd - diff * subd;
				if (tempa * tempb < zero) {
					temp = tempd * tempd / diff;
					if (Math.abs(temp) > Math.abs(vlag)) {
						step = tempd / diff;
						vlag = temp;
						isbd = 0;
					}
				}
			} else {
				step = slbd;
				vlag = slbd * (one - slbd);
				isbd = ilbd;
				temp = subd * (one - subd);
				if (Math.abs(temp) > Math.abs(vlag)) {
					step = subd;
					vlag = temp;
					isbd = iubd;
				}
				if (subd > half) {
					if (Math.abs(vlag) < 0.25) {
						step = half;
						vlag = 0.25;
						isbd = 0;
					}
				}
				vlag *= dderiv;
			}

			// Calculate PREDSQ for the current line search and maintain PRESAV.
			temp = step * (one - step) * distsq;
			predsq = vlag * vlag * (vlag * vlag + ha * temp * temp);
			if (predsq > presav) {
				presav = predsq;
				ksav = k;
				stpsav = step;
				ibdsav = isbd;
			}
		}

		// Construct XNEW in a way that satisfies the bound constraints exactly.
		for (i = 1; i <= n; i++) {
			temp = xopt[i - 1] + stpsav * (xpt[ksav - 1][i - 1] - xopt[i - 1]);
			xnew[i - 1] = Math.max(sl[i - 1], Math.min(su[i - 1], temp));
		}
		if (ibdsav < 0) {
			xnew[-ibdsav - 1] = sl[-ibdsav - 1];
		} else if (ibdsav > 0) {
			xnew[ibdsav - 1] = su[ibdsav - 1];
		}

		// Prepare for the iterative method that assembles the constrained Cauchy
		// step in W. The sum of squares of the fixed components of W is formed in
		// WFIXSQ, and the free components of W are set to BIGSTP.
		bigstp = adelt + adelt;
		iflag = 0;
		while (true) {

			wfixsq = ggfree = zero;
			for (i = 1; i <= n; i++) {
				w[i - 1 + iw - 1] = zero;
				tempa = Math.min(xopt[i - 1] - sl[i - 1], glag[i - 1 + iglag - 1]);
				tempb = Math.max(xopt[i - 1] - su[i - 1], glag[i - 1 + iglag - 1]);
				if (tempa > zero || tempb < zero) {
					w[i - 1 + iw - 1] = bigstp;
					ggfree += glag[i - 1 + iglag - 1] * glag[i - 1 + iglag - 1];
				}
			}
			if (ggfree == zero) {
				cauchy[0] = zero;
				return;
			}

			// Investigate whether more components of W can be fixed.
			while (true) {
				temp = adelt * adelt - wfixsq;
				if (temp > zero) {
					wsqsav = wfixsq;
					step = Math.sqrt(temp / ggfree);
					ggfree = zero;
					for (i = 1; i <= n; i++) {
						if (w[i - 1 + iw - 1] == bigstp) {
							temp = xopt[i - 1] - step * glag[i - 1 + iglag - 1];
							if (temp <= sl[i - 1]) {
								w[i - 1 + iw - 1] = sl[i - 1] - xopt[i - 1];
								wfixsq += w[i - 1 + iw - 1] * w[i - 1 + iw - 1];
							} else if (temp >= su[i - 1]) {
								w[i - 1 + iw - 1] = su[i - 1] - xopt[i - 1];
								wfixsq += w[i - 1 + iw - 1] * w[i - 1 + iw - 1];
							} else {
								ggfree += glag[i - 1 + iglag - 1] * glag[i - 1 + iglag - 1];
							}
						}
					}
					if (!(wfixsq > wsqsav && ggfree > zero)) {
						break;
					}
				} else {
					break;
				}
			}

			// Set the remaining free components of W and all components of XALT,
			// except that W may be scaled later.
			gw = zero;
			for (i = 1; i <= n; i++) {
				if (w[i - 1 + iw - 1] == bigstp) {
					w[i - 1 + iw - 1] = -step * glag[i - 1 + iglag - 1];
					final double min = Math.min(su[i - 1], xopt[i - 1] + w[i - 1 + iw - 1]);
					xalt[i - 1] = Math.max(sl[i - 1], min);
				} else if (w[i - 1 + iw - 1] == zero) {
					xalt[i - 1] = xopt[i - 1];
				} else if (glag[i - 1 + iglag - 1] > zero) {
					xalt[i - 1] = sl[i - 1];
				} else {
					xalt[i - 1] = su[i - 1];
				}
				gw += glag[i - 1 + iglag - 1] * w[i - 1 + iw - 1];
			}

			// Set CURV to the curvature of the KNEW-th Lagrange function along W.
			curv = zero;
			for (k = 1; k <= npt; k++) {
				temp = BlasMath.ddotm(n, xpt[k - 1], 1, w, iw);
				curv += hcol[k - 1 + ihcol - 1] * temp * temp;
			}
			if (iflag == 1) {
				curv = -curv;
			}
			if (curv > -gw && curv < -cons * gw) {
				scale = -gw / curv;
				for (i = 1; i <= n; i++) {
					temp = xopt[i - 1] + scale * w[i - 1 + iw - 1];
					xalt[i - 1] = Math.max(sl[i - 1], Math.min(su[i - 1], temp));
				}
				cauchy[0] = (half * gw * scale) * (half * gw * scale);
			} else {
				cauchy[0] = (gw + half * curv) * (gw + half * curv);
			}

			// If IFLAG is zero, then XALT is calculated as before after reversing
			// the sign of GLAG. Thus two XALT vectors become available. The one that
			// is chosen is the one that gives the larger value of CAUCHY.
			if (iflag == 0) {
				BlasMath.dscalm(n, -1.0, glag, iglag);
				System.arraycopy(xalt, 0, w, n + iw - 1, n);
				csave = cauchy[0];
				iflag = 1;
			} else {
				break;
			}
		}
		if (csave > cauchy[0]) {
			System.arraycopy(w, n + iw - 1, xalt, 0, n);
			cauchy[0] = csave;
		}
	}

	private static void prelim(final Function<? super double[], Double> func, final int n, final int npt,
			final double[] x, final double[] xl, final double[] xu, final double rhobeg, final int iprint,
			final int maxfun, final double[] xbase, final double[][] xpt, final double[] fval, final double[] gopt,
			final double[] hq, final double[] pq, final double[][] bmat, final double[][] zmat, final int ndim,
			final double[] sl, final double[] su, final int[] nf, final int[] kopt) {

		final double half = 0.5, one = 1.0, two = 2.0, zero = 0.0, rhosq = rhobeg * rhobeg, recip = one / rhosq;
		final int np = n + 1;
		int i, ipt = 0, itemp, j, jpt = 0, k, ih, nfm, nfx;
		double stepa = 0.0, stepb = 0.0, fbeg = 0.0, f, temp, diff;

		// Set XBASE to the initial vector of variables, and set the initial
		// elements of XPT, BMAT, HQ, PQ and ZMAT to zero.
		System.arraycopy(x, 0, xbase, 0, n);
		for (j = 1; j <= n; j++) {
			for (k = 1; k <= npt; k++) {
				xpt[k - 1][j - 1] = zero;
			}
			for (i = 1; i <= ndim; i++) {
				bmat[i - 1][j - 1] = zero;
			}
		}
		Arrays.fill(hq, 0, n * np / 2, zero);
		Arrays.fill(pq, 0, npt, zero);
		for (k = 1; k <= npt; k++) {
			Arrays.fill(zmat[k - 1], 0, npt - np, zero);
		}

		// Begin the initialization procedure.
		nf[0] = 0;
		while (true) {

			nfm = nf[0];
			nfx = nf[0] - n;
			++nf[0];
			if (nfm <= 2 * n) {
				if (nfm >= 1 && nfm <= n) {
					stepa = rhobeg;
					if (su[nfm - 1] == zero) {
						stepa = -stepa;
					}
					xpt[nf[0] - 1][nfm - 1] = stepa;
				} else if (nfm > n) {
					stepa = xpt[nf[0] - n - 1][nfx - 1];
					stepb = -rhobeg;
					if (sl[nfx - 1] == zero) {
						stepb = Math.min(two * rhobeg, su[nfx - 1]);
					}
					if (su[nfx - 1] == zero) {
						stepb = Math.max(-two * rhobeg, sl[nfx - 1]);
					}
					xpt[nf[0] - 1][nfx - 1] = stepb;
				}
			} else {
				itemp = (nfm - np) / n;
				jpt = nfm - itemp * n - n;
				ipt = jpt + itemp;
				if (ipt > n) {
					itemp = jpt;
					jpt = ipt - n;
					ipt = itemp;
				}
				xpt[nf[0] - 1][ipt - 1] = xpt[ipt + 1 - 1][ipt - 1];
				xpt[nf[0] - 1][jpt - 1] = xpt[jpt + 1 - 1][jpt - 1];
			}

			// Calculate the next value of F. The least function value so far and
			// its index are required.
			for (j = 1; j <= n; j++) {
				final double sum = xbase[j - 1] + xpt[nf[0] - 1][j - 1];
				x[j - 1] = Math.min(Math.max(xl[j - 1], sum), xu[j - 1]);
				if (xpt[nf[0] - 1][j - 1] == sl[j - 1]) {
					x[j - 1] = xl[j - 1];
				}
				if (xpt[nf[0] - 1][j - 1] == su[j - 1]) {
					x[j - 1] = xu[j - 1];
				}
			}
			f = func.apply(x);
			fval[nf[0] - 1] = f;
			if (nf[0] == 1) {
				fbeg = f;
				kopt[0] = 1;
			} else if (f < fval[kopt[0] - 1]) {
				kopt[0] = nf[0];
			}

			// Set the nonzero initial elements of BMAT and the quadratic model in
			// the
			// cases when NF is at most 2*N+1.
			if (nf[0] <= 2 * n + 1) {
				if (nf[0] >= 2 && nf[0] <= n + 1) {
					gopt[nfm - 1] = (f - fbeg) / stepa;
					if (npt < nf[0] + n) {
						bmat[1 - 1][nfm - 1] = -one / stepa;
						bmat[nf[0] - 1][nfm - 1] = one / stepa;
						bmat[npt + nfm - 1][nfm - 1] = -half * rhosq;
					}
				} else if (nf[0] >= n + 2) {
					ih = nfx * (nfx + 1) / 2;
					temp = (f - fbeg) / stepb;
					diff = stepb - stepa;
					hq[ih - 1] = two * (temp - gopt[nfx - 1]) / diff;
					gopt[nfx - 1] = (gopt[nfx - 1] * stepb - temp * stepa) / diff;
					if (stepa * stepb < zero) {
						if (f < fval[nf[0] - n - 1]) {
							fval[nf[0] - 1] = fval[nf[0] - n - 1];
							fval[nf[0] - n - 1] = f;
							if (kopt[0] == nf[0]) {
								kopt[0] = nf[0] - n;
							}
							xpt[nf[0] - n - 1][nfx - 1] = stepb;
							xpt[nf[0] - 1][nfx - 1] = stepa;
						}
					}
					bmat[1 - 1][nfx - 1] = -(stepa + stepb) / (stepa * stepb);
					bmat[nf[0] - 1][nfx - 1] = -half / xpt[nf[0] - n - 1][nfx - 1];
					bmat[nf[0] - n - 1][nfx - 1] = -bmat[1 - 1][nfx - 1] - bmat[nf[0] - 1][nfx - 1];
					zmat[1 - 1][nfx - 1] = Math.sqrt(two) / (stepa * stepb);
					zmat[nf[0] - 1][nfx - 1] = Math.sqrt(half) / rhosq;
					zmat[nf[0] - n - 1][nfx - 1] = -zmat[1 - 1][nfx - 1] - zmat[nf[0] - 1][nfx - 1];
				}
			} else {

				// Set the off-diagonal second derivatives of the Lagrange functions
				// and
				// the initial quadratic model.
				ih = ipt * (ipt - 1) / 2 + jpt;
				zmat[1 - 1][nfx - 1] = recip;
				zmat[nf[0] - 1][nfx - 1] = recip;
				zmat[ipt + 1 - 1][nfx - 1] = -recip;
				zmat[jpt + 1 - 1][nfx - 1] = -recip;
				temp = xpt[nf[0] - 1][ipt - 1] * xpt[nf[0] - 1][jpt - 1];
				hq[ih - 1] = (fbeg - fval[ipt + 1 - 1] - fval[jpt + 1 - 1] + f) / temp;
			}
			if (!(nf[0] < npt && nf[0] < maxfun)) {
				break;
			}
		}
	}

	private static void update(final int n, final int npt, final double[][] bmat, final double[][] zmat, final int ndim,
			final double[] vlag, final double[] beta, final double[] denom, final int knew, final double[] w,
			final int iw) {

		double ztest, temp, tempa, tempb, alpha, tau;
		int i, j, jl, jp, k;

		// Set some constants.
		final double one = 1.0, zero = 0.0;
		final int nptm = npt - n - 1;
		ztest = zero;
		for (k = 1; k <= npt; ++k) {
			for (j = 1; j <= nptm; ++j) {
				ztest = Math.max(ztest, Math.abs(zmat[k - 1][j - 1]));
			}
		}
		ztest *= 1.0e-20;

		// Apply the rotations that put zeros in the KNEW-th row of ZMAT.
		jl = 1;
		for (j = 2; j <= nptm; ++j) {
			if (Math.abs(zmat[knew - 1][j - 1]) > ztest) {
				temp = RealMath.hypot(zmat[knew - 1][1 - 1], zmat[knew - 1][j - 1]);
				tempa = zmat[knew - 1][1 - 1] / temp;
				tempb = zmat[knew - 1][j - 1] / temp;
				for (i = 1; i <= npt; ++i) {
					temp = tempa * zmat[i - 1][1 - 1] + tempb * zmat[i - 1][j - 1];
					zmat[i - 1][j - 1] = tempa * zmat[i - 1][j - 1] - tempb * zmat[i - 1][1 - 1];
					zmat[i - 1][1 - 1] = temp;
				}
			}
			zmat[knew - 1][j - 1] = zero;
		}

		// Put the first NPT components of the KNEW-th column of HLAG into W,
		// and calculate the parameters of the updating formula.
		for (i = 1; i <= npt; ++i) {
			w[i - 1 + iw - 1] = zmat[knew - 1][1 - 1] * zmat[i - 1][1 - 1];
		}
		alpha = w[knew - 1 + iw - 1];
		tau = vlag[knew - 1];
		vlag[knew - 1] -= one;

		// Complete the updating of ZMAT.
		temp = Math.sqrt(denom[0]);
		tempb = zmat[knew - 1][1 - 1] / temp;
		tempa = tau / temp;
		for (i = 1; i <= npt; ++i) {
			zmat[i - 1][1 - 1] = tempa * zmat[i - 1][1 - 1] - tempb * vlag[i - 1];
		}

		// Finally, update the matrix BMAT.
		for (j = 1; j <= n; ++j) {
			jp = npt + j;
			w[jp - 1 + iw - 1] = bmat[knew - 1][j - 1];
			tempa = (alpha * vlag[jp - 1] - tau * w[jp - 1 + iw - 1]) / denom[0];
			tempb = (-beta[0] * w[jp - 1 + iw - 1] - tau * vlag[jp - 1]) / denom[0];
			for (i = 1; i <= jp; ++i) {
				bmat[i - 1][j - 1] += tempa * vlag[i - 1] + tempb * w[i - 1 + iw - 1];
				if (i > npt) {
					bmat[jp - 1][i - npt - 1] = bmat[i - 1][j - 1];
				}
			}
		}
	}
}
