package opt.multivariate.unconstrained.order0.quad;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.unconstrained.order0.GradientFreeOptimizer;
import utils.BlasMath;
import utils.Constants;
import utils.RealMath;

/**
 * 
 * @author Michael
 */
public final class NewuoaAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final Function<Integer, Integer> mySize;
	private final double myRho0;
	private final int myMaxFev;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param initialStep
	 * @param maxEvaluations
	 * @param sizeFunction
	 */
	public NewuoaAlgorithm(final double tolerance, final double initialStep, final int maxEvaluations,
			final Function<Integer, Integer> sizeFunction) {
		super(tolerance);
		myRho0 = initialStep;
		myMaxFev = maxEvaluations;
		mySize = sizeFunction;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialStep
	 * @param maxEvaluations
	 */
	public NewuoaAlgorithm(final double tolerance, final double initialStep, final int maxEvaluations) {
		this(tolerance, initialStep, maxEvaluations, d -> 2 * d + 1);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {
		// nothing to do here
	}

	@Override
	public final void iterate() {
		// nothing to do here
	}

	@Override
	public final double[] optimize(final Function<? super double[], Double> func, final double[] guess) {

		// prepare data
		final int n = guess.length;
		final int npt = mySize.apply(n);
		final int[] fev = { 0 };
		double[] x = Arrays.copyOf(guess, n);

		// call main subroutine
		x = newuoa(func, n, npt, x, myRho0, myTol, myMaxFev, fev);
		myEvals = fev[0];
		return x;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double[] newuoa(final Function<? super double[], Double> calfun, final int n, final int npt,
			final double[] x, final double rhobeg, final double rhoend, final int maxfun, final int[] fev) {

		// prepare variables
		final int ndim = npt + n;
		final double[] xbase = new double[n], xopt = new double[n], xnew = new double[n], fval = new double[npt],
				gq = new double[n], hq = new double[n * (n + 1) / 2], pq = new double[npt], d = new double[n],
				vlag = new double[ndim], w = new double[10 * ndim];
		final double[][] xpt = new double[npt][n], bmat = new double[ndim][n], zmat = new double[npt][npt - (n + 1)];
		fev[0] = 0;

		// call main subroutine
		newuob(calfun, n, npt, x, rhobeg, rhoend, maxfun, xbase, xopt, xnew, xpt, fval, gq, hq, pq, bmat, zmat, ndim, d,
				vlag, w, fev);
		return x;
	}

	private static void newuob(final Function<? super double[], Double> calfun, final int n, final int npt,
			final double[] x, final double rhobeg, final double rhoend, final int maxfun, final double[] xbase,
			final double[] xopt, final double[] xnew, final double[][] xpt, final double[] fval, final double[] gq,
			final double[] hq, final double[] pq, final double[][] bmat, final double[][] zmat, final int ndim,
			final double[] d, final double[] vlag, final double[] w, final int[] fev) {

		int i, ih, ip, ipt = 0, itemp, itest = 0, j, jp, jpt = 0, k, knew = 0, ktemp, ksave = 0, kopt = 0, nf, nfm = 0,
				nfmm = 0, nfsav = 0;
		double bsum, dx, delta = 0.0, detrat, diff = 0.0, diffa = 0.0, diffb = 0.0, diffc = 0.0, dsq = 0.0, dnorm = 0.0,
				dstep = 0.0, distsq, f = 0.0, fbeg = 0.0, fopt = 0.0, fsave = 0.0, gqsq, gisq, hdiag, ratio = 0.0,
				rhosq, recip, reciq, rho = 0.0, sum, suma, sumb, sumz, temp, tempq, xipt = 0.0, xjpt = 0.0,
				xoptsq = 0.0, vquad = 0.0;
		final double[] crvmin = new double[1], alpha = new double[1], beta = new double[1];
		final double[][] wvec1 = new double[ndim][5], prod1 = new double[ndim][5];
		final int[] idz = new int[1];

		// Set some constants
		final double HALF = 0.5, ONE = 1.0, TENTH = 0.1, ZERO = 0.0;
		final int np = n + 1, nh = (n * np) / 2, nptm = npt - np, nftest = Math.max(maxfun, 1);

		// Set the initial elements of XPT, BMAT, HQ, PQ and ZMAT to zero
		// this is done in Java
		// Begin the initialization procedure. NF becomes one more than the number
		// of function values so far. The coordinates of the displacement of the
		// next initial interpolation point from XBASE are set in XPT(NF,.)
		rhosq = rhobeg * rhobeg;
		recip = ONE / rhosq;
		reciq = Math.sqrt(HALF) / rhosq;
		nf = 0;

		int gotoflag = 50;
		while (true) {

			if (gotoflag == 50) {
				nfm = nf;
				nfmm = nf - n;
				++nf;
				if (nfm <= 2 * n) {
					if (nfm >= 1 && nfm <= n) {
						xpt[nf - 1][nfm - 1] = rhobeg;
					} else if (nfm > n) {
						xpt[nf - 1][nfmm - 1] = -rhobeg;
					}
				} else {
					itemp = (nfmm - 1) / n;
					jpt = nfm - itemp * n - n;
					ipt = jpt + itemp;
					if (ipt > n) {
						itemp = jpt;
						jpt = ipt - n;
						ipt = itemp;
					}
					xipt = rhobeg;
					if (fval[ipt + np - 1] < fval[ipt + 1 - 1]) {
						xipt = -xipt;
					}
					xjpt = rhobeg;
					if (fval[jpt + np - 1] < fval[jpt + 1 - 1]) {
						xjpt = -xjpt;
					}
					xpt[nf - 1][ipt - 1] = xipt;
					xpt[nf - 1][jpt - 1] = xjpt;
				}

				// Calculate the next value of F, label 70 being reached immediately
				// after this calculation. The least function value so far and its
				// index
				// are required
				BlasMath.dxpy1(n, xpt[nf - 1], 1, xbase, 1, x, 1);
				gotoflag = 310;
			}

			if (gotoflag == 70) {
				fval[nf - 1] = f;
				if (nf == 1) {
					fbeg = fopt = f;
					kopt = 1;
				} else if (f < fopt) {
					fopt = f;
					kopt = nf;
				}

				// Set the nonzero initial elements of BMAT and the quadratic model
				// in
				// the cases when NF is at most 2*N+1
				if (nfm <= 2 * n) {
					if (nfm >= 1 && nfm <= n) {
						gq[nfm - 1] = (f - fbeg) / rhobeg;
						if (npt < nf + n) {
							bmat[1 - 1][nfm - 1] = -ONE / rhobeg;
							bmat[nf - 1][nfm - 1] = ONE / rhobeg;
							bmat[npt + nfm - 1][nfm - 1] = -HALF / rhosq;
						}
					} else if (nfm > n) {
						bmat[nf - n - 1][nfmm - 1] = HALF / rhobeg;
						bmat[nf - 1][nfmm - 1] = -HALF / rhobeg;
						zmat[1 - 1][nfmm - 1] = -reciq - reciq;
						zmat[nf - n - 1][nfmm - 1] = reciq;
						zmat[nf - 1][nfmm - 1] = reciq;
						ih = (nfmm * (nfmm + 1)) / 2;
						temp = (fbeg - f) / rhobeg;
						hq[ih - 1] = (gq[nfmm - 1] - temp) / rhobeg;
						gq[nfmm - 1] = HALF * (gq[nfmm - 1] + temp);
					}
				} else {

					// Set the off-diagonal second derivatives of the Lagrange
					// functions and
					// the initial quadratic model
					ih = (ipt * (ipt - 1)) / 2 + jpt;
					if (xipt < ZERO) {
						ipt += n;
					}
					if (xjpt < ZERO) {
						jpt += n;
					}
					zmat[1 - 1][nfmm - 1] = recip;
					zmat[nf - 1][nfmm - 1] = recip;
					zmat[ipt + 1 - 1][nfmm - 1] = -recip;
					zmat[jpt + 1 - 1][nfmm - 1] = -recip;
					hq[ih - 1] = (fbeg - fval[ipt + 1 - 1] - fval[jpt + 1 - 1] + f) / (xipt * xjpt);
				}
				if (nf < npt) {
					gotoflag = 50;
					continue;
				} else {

					// Begin the iterative procedure, because the initial model is
					// complete
					rho = rhobeg;
					delta = rho;
					idz[0] = 1;
					diffa = diffb = ZERO;
					itest = 0;
					System.arraycopy(xpt[kopt - 1], 0, xopt, 0, n);
					xoptsq = BlasMath.ddotm(n, xopt, 1, xopt, 1);
					nfsav = nf;
					gotoflag = 100;
				}
			}

			if (gotoflag == 100) {

				// Generate the next trust region step and test its length. Set KNEW
				// to -1 if the purpose of the next F will be to improve the model
				knew = 0;
				trsapp(n, npt, xopt, xpt, gq, hq, pq, delta, d, w, w, w, w, crvmin, 0, np - 1, np + n - 1,
						np + 2 * n - 1);
				dsq = BlasMath.ddotm(n, d, 1, d, 1);
				dnorm = Math.min(delta, Math.sqrt(dsq));
				if (dnorm < HALF * rho) {
					knew = -1;
					delta *= TENTH;
					ratio = -1.0;
					if (delta <= 1.5 * rho) {
						delta = rho;
					}
					if (nf <= nfsav + 2) {
						gotoflag = 460;
					} else {
						temp = 0.125 * crvmin[0] * rho * rho;
						if (temp <= Math.max(diffa, Math.max(diffb, diffc))) {
							gotoflag = 460;
						} else {
							gotoflag = 490;
						}
					}
				} else {
					gotoflag = 120;
				}
			}

			if (gotoflag == 120) {

				// Shift XBASE if XOPT may be too far from XBASE. First make the
				// changes
				// to BMAT that do not depend on ZMAT
				if (dsq <= 1.0e-3 * xoptsq) {
					tempq = 0.25 * xoptsq;
					for (k = 1; k <= npt; ++k) {
						sum = BlasMath.ddotm(n, xpt[k - 1], 1, xopt, 1);
						temp = pq[k - 1] * sum;
						sum -= (HALF * xoptsq);
						w[npt + k - 1] = sum;
						for (i = 1; i <= n; ++i) {
							gq[i - 1] += (temp * xpt[k - 1][i - 1]);
							xpt[k - 1][i - 1] -= (HALF * xopt[i - 1]);
							vlag[i - 1] = bmat[k - 1][i - 1];
							w[i - 1] = sum * xpt[k - 1][i - 1] + tempq * xopt[i - 1];
							ip = npt + i;
							for (j = 1; j <= i; ++j) {
								bmat[ip - 1][j - 1] += (vlag[i - 1] * w[j - 1] + w[i - 1] * vlag[j - 1]);
							}
						}
					}

					// Then the revisions of BMAT that depend on ZMAT are calculated
					for (k = 1; k <= nptm; ++k) {
						sumz = ZERO;
						for (i = 1; i <= npt; ++i) {
							sumz += zmat[i - 1][k - 1];
							w[i - 1] = w[npt + i - 1] * zmat[i - 1][k - 1];
						}
						for (j = 1; j <= n; ++j) {
							sum = tempq * sumz * xopt[j - 1];
							for (i = 1; i <= npt; ++i) {
								sum += (w[i - 1] * xpt[i - 1][j - 1]);
							}
							vlag[j - 1] = sum;
							if (k < idz[0]) {
								sum = -sum;
							}
							for (i = 1; i <= npt; ++i) {
								bmat[i - 1][j - 1] += (sum * zmat[i - 1][k - 1]);
							}
						}
						for (i = 1; i <= n; ++i) {
							ip = i + npt;
							temp = vlag[i - 1];
							if (k < idz[0]) {
								temp = -temp;
							}
							BlasMath.daxpym(i, temp, vlag, 1, bmat[ip - 1], 1);
						}
					}

					// The following instructions complete the shift of XBASE,
					// including
					// the changes to the parameters of the quadratic model
					ih = 0;
					for (j = 1; j <= n; ++j) {
						w[j - 1] = ZERO;
						for (k = 1; k <= npt; ++k) {
							w[j - 1] += (pq[k - 1] * xpt[k - 1][j - 1]);
							xpt[k - 1][j - 1] -= (HALF * xopt[j - 1]);
						}
						for (i = 1; i <= j; ++i) {
							++ih;
							if (i < j) {
								gq[j - 1] += (hq[ih - 1] * xopt[i - 1]);
							}
							gq[i - 1] += (hq[ih - 1] * xopt[j - 1]);
							hq[ih - 1] += (w[i - 1] * xopt[j - 1] + xopt[i - 1] * w[j - 1]);
							bmat[npt + i - 1][j - 1] = bmat[npt + j - 1][i - 1];
						}
					}
					for (j = 1; j <= n; ++j) {
						xbase[j - 1] += xopt[j - 1];
						xopt[j - 1] = ZERO;
					}
					xoptsq = ZERO;
				}

				// Pick the model step if KNEW is positive. A different choice of D
				// may be made later, if the choice of D by BIGLAG causes substantial
				// cancellation in DENOM
				if (knew > 0) {
					biglag(n, npt, xopt, xpt, bmat, zmat, idz[0], ndim, knew, dstep, d, alpha, vlag, vlag, w, w, w, 0,
							npt, 0, np - 1, np + n - 1);
				}

				// Calculate VLAG and BETA for the current choice of D. The first NPT
				// components of W_check will be held in W
				for (k = 1; k <= npt; ++k) {
					suma = sumb = sum = ZERO;
					for (j = 1; j <= n; ++j) {
						suma += (xpt[k - 1][j - 1] * d[j - 1]);
						sumb += (xpt[k - 1][j - 1] * xopt[j - 1]);
						sum += (bmat[k - 1][j - 1] * d[j - 1]);
					}
					w[k - 1] = suma * (HALF * suma + sumb);
					vlag[k - 1] = sum;
				}
				beta[0] = ZERO;
				for (k = 1; k <= nptm; ++k) {
					sum = ZERO;
					for (i = 1; i <= npt; ++i) {
						sum += (zmat[i - 1][k - 1] * w[i - 1]);
					}
					if (k < idz[0]) {
						beta[0] += (sum * sum);
						sum = -sum;
					} else {
						beta[0] -= (sum * sum);
					}
					for (i = 1; i <= npt; ++i) {
						vlag[i - 1] += (sum * zmat[i - 1][k - 1]);
					}
				}
				bsum = dx = ZERO;
				for (j = 1; j <= n; ++j) {
					sum = ZERO;
					for (i = 1; i <= npt; ++i) {
						sum += (w[i - 1] * bmat[i - 1][j - 1]);
					}
					bsum += (sum * d[j - 1]);
					jp = npt + j;
					sum += BlasMath.ddotm(n, bmat[jp - 1], 1, d, 1);
					vlag[jp - 1] = sum;
					bsum += (sum * d[j - 1]);
					dx += (d[j - 1] * xopt[j - 1]);
				}
				beta[0] = dx * dx + dsq * (xoptsq + dx + dx + HALF * dsq) + beta[0] - bsum;
				vlag[kopt - 1] += ONE;

				// If KNEW is positive and if the cancellation in DENOM is
				// unacceptable,
				// then BIGDEN calculates an alternative model step, XNEW being used
				// for
				// working space
				int i1, i2;
				if (knew > 0) {
					temp = ONE + alpha[0] * beta[0] / (vlag[knew - 1] * vlag[knew - 1]);
					if (Math.abs(temp) <= 0.8) {
						for (i1 = 1; i1 <= ndim; ++i1) {
							for (i2 = 1; i2 <= 5; ++i2) {
								wvec1[i1 - 1][i2 - 1] = w[ndim + (i2 - 1) * ndim + (i1 - 1)];
								prod1[i1 - 1][i2 - 1] = w[6 * ndim + (i2 - 1) * ndim + (i1 - 1)];
							}
						}
						bigden(n, npt, xopt, xpt, bmat, zmat, idz[0], ndim, kopt, knew, d, w, vlag, beta, xnew, wvec1,
								prod1);
						for (i1 = 1; i1 <= ndim; ++i1) {
							for (i2 = 1; i2 <= 5; ++i2) {
								w[ndim + (i2 - 1) * ndim + (i1 - 1)] = wvec1[i1 - 1][i2 - 1];
								w[6 * ndim + (i2 - 1) * ndim + (i1 - 1)] = prod1[i1 - 1][i2 - 1];
							}
						}
					}
				}
				gotoflag = 290;
			}

			if (gotoflag == 290) {

				// Calculate the next value of the objective function
				for (i = 1; i <= n; ++i) {
					xnew[i - 1] = xopt[i - 1] + d[i - 1];
					x[i - 1] = xbase[i - 1] + xnew[i - 1];
				}
				++nf;
				gotoflag = 310;
			}

			if (gotoflag == 310) {
				if (nf > nftest) {
					--nf;
					if (fopt <= f) {
						BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
					}
					return;
				}
				f = calfun.apply(x);
				++fev[0];
				if (nf <= npt) {
					gotoflag = 70;
					continue;
				}
				if (knew == -1) {
					if (fopt <= f) {
						BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
					}
					return;
				}

				// Use the quadratic model to predict the change in F due to the step
				// D, and set DIFF to the error of this prediction
				vquad = ZERO;
				ih = 0;
				for (j = 1; j <= n; ++j) {
					vquad += (d[j - 1] * gq[j - 1]);
					for (i = 1; i <= j; ++i) {
						++ih;
						temp = d[i - 1] * xnew[j - 1] + d[j - 1] * xopt[i - 1];
						if (i == j) {
							temp *= HALF;
						}
						vquad += (temp * hq[ih - 1]);
					}
				}
				vquad += BlasMath.ddotm(npt, pq, 1, w, 1);
				diff = f - fopt - vquad;
				diffc = diffb;
				diffb = diffa;
				diffa = Math.abs(diff);
				if (dnorm > rho) {
					nfsav = nf;
				}

				// Update FOPT and XOPT if the new F is the least value of the
				// objective
				// function so far. The branch when KNEW is positive occurs if D is
				// not
				// a trust region step
				fsave = fopt;
				if (f < fopt) {
					fopt = f;
					System.arraycopy(xnew, 0, xopt, 0, n);
					xoptsq = BlasMath.ddotm(n, xopt, 1, xopt, 1);
				}
				ksave = knew;
				if (knew > 0) {
					gotoflag = 410;
				} else {

					// Pick the next value of DELTA after a trust region step
					if (vquad >= ZERO) {
						if (fopt <= f) {
							BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
						}
						return;
					}
					ratio = (f - fsave) / vquad;
					if (ratio <= TENTH) {
						delta = HALF * dnorm;
					} else if (ratio <= 0.7) {
						delta = Math.max(HALF * delta, dnorm);
					} else {
						delta = Math.max(HALF * delta, dnorm + dnorm);
					}
					if (delta <= 1.5 * rho) {
						delta = rho;
					}

					// Set KNEW to the index of the next interpolation point to be
					// deleted
					final double tmp = Math.max(TENTH * delta, rho);
					rhosq = (tmp * tmp);
					ktemp = 0;
					detrat = ZERO;
					if (f >= fsave) {
						ktemp = kopt;
						detrat = ONE;
					}
					for (k = 1; k <= npt; ++k) {
						hdiag = ZERO;
						for (j = 1; j <= nptm; ++j) {
							temp = ONE;
							if (j < idz[0]) {
								temp = -ONE;
							}
							hdiag += (temp * zmat[k - 1][j - 1] * zmat[k - 1][j - 1]);
						}
						temp = Math.abs(beta[0] * hdiag + vlag[k - 1] * vlag[k - 1]);
						distsq = ZERO;
						for (j = 1; j <= n; ++j) {
							final double dst = xpt[k - 1][j - 1] - xopt[j - 1];
							distsq += (dst * dst);
						}
						if (distsq > rhosq) {
							temp *= (distsq / rhosq) * (distsq / rhosq) * (distsq / rhosq);
						}
						if (temp > detrat && k != ktemp) {
							detrat = temp;
							knew = k;
						}
					}
					if (knew == 0) {
						gotoflag = 460;
					} else {
						gotoflag = 410;
					}
				}
			}

			if (gotoflag == 410) {

				// Update BMAT, ZMAT and IDZ, so that the KNEW-th interpolation point
				// can be moved. Begin the updating of the quadratic model, starting
				// with the explicit second derivative term
				update(n, npt, bmat, zmat, idz, ndim, vlag, beta[0], knew, w);
				fval[knew - 1] = f;
				ih = 0;
				for (i = 1; i <= n; ++i) {
					temp = pq[knew - 1] * xpt[knew - 1][i - 1];
					for (j = 1; j <= i; ++j) {
						++ih;
						hq[ih - 1] += (temp * xpt[knew - 1][j - 1]);
					}
				}
				pq[knew - 1] = ZERO;

				// Update the other second derivative parameters, and then the
				// gradient
				// vector of the model. Also include the new interpolation point
				for (j = 1; j <= nptm; ++j) {
					temp = diff * zmat[knew - 1][j - 1];
					if (j < idz[0]) {
						temp = -temp;
					}
					for (k = 1; k <= npt; ++k) {
						pq[k - 1] += (temp * zmat[k - 1][j - 1]);
					}
				}
				gqsq = ZERO;
				for (i = 1; i <= n; ++i) {
					gq[i - 1] += (diff * bmat[knew - 1][i - 1]);
					gqsq += (gq[i - 1] * gq[i - 1]);
				}
				System.arraycopy(xnew, 0, xpt[knew - 1], 0, n);

				// If a trust region step makes a small change to the objective
				// function,
				// then calculate the gradient of the least Frobenius norm
				// interpolant at
				// XBASE, and store it in W, using VLAG for a vector of right hand
				// sides
				if (ksave == 0 && delta == rho) {
					if (Math.abs(ratio) > 1.0e-2) {
						itest = 0;
					} else {
						for (k = 1; k <= npt; ++k) {
							vlag[k - 1] = fval[k - 1] - fval[kopt - 1];
						}
						gisq = ZERO;
						for (i = 1; i <= n; ++i) {
							sum = ZERO;
							for (k = 1; k <= npt; ++k) {
								sum += (bmat[k - 1][i - 1] * vlag[k - 1]);
							}
							gisq += (sum * sum);
							w[i - 1] = sum;
						}

						// Test whether to replace the new quadratic model by the
						// least Frobenius
						// norm interpolant, making the replacement if the test is
						// satisfied
						++itest;
						if (gqsq < 1.0e2 * gisq) {
							itest = 0;
						}
						if (itest >= 3) {
							System.arraycopy(w, 0, gq, 0, n);
							Arrays.fill(hq, ZERO);
							for (j = 1; j <= nptm; ++j) {
								w[j - 1] = ZERO;
								for (k = 1; k <= npt; ++k) {
									w[j - 1] += (vlag[k - 1] * zmat[k - 1][j - 1]);
								}
								if (j < idz[0]) {
									w[j - 1] = -w[j - 1];
								}
							}
							for (k = 1; k <= npt; ++k) {
								pq[k - 1] = BlasMath.ddotm(nptm, zmat[k - 1], 1, w, 1);
							}
							itest = 0;
						}
					}
				}
				if (f < fsave) {
					kopt = knew;
				}

				// If a trust region step has provided a sufficient decrease in F,
				// then
				// branch for another trust region calculation. The case KSAVE>0
				// occurs
				// when the new function value was calculated by a model step
				if (f <= fsave + TENTH * vquad || ksave > 0) {
					gotoflag = 100;
					continue;
				}

				// Alternatively, find out if the interpolation points are close
				// enough
				// to the best point so far
				knew = 0;
				gotoflag = 460;
			}

			if (gotoflag == 460) {
				distsq = 4.0 * delta * delta;
				for (k = 1; k <= npt; ++k) {
					sum = ZERO;
					for (j = 1; j <= n; ++j) {
						final double dst = xpt[k - 1][j - 1] - xopt[j - 1];
						sum += (dst * dst);
					}
					if (sum > distsq) {
						knew = k;
						distsq = sum;
					}
				}

				// If KNEW is positive, then set DSTEP, and branch back for the next
				// iteration, which will generate a "model step"
				if (knew > 0) {
					dstep = TENTH * Math.sqrt(distsq);
					dstep = Math.max(Math.min(dstep, HALF * delta), rho);
					dsq = dstep * dstep;
					gotoflag = 120;
					continue;
				}
				if (ratio > ZERO || Math.max(delta, dnorm) > rho) {
					gotoflag = 100;
					continue;
				} else {
					gotoflag = 490;
				}
			}

			if (gotoflag == 490) {

				// The calculations with the current value of RHO are complete. Pick
				// the
				// next values of RHO and DELTA
				if (rho > rhoend) {
					delta = HALF * rho;
					ratio = rho / rhoend;
					if (ratio <= 16.0) {
						rho = rhoend;
					} else if (ratio <= 250.0) {
						rho = Math.sqrt(ratio) * rhoend;
					} else {
						rho *= TENTH;
					}
					delta = Math.max(delta, rho);
					nfsav = nf;
					gotoflag = 100;
					continue;
				}

				// Return from the calculation, after another Newton-Raphson step, if
				// it is too short to have been tried before
				if (knew == -1) {
					gotoflag = 290;
				} else {
					if (fopt <= f) {
						BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
					}
					return;
				}
			}
		}
	}

	private static void bigden(final int n, final int npt, final double[] xopt, final double[][] xpt,
			final double[][] bmat, final double[][] zmat, final int idz, final int ndim, final int kopt, final int knew,
			final double[] d, final double[] w, final double[] vlag, final double[] beta, final double[] s,
			final double[][] wvec, final double[][] prod) {

		final double[] den = new double[9], denex = new double[9], par = new double[9];
		double alpha, angle, temp, dd, ds, ss, xoptsq, xoptd, xopts, dtest, dstemp, sstemp, diff, ssden, step, sum,
				sumold, densav, denold, denmax, tau, tempa, tempb, tempc;
		int i, ip, isave, iu, iterc, j, jc, k, ksav, nw;

		// Set some constants
		final double HALF = 0.5, ONE = 1.0, QUART = 0.25, TWO = 2.0, ZERO = 0.0, TWOPI = 2.0 * Constants.PI;
		final int nptm = npt - n - 1;

		// Store the first NPT elements of the KNEW-th column of H in W(N+1)
		// to W(N+NPT)
		for (k = 1; k <= npt; ++k) {
			w[n + k - 1] = ZERO;
		}
		for (j = 1; j <= nptm; ++j) {
			temp = zmat[knew - 1][j - 1];
			if (j < idz) {
				temp = -temp;
			}
			for (k = 1; k <= npt; ++k) {
				w[n + k - 1] += (temp * zmat[k - 1][j - 1]);
			}
		}
		alpha = w[n + knew - 1];

		// The initial search direction D is taken from the last call of BIGLAG,
		// and the initial S is set below, usually to the direction from X_OPT
		// to X_KNEW, but a different direction to an interpolation point may
		// be chosen, in order to prevent S from being nearly parallel to D
		dd = ds = ss = xoptsq = ZERO;
		for (i = 1; i <= n; ++i) {
			dd += (d[i - 1] * d[i - 1]);
			s[i - 1] = xpt[knew - 1][i - 1] - xopt[i - 1];
			ds += (d[i - 1] * s[i - 1]);
			ss += (s[i - 1] * s[i - 1]);
			xoptsq += (xopt[i - 1] * xopt[i - 1]);
		}
		if (ds * ds > 0.99 * dd * ss) {
			ksav = knew;
			dtest = ds * ds / ss;
			for (k = 1; k <= npt; ++k) {
				if (k != kopt) {
					dstemp = sstemp = ZERO;
					for (i = 1; i <= n; ++i) {
						diff = xpt[k - 1][i - 1] - xopt[i - 1];
						dstemp += (d[i - 1] * diff);
						sstemp += (diff * diff);
					}
					if (dstemp * dstemp / sstemp < dtest) {
						ksav = k;
						dtest = dstemp * dstemp / sstemp;
						ds = dstemp;
						ss = sstemp;
					}
				}
			}
			BlasMath.daxpy1(n, -1.0, xopt, 1, xpt[ksav - 1], 1, s, 1);
		}
		ssden = dd * ss - ds * ds;
		iterc = 0;
		densav = ZERO;

		while (true) {

			// Begin the iteration by overwriting S with a vector that has the
			// required length and direction
			++iterc;
			temp = ONE / Math.sqrt(ssden);
			xoptd = xopts = ZERO;
			for (i = 1; i <= n; ++i) {
				s[i - 1] = temp * (dd * s[i - 1] - ds * d[i - 1]);
				xoptd += (xopt[i - 1] * d[i - 1]);
				xopts += (xopt[i - 1] * s[i - 1]);
			}

			// Set the coefficients of the first two terms of BETA
			tempa = HALF * xoptd * xoptd;
			tempb = HALF * xopts * xopts;
			den[1 - 1] = dd * (xoptsq + HALF * dd) + tempa + tempb;
			den[2 - 1] = TWO * xoptd * dd;
			den[3 - 1] = TWO * xopts * dd;
			den[4 - 1] = tempa - tempb;
			den[5 - 1] = xoptd * xopts;
			for (i = 6; i <= 9; ++i) {
				den[i - 1] = ZERO;
			}

			// Put the coefficients of Wcheck in WVEC
			for (k = 1; k <= npt; ++k) {
				tempa = tempb = tempc = ZERO;
				for (i = 1; i <= n; ++i) {
					tempa += (xpt[k - 1][i - 1] * d[i - 1]);
					tempb += (xpt[k - 1][i - 1] * s[i - 1]);
					tempc += (xpt[k - 1][i - 1] * xopt[i - 1]);
				}
				wvec[k - 1][1 - 1] = QUART * (tempa * tempa + tempb * tempb);
				wvec[k - 1][2 - 1] = tempa * tempc;
				wvec[k - 1][3 - 1] = tempb * tempc;
				wvec[k - 1][4 - 1] = QUART * (tempa * tempa - tempb * tempb);
				wvec[k - 1][5 - 1] = HALF * tempa * tempb;
			}
			for (i = 1; i <= n; ++i) {
				ip = i + npt;
				wvec[ip - 1][1 - 1] = ZERO;
				wvec[ip - 1][2 - 1] = d[i - 1];
				wvec[ip - 1][3 - 1] = s[i - 1];
				wvec[ip - 1][4 - 1] = ZERO;
				wvec[ip - 1][5 - 1] = ZERO;
			}

			// Put the coefficents of THETA*Wcheck in PROD
			for (jc = 1; jc <= 5; ++jc) {
				nw = npt;
				if (jc == 2 || jc == 3) {
					nw = ndim;
				}
				for (k = 1; k <= npt; ++k) {
					prod[k - 1][jc - 1] = ZERO;
				}
				for (j = 1; j <= nptm; ++j) {
					sum = ZERO;
					for (k = 1; k <= npt; ++k) {
						sum += (zmat[k - 1][j - 1] * wvec[k - 1][jc - 1]);
					}
					if (j < idz) {
						sum = -sum;
					}
					for (k = 1; k <= npt; ++k) {
						prod[k - 1][jc - 1] += (sum * zmat[k - 1][j - 1]);
					}
				}
				if (nw == ndim) {
					for (k = 1; k <= npt; ++k) {
						sum = ZERO;
						for (j = 1; j <= n; ++j) {
							sum += (bmat[k - 1][j - 1] * wvec[npt + j - 1][jc - 1]);
						}
						prod[k - 1][jc - 1] += sum;
					}
				}
				for (j = 1; j <= n; ++j) {
					sum = ZERO;
					for (i = 1; i <= nw; ++i) {
						sum += (bmat[i - 1][j - 1] * wvec[i - 1][jc - 1]);
					}
					prod[npt + j - 1][jc - 1] = sum;
				}
			}

			// Include in DEN the part of BETA that depends on THETA
			for (k = 1; k <= ndim; ++k) {
				sum = ZERO;
				for (i = 1; i <= 5; ++i) {
					par[i - 1] = HALF * prod[k - 1][i - 1] * wvec[k - 1][i - 1];
					sum += par[i - 1];
				}
				den[1 - 1] = den[1 - 1] - par[1 - 1] - sum;
				tempa = prod[k - 1][1 - 1] * wvec[k - 1][2 - 1] + prod[k - 1][2 - 1] * wvec[k - 1][1 - 1];
				tempb = prod[k - 1][2 - 1] * wvec[k - 1][4 - 1] + prod[k - 1][4 - 1] * wvec[k - 1][2 - 1];
				tempc = prod[k - 1][3 - 1] * wvec[k - 1][5 - 1] + prod[k - 1][5 - 1] * wvec[k - 1][3 - 1];
				den[2 - 1] = den[2 - 1] - tempa - HALF * (tempb + tempc);
				den[6 - 1] -= (HALF * (tempb - tempc));
				tempa = prod[k - 1][1 - 1] * wvec[k - 1][3 - 1] + prod[k - 1][3 - 1] * wvec[k - 1][1 - 1];
				tempb = prod[k - 1][2 - 1] * wvec[k - 1][5 - 1] + prod[k - 1][5 - 1] * wvec[k - 1][2 - 1];
				tempc = prod[k - 1][3 - 1] * wvec[k - 1][4 - 1] + prod[k - 1][4 - 1] * wvec[k - 1][3 - 1];
				den[3 - 1] = den[3 - 1] - tempa - HALF * (tempb - tempc);
				den[7 - 1] -= (HALF * (tempb + tempc));
				tempa = prod[k - 1][1 - 1] * wvec[k - 1][4 - 1] + prod[k - 1][4 - 1] * wvec[k - 1][1 - 1];
				den[4 - 1] = den[4 - 1] - tempa - par[2 - 1] + par[3 - 1];
				tempa = prod[k - 1][1 - 1] * wvec[k - 1][5 - 1] + prod[k - 1][5 - 1] * wvec[k - 1][1 - 1];
				tempb = prod[k - 1][2 - 1] * wvec[k - 1][3 - 1] + prod[k - 1][3 - 1] * wvec[k - 1][2 - 1];
				den[5 - 1] = den[5 - 1] - tempa - HALF * tempb;
				den[8 - 1] = den[8 - 1] - par[4 - 1] + par[5 - 1];
				tempa = prod[k - 1][4 - 1] * wvec[k - 1][5 - 1] + prod[k - 1][5 - 1] * wvec[k - 1][4 - 1];
				den[9 - 1] -= (HALF * tempa);
			}

			// Extend DEN so that it holds all the coefficients of DENOM
			sum = ZERO;
			for (i = 1; i <= 5; ++i) {
				par[i - 1] = HALF * prod[knew - 1][i - 1] * prod[knew - 1][i - 1];
				sum += par[i - 1];
			}
			denex[1 - 1] = alpha * den[1 - 1] + par[1 - 1] + sum;
			tempa = TWO * prod[knew - 1][1 - 1] * prod[knew - 1][2 - 1];
			tempb = prod[knew - 1][2 - 1] * prod[knew - 1][4 - 1];
			tempc = prod[knew - 1][3 - 1] * prod[knew - 1][5 - 1];
			denex[2 - 1] = alpha * den[2 - 1] + tempa + tempb + tempc;
			denex[6 - 1] = alpha * den[6 - 1] + tempb - tempc;
			tempa = TWO * prod[knew - 1][1 - 1] * prod[knew - 1][3 - 1];
			tempb = prod[knew - 1][2 - 1] * prod[knew - 1][5 - 1];
			tempc = prod[knew - 1][3 - 1] * prod[knew - 1][4 - 1];
			denex[3 - 1] = alpha * den[3 - 1] + tempa + tempb - tempc;
			denex[7 - 1] = alpha * den[7 - 1] + tempb + tempc;
			tempa = TWO * prod[knew - 1][1 - 1] * prod[knew - 1][4 - 1];
			denex[4 - 1] = alpha * den[4 - 1] + tempa + par[2 - 1] - par[3 - 1];
			tempa = TWO * prod[knew - 1][1 - 1] * prod[knew - 1][5 - 1];
			denex[5 - 1] = alpha * den[5 - 1] + tempa + prod[knew - 1][2 - 1] * prod[knew - 1][3 - 1];
			denex[8 - 1] = alpha * den[8 - 1] + par[4 - 1] - par[5 - 1];
			denex[9 - 1] = alpha * den[9 - 1] + prod[knew - 1][4 - 1] * prod[knew - 1][5 - 1];

			// Seek the value of the angle that maximizes the modulus of DENOM
			sum = denex[1 - 1] + denex[2 - 1] + denex[4 - 1] + denex[6 - 1] + denex[8 - 1];
			denold = denmax = sum;
			isave = 0;
			iu = 49;
			temp = TWOPI / (iu + 1);
			par[1 - 1] = ONE;
			for (i = 1; i <= iu; ++i) {
				angle = (double) i * temp;
				par[2 - 1] = Math.cos(angle);
				par[3 - 1] = Math.sin(angle);
				for (j = 4; j <= 8; j += 2) {
					par[j - 1] = par[2 - 1] * par[j - 2 - 1] - par[3 - 1] * par[j - 1 - 1];
					par[j + 1 - 1] = par[2 - 1] * par[j - 1 - 1] + par[3 - 1] * par[j - 2 - 1];
				}
				sumold = sum;
				sum = BlasMath.ddotm(9, denex, 1, par, 1);
				if (Math.abs(sum) > Math.abs(denmax)) {
					denmax = sum;
					isave = i;
					tempa = sumold;
				} else if (i == isave + 1) {
					tempb = sum;
				}
			}
			if (isave == 0) {
				tempa = sum;
			}
			if (isave == iu) {
				tempb = denold;
			}
			step = ZERO;
			if (tempa != tempb) {
				tempa -= denmax;
				tempb -= denmax;
				step = HALF * (tempa - tempb) / (tempa + tempb);
			}
			angle = temp * ((double) isave + step);

			// Calculate the new parameters of the denominator, the new VLAG vector
			// and the new D. Then test for convergence
			par[2 - 1] = Math.cos(angle);
			par[3 - 1] = Math.sin(angle);
			for (j = 4; j <= 8; j += 2) {
				par[j - 1] = par[2 - 1] * par[j - 2 - 1] - par[3 - 1] * par[j - 1 - 1];
				par[j + 1 - 1] = par[2 - 1] * par[j - 1 - 1] + par[3 - 1] * par[j - 2 - 1];
			}
			beta[0] = denmax = ZERO;
			for (j = 1; j <= 9; ++j) {
				beta[0] += (den[j - 1] * par[j - 1]);
				denmax += (denex[j - 1] * par[j - 1]);
			}
			for (k = 1; k <= ndim; ++k) {
				vlag[k - 1] = BlasMath.ddotm(5, prod[k - 1], 1, par, 1);
			}
			tau = vlag[knew - 1];
			dd = tempa = tempb = ZERO;
			for (i = 1; i <= n; ++i) {
				d[i - 1] = par[2 - 1] * d[i - 1] + par[3 - 1] * s[i - 1];
				w[i - 1] = xopt[i - 1] + d[i - 1];
				dd += (d[i - 1] * d[i - 1]);
				tempa += (d[i - 1] * w[i - 1]);
				tempb += (w[i - 1] * w[i - 1]);
			}
			if (iterc >= n) {
				break;
			}
			if (iterc > 1) {
				densav = Math.max(densav, denold);
			}
			if (Math.abs(denmax) <= 1.1 * Math.abs(densav)) {
				break;
			}
			densav = denmax;

			// Set S to half the gradient of the denominator with respect to D.
			// Then branch for the next iteration
			for (i = 1; i <= n; ++i) {
				temp = tempa * xopt[i - 1] + tempb * d[i - 1] - vlag[npt + i - 1];
				s[i - 1] = tau * bmat[knew - 1][i - 1] + alpha * temp;
			}
			for (k = 1; k <= npt; ++k) {
				sum = BlasMath.ddotm(n, xpt[k - 1], 1, w, 1);
				temp = (tau * w[n + k - 1] - alpha * vlag[k - 1]) * sum;
				BlasMath.daxpym(n, temp, xpt[k - 1], 1, s, 1);
			}
			ss = ds = ZERO;
			for (i = 1; i <= n; ++i) {
				ss += (s[i - 1] * s[i - 1]);
				ds += (d[i - 1] * s[i - 1]);
			}
			ssden = dd * ss - ds * ds;
			if (ssden < 1.0e-8 * dd * ss) {
				break;
			}
		}

		// Set the vector W before the RETURN from the subroutine
		for (k = 1; k <= ndim; ++k) {
			w[k - 1] = BlasMath.ddotm(5, wvec[k - 1], 1, par, 1);
		}
		vlag[kopt - 1] += ONE;
	}

	private static void biglag(final int n, final int npt, final double[] xopt, final double[][] xpt,
			final double[][] bmat, final double[][] zmat, final int idz, final int ndim, final int knew,
			final double delta, final double[] d, final double[] alpha, final double[] hcol, final double[] gc,
			final double[] gd, final double[] s, final double[] w, final int hcoli, final int gci, final int gdi,
			final int si, final int wi) {

		// N.B. the last five parameters are added by me to simulate passing
		// arrays by index reference, e.g. w(wi)
		int i, isave, iu, j, k, iterc;
		double angle, cf1, cf2, cf3, cf4, cf5, cth, dd, denom, gg, sp, dhd, scale, ss, sth, tau, taubeg, taumax, tauold,
				temp, tempa = 0.0, tempb = 0.0, step, sum;

		// Set some constants
		final double HALF = 0.5, ZERO = 0.0, ONE = 1.0, TWOPI = 2.0 * Constants.PI, delsq = delta * delta;
		final int nptm = npt - n - 1;

		// Set the first NPT components of HCOL to the leading elements of the
		// KNEW-th column of H
		iterc = 0;
		Arrays.fill(hcol, hcoli, npt + hcoli, ZERO);
		for (j = 1; j <= nptm; ++j) {
			temp = zmat[knew - 1][j - 1];
			if (j < idz) {
				temp = -temp;
			}
			for (k = 1; k <= npt; ++k) {
				hcol[k - 1 + hcoli] += (temp * zmat[k - 1][j - 1]);
			}
		}
		alpha[0] = hcol[knew - 1 + hcoli];

		// Set the unscaled initial direction D. Form the gradient of LFUNC at
		// XOPT, and multiply D by the second derivative matrix of LFUNC
		dd = ZERO;
		for (i = 1; i <= n; ++i) {
			d[i - 1] = xpt[knew - 1][i - 1] - xopt[i - 1];
			gc[i - 1 + gci] = bmat[knew - 1][i - 1];
			gd[i - 1 + gdi] = ZERO;
			dd += (d[i - 1] * d[i - 1]);
		}
		for (k = 1; k <= npt; ++k) {
			temp = sum = ZERO;
			for (j = 1; j <= n; ++j) {
				temp += (xpt[k - 1][j - 1] * xopt[j - 1]);
				sum += (xpt[k - 1][j - 1] * d[j - 1]);
			}
			temp *= hcol[k - 1 + hcoli];
			sum *= hcol[k - 1 + hcoli];
			for (i = 1; i <= n; ++i) {
				gc[i - 1 + gci] += (temp * xpt[k - 1][i - 1]);
				gd[i - 1 + gdi] += (sum * xpt[k - 1][i - 1]);
			}
		}

		// Scale D and GD, with a sign change if required. Set S to another
		// vector in the initial two dimensional subspace
		gg = sp = dhd = ZERO;
		for (i = 1; i <= n; ++i) {
			gg += (gc[i - 1 + gci] * gc[i - 1 + gci]);
			sp += (d[i - 1] * gc[i - 1 + gci]);
			dhd += (d[i - 1] * gd[i - 1 + gdi]);
		}
		scale = delta / Math.sqrt(dd);
		if (sp * dhd < ZERO) {
			scale = -scale;
		}
		temp = ZERO;
		if (sp * sp > 0.99 * dd * gg) {
			temp = ONE;
		}
		tau = scale * (Math.abs(sp) + HALF * scale * Math.abs(dhd));
		if (gg * delsq < 0.01 * tau * tau) {
			temp = ONE;
		}
		for (i = 1; i <= n; ++i) {
			d[i - 1] *= scale;
			gd[i - 1 + gdi] *= scale;
			s[i - 1 + si] = gc[i - 1 + gci] + temp * gd[i - 1 + gdi];
		}

		while (true) {

			// Begin the iteration by overwriting S with a vector that has the
			// required length and direction, except that termination occurs if
			// the given D and S are nearly parallel
			++iterc;
			dd = sp = ss = ZERO;
			for (i = 1; i <= n; ++i) {
				dd += (d[i - 1] * d[i - 1]);
				sp += (d[i - 1] * s[i - 1 + si]);
				ss += (s[i - 1 + si] * s[i - 1 + si]);
			}
			temp = dd * ss - sp * sp;
			if (temp <= 1.0e-8 * dd * ss) {
				return;
			}
			denom = Math.sqrt(temp);
			for (i = 1; i <= n; ++i) {
				s[i - 1 + si] = (dd * s[i - 1 + si] - sp * d[i - 1]) / denom;
				w[i - 1 + wi] = ZERO;
			}

			// Calculate the coefficients of the objective function on the circle,
			// beginning with the multiplication of S by the second derivative matrix
			for (k = 1; k <= npt; ++k) {
				sum = BlasMath.ddotm(n, xpt[k - 1], 1, s, si + 1);
				sum *= hcol[k - 1 + hcoli];
				BlasMath.daxpym(n, sum, xpt[k - 1], 1, w, wi + 1);
			}
			cf1 = cf2 = cf3 = cf4 = cf5 = ZERO;
			for (i = 1; i <= n; ++i) {
				cf1 += (s[i - 1 + si] * w[i - 1 + wi]);
				cf2 += (d[i - 1] * gc[i - 1 + gci]);
				cf3 += (s[i - 1 + si] * gc[i - 1 + gci]);
				cf4 += (d[i - 1] * gd[i - 1 + gdi]);
				cf5 += (s[i - 1 + si] * gd[i - 1 + gdi]);
			}
			cf1 *= HALF;
			cf4 = HALF * cf4 - cf1;

			// Seek the value of the angle that maximizes the modulus of TAU
			taubeg = cf1 + cf2 + cf4;
			taumax = tauold = taubeg;
			isave = 0;
			iu = 49;
			temp = TWOPI / (iu + 1);
			for (i = 1; i <= iu; ++i) {
				angle = (double) i * temp;
				cth = Math.cos(angle);
				sth = Math.sin(angle);
				tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
				if (Math.abs(tau) > Math.abs(taumax)) {
					taumax = tau;
					isave = i;
					tempa = tauold;
				} else if (i == isave + 1) {
					tempb = tau;
				}
				tauold = tau;
			}
			if (isave == 0) {
				tempa = tau;
			}
			if (isave == iu) {
				tempb = taubeg;
			}
			step = ZERO;
			if (tempa != tempb) {
				tempa -= taumax;
				tempb -= taumax;
				step = HALF * (tempa - tempb) / (tempa + tempb);
			}
			angle = temp * ((double) isave + step);

			// Calculate the new D and GD. Then test for convergence
			cth = Math.cos(angle);
			sth = Math.sin(angle);
			tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
			for (i = 1; i <= n; ++i) {
				d[i - 1] = cth * d[i - 1] + sth * s[i - 1 + si];
				gd[i - 1 + gdi] = cth * gd[i - 1 + gdi] + sth * w[i - 1 + wi];
				s[i - 1 + si] = gc[i - 1 + gci] + gd[i - 1 + gdi];
			}
			if (Math.abs(tau) <= 1.1 * Math.abs(taubeg) || iterc >= n) {
				return;
			}
		}
	}

	private static void trsapp(final int n, final int npt, final double[] xopt, final double[][] xpt, final double[] gq,
			final double[] hq, final double[] pq, final double delta, final double[] step, final double[] d,
			final double[] g, final double[] hd, final double[] hs, final double[] crvmin, final int di, final int gi,
			final int hdi, final int hsi) {

		// Initialization, which includes setting HD to H times XOPT
		double alpha, angtest, angle, bstep = 0.0, cf, cth, dd = 0.0, dhd, dhs, dg, ds, gg = 0.0, ggbeg = 0.0, ggsav,
				qadd, qbeg, qred = 0.0, qmin, qnew = 0.0, qsav, ss, sth, sg = 0.0, shs = 0.0, sgk, ratio, reduc, temp,
				tempa = 0.0, tempb = 0.0;
		int i, ih, iterc, itermax, itersw, isave, iu, j, k;

		final double HALF = 0.5, ZERO = 0.0, TWOPI = 2.0 * Constants.PI, delsq = delta * delta;
		iterc = 0;
		itermax = n;
		itersw = itermax;
		System.arraycopy(xopt, 0, d, di, n);

		while (true) {

			// The following instructions act as a subroutine for setting the vector
			// HD to the vector D multiplied by the second derivative matrix of Q.
			// They are called from three different places, which are distinguished
			// by the value of ITERC
			Arrays.fill(hd, hdi, n + hdi, ZERO);
			for (k = 1; k <= npt; ++k) {
				temp = BlasMath.ddotm(n, xpt[k - 1], 1, d, di + 1);
				temp *= pq[k - 1];
				BlasMath.daxpym(n, temp, xpt[k - 1], 1, hd, hdi + 1);
			}
			ih = 0;
			for (j = 1; j <= n; ++j) {
				for (i = 1; i <= j; ++i) {
					++ih;
					if (i < j) {
						hd[j - 1 + hdi] += (hq[ih - 1] * d[i - 1 + di]);
					}
					hd[i - 1 + hdi] += (hq[ih - 1] * d[j - 1 + di]);
				}
			}
			if (iterc == 0) {

				// Prepare for the first line search
				qred = dd = ZERO;
				for (i = 1; i <= n; ++i) {
					step[i - 1] = hs[i - 1 + hsi] = ZERO;
					g[i - 1 + gi] = gq[i - 1] + hd[i - 1 + hdi];
					d[i - 1 + di] = -g[i - 1 + gi];
					dd += (d[i - 1 + di] * d[i - 1 + di]);
				}
				crvmin[0] = ZERO;
				if (dd == ZERO) {
					return;
				}
				ds = ss = ZERO;
				gg = dd;
				ggbeg = gg;

				// Calculate the step to the trust region boundary and the product HD
				++iterc;
				temp = delsq - ss;
				bstep = temp / (ds + Math.sqrt(ds * ds + dd * temp));
				continue;
			}
			if (iterc <= itersw) {

				dhd = BlasMath.ddotm(n, d, di + 1, hd, hdi + 1);

				// Update CRVMIN and set the step-length ALPHA
				alpha = bstep;
				if (dhd > ZERO) {
					temp = dhd / dd;
					if (iterc == 1) {
						crvmin[0] = temp;
					}
					crvmin[0] = Math.min(crvmin[0], temp);
					alpha = Math.min(alpha, gg / dhd);
				}
				qadd = alpha * (gg - HALF * alpha * dhd);
				qred += qadd;

				// Update STEP and HS
				ggsav = gg;
				gg = ZERO;
				for (i = 1; i <= n; ++i) {
					step[i - 1] += (alpha * d[i - 1 + di]);
					hs[i - 1 + hsi] += (alpha * hd[i - 1 + hdi]);
					final double dst = g[i - 1 + gi] + hs[i - 1 + hsi];
					gg += (dst * dst);
				}

				// Begin another conjugate direction iteration if required
				if (alpha < bstep) {
					if (qadd <= 0.01 * qred || gg <= 1.0e-4 * ggbeg || iterc == itermax) {
						return;
					}
					temp = gg / ggsav;
					dd = ds = ss = ZERO;
					for (i = 1; i <= n; ++i) {
						d[i - 1 + di] = temp * d[i - 1 + di] - g[i - 1 + gi] - hs[i - 1 + hsi];
						dd += (d[i - 1 + di] * d[i - 1 + di]);
						ds += (d[i - 1 + di] * step[i - 1]);
						ss += (step[i - 1] * step[i - 1]);
					}
					if (ds <= ZERO) {
						return;
					} else if (ss < delsq) {

						// Calculate the step to the trust region boundary
						// and the product HD
						++iterc;
						temp = delsq - ss;
						bstep = temp / (ds + Math.sqrt(ds * ds + dd * temp));
						continue;
					}
				}
				crvmin[0] = ZERO;
				itersw = iterc;

				// Test whether an alternative iteration is required
				if (gg <= 1.0e-4 * ggbeg) {
					return;
				}
				sg = shs = ZERO;
				for (i = 1; i <= n; ++i) {
					sg += (step[i - 1] * g[i - 1 + gi]);
					shs += (step[i - 1] * hs[i - 1 + hsi]);
				}
				sgk = sg + shs;
				angtest = sgk / Math.sqrt(gg * delsq);
				if (angtest <= -0.99) {
					return;
				}

				// Begin the alternative iteration by calculating D and HD and some
				// scalar products
				++iterc;
				temp = Math.sqrt(delsq * gg - sgk * sgk);
				tempa = delsq / temp;
				tempb = sgk / temp;
				for (i = 1; i <= n; ++i) {
					d[i - 1 + di] = tempa * (g[i - 1 + gi] + hs[i - 1 + hsi]) - tempb * step[i - 1];
				}
				continue;
			}
			dg = dhd = dhs = ZERO;
			for (i = 1; i <= n; ++i) {
				dg += (d[i - 1 + di] * g[i - 1 + gi]);
				dhd += (hd[i - 1 + hdi] * d[i - 1 + di]);
				dhs += (hd[i - 1 + hdi] * step[i - 1]);
			}

			// Seek the value of the angle that minimizes Q
			cf = HALF * (shs - dhd);
			qbeg = sg + cf;
			qsav = qmin = qbeg;
			isave = 0;
			iu = 49;
			temp = TWOPI / (iu + 1);
			for (i = 1; i <= iu; ++i) {
				angle = (double) i * temp;
				cth = Math.cos(angle);
				sth = Math.sin(angle);
				qnew = (sg + cf * cth) * cth + (dg + dhs * cth) * sth;
				if (qnew < qmin) {
					qmin = qnew;
					isave = i;
					tempa = qsav;
				} else if (i == isave + 1) {
					tempb = qnew;
				}
				qsav = qnew;
			}
			if (isave == 0) {
				tempa = qnew;
			}
			if (isave == iu) {
				tempb = qbeg;
			}
			angle = ZERO;
			if (tempa != tempb) {
				tempa -= qmin;
				tempb -= qmin;
				angle = HALF * (tempa - tempb) / (tempa + tempb);
			}
			angle = temp * ((double) isave + angle);

			// Calculate the new STEP and HS. Then test for convergence
			cth = Math.cos(angle);
			sth = Math.sin(angle);
			reduc = qbeg - (sg + cf * cth) * cth - (dg + dhs * cth) * sth;
			gg = ZERO;
			for (i = 1; i <= n; ++i) {
				step[i - 1] = cth * step[i - 1] + sth * d[i - 1 + di];
				hs[i - 1 + hsi] = cth * hs[i - 1 + hsi] + sth * hd[i - 1 + hdi];
				final double dst = g[i - 1 + gi] + hs[i - 1 + hsi];
				gg += (dst * dst);
			}
			qred += reduc;
			ratio = reduc / qred;
			if (iterc < itermax && ratio > 0.01) {

				// Test whether an alternative iteration is required
				if (gg <= 1.0e-4 * ggbeg) {
					return;
				}
				sg = shs = ZERO;
				for (i = 1; i <= n; ++i) {
					sg += (step[i - 1] * g[i - 1 + gi]);
					shs += (step[i - 1] * hs[i - 1 + hsi]);
				}
				sgk = sg + shs;
				angtest = sgk / Math.sqrt(gg * delsq);
				if (angtest <= -0.99) {
					return;
				}

				// Begin the alternative iteration by calculating D and HD and some
				// scalar products
				++iterc;
				temp = Math.sqrt(delsq * gg - sgk * sgk);
				tempa = delsq / temp;
				tempb = sgk / temp;
				for (i = 1; i <= n; ++i) {
					d[i - 1 + di] = tempa * (g[i - 1 + gi] + hs[i - 1 + hsi]) - tempb * step[i - 1];
				}
			} else {
				return;
			}
		}
	}

	private static void update(final int n, final int npt, final double[][] bmat, final double[][] zmat,
			final int[] idz, final int ndim, final double[] vlag, final double beta, final int knew, final double[] w) {

		// The arrays BMAT and ZMAT with IDZ are updated, in order to shift the
		// interpolation point that has index KNEW. On entry, VLAG contains the
		// components of the vector Theta*Wcheck+e_b of the updating formula
		// (6.11), and BETA holds the value of the parameter that has this name.
		// The vector W is used for working space
		int i, iflag, j, ja, jb, jl, jp;
		double alpha, tau, tausq, denom, scala, scalb, temp, tempa, tempb = 0.0;

		// Set some constants
		final double ONE = 1.0, ZERO = 0.0;
		final int nptm = npt - n - 1;

		// Apply the rotations that put zeros in the KNEW-th row of ZMAT
		jl = 1;
		for (j = 2; j <= nptm; ++j) {
			if (j == idz[0]) {
				jl = idz[0];
			} else if (zmat[knew - 1][j - 1] != ZERO) {
				temp = RealMath.hypot(zmat[knew - 1][jl - 1], zmat[knew - 1][j - 1]);
				tempa = zmat[knew - 1][jl - 1] / temp;
				tempb = zmat[knew - 1][j - 1] / temp;
				for (i = 1; i <= npt; ++i) {
					temp = tempa * zmat[i - 1][jl - 1] + tempb * zmat[i - 1][j - 1];
					zmat[i - 1][j - 1] = tempa * zmat[i - 1][j - 1] - tempb * zmat[i - 1][jl - 1];
					zmat[i - 1][jl - 1] = temp;
				}
				zmat[knew - 1][j - 1] = ZERO;
			}
		}

		// Put the first NPT components of the KNEW-th column of HLAG into W,
		// and calculate the parameters of the updating formula
		tempa = zmat[knew - 1][1 - 1];
		if (idz[0] >= 2) {
			tempa = -tempa;
		}
		if (jl > 1) {
			tempb = zmat[knew - 1][jl - 1];
		}
		for (i = 1; i <= npt; ++i) {
			w[i - 1] = tempa * zmat[i - 1][1 - 1];
			if (jl > 1) {
				w[i - 1] += (tempb * zmat[i - 1][jl - 1]);
			}
		}
		alpha = w[knew - 1];
		tau = vlag[knew - 1];
		tausq = tau * tau;
		denom = alpha * beta + tausq;
		vlag[knew - 1] -= ONE;

		// Complete the updating of ZMAT when there is only one nonzero element
		// in the KNEW-th row of the new matrix ZMAT, but, if IFLAG is set to one,
		// then the first column of ZMAT will be exchanged with another one later.
		iflag = 0;
		if (jl == 1) {
			temp = Math.sqrt(Math.abs(denom));
			tempb = tempa / temp;
			tempa = tau / temp;
			for (i = 1; i <= npt; ++i) {
				zmat[i - 1][1 - 1] = tempa * zmat[i - 1][1 - 1] - tempb * vlag[i - 1];
			}
			if (idz[0] == 1 && temp < ZERO) {
				idz[0] = 2;
			}
			if (idz[0] >= 2 && temp >= ZERO) {
				iflag = 1;
			}
		} else {

			// Complete the updating of ZMAT in the alternative case.
			ja = 1;
			if (beta >= ZERO) {
				ja = jl;
			}
			jb = jl + 1 - ja;
			temp = zmat[knew - 1][jb - 1] / denom;
			tempa = temp * beta;
			tempb = temp * tau;
			temp = zmat[knew - 1][ja - 1];
			scala = ONE / Math.sqrt(Math.abs(beta) * temp * temp + tausq);
			scalb = scala * Math.sqrt(Math.abs(denom));
			for (i = 1; i <= npt; ++i) {
				zmat[i - 1][ja - 1] = scala * (tau * zmat[i - 1][ja - 1] - temp * vlag[i - 1]);
				zmat[i - 1][jb - 1] = scalb * (zmat[i - 1][jb - 1] - tempa * w[i - 1] - tempb * vlag[i - 1]);
			}
			if (denom <= ZERO) {
				if (beta < ZERO) {
					++idz[0];
				} else {
					iflag = 1;
				}
			}
		}

		// IDZ is reduced in the following case, and usually the first column
		// of ZMAT is exchanged with a later one
		if (iflag == 1) {
			--idz[0];
			for (i = 1; i <= npt; ++i) {
				temp = zmat[i - 1][1 - 1];
				zmat[i - 1][1 - 1] = zmat[i - 1][idz[0] - 1];
				zmat[i - 1][idz[0] - 1] = temp;
			}
		}

		// Finally, update the matrix BMAT
		for (j = 1; j <= n; ++j) {
			jp = npt + j;
			w[jp - 1] = bmat[knew - 1][j - 1];
			tempa = (alpha * vlag[jp - 1] - tau * w[jp - 1]) / denom;
			tempb = (-beta * w[jp - 1] - tau * vlag[jp - 1]) / denom;
			for (i = 1; i <= jp; ++i) {
				bmat[i - 1][j - 1] += (tempa * vlag[i - 1] + tempb * w[i - 1]);
				if (i > npt) {
					bmat[jp - 1][i - npt - 1] = bmat[i - 1][j - 1];
				}
			}
		}
	}
}
