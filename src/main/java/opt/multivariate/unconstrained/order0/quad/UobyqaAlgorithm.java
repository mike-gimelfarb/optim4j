/*
UOBYQA---Unconstrained Optimization BY Quadratic Approximation.
Copyright (C) 2000 M. J. D. Powell (University of Cambridge)

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

import opt.multivariate.unconstrained.order0.GradientFreeOptimizer;
import utils.BlasMath;
import utils.RealMath;

/**
 *
 * REFERENCES:
 * 
 * [1] Powell, Michael JD. "UOBYQA: unconstrained optimization by quadratic
 * approximation." Mathematical Programming 92.3 (2002): 555-582.
 */
public final class UobyqaAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
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
	 */
	public UobyqaAlgorithm(final double tolerance, final double initialStep, final int maxEvaluations) {
		super(tolerance);
		myRho0 = initialStep;
		myMaxFev = maxEvaluations;
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

		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		final int[] fev = new int[1];

		// call main subroutine
		uobyqa1(func, n, x, myMaxFev, myRho0, myTol, fev);
		myEvals += fev[0];
		return x;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static void uobyqa1(final Function<? super double[], Double> func, final int n, final double[] x,
			final int maxfev, final double rhobeg, final double rhoend, final int[] fev) {
		final int iprint = 0, npt = (n * n + 3 * n + 2) / 2;
		final double[][] pl = new double[npt][npt], h = new double[n][n], xpt = new double[npt][n];
		final double[] xbase = new double[n], xopt = new double[n], xnew = new double[n], pq = new double[npt],
				g = new double[n], d = new double[n], vlag = new double[npt], w = new double[Math.max(6 * n, npt)];
		fev[0] = 0;

		uobyqb(func, n, x, rhobeg, rhoend, iprint, maxfev, npt, xbase, xopt, xnew, xpt, pq, pl, h, g, d, vlag, w, fev);
	}

	private static void uobyqb(final Function<? super double[], Double> func, final int n, final double[] x,
			final double rhobeg, final double rhoend, final int iprint, final int maxfun, final int npt,
			final double[] xbase, final double[] xopt, final double[] xnew, final double[][] xpt, final double[] pq,
			final double[][] pl, final double[][] h, final double[] g, final double[] d, final double[] vlag,
			final double[] w, final int[] fev) {

		final double[] empty = new double[npt], evalue = new double[1], vmax = new double[1];
		final double one, two, zero, half, tol;
		final int nnp, nptm, nftest;
		double delta = 0.0, detrat, distest, ddknew = 0.0, dnorm = 0.0, diff = 0.0, errtol = 0.0, estim, rho, rhosq,
				tworsq = 0.0, sixthm = 0.0, f = 0.0, fbase = 0.0, fopt = 0.0, fsave = 0.0, ratio, sum, sumh, sumg,
				temp = 0.0, tempa, vquad, wmult = 0.0;
		int nf, i, ih = 0, ip = 0, iq = 0, iw, j = 0, jswitch = 0, k, knew = 0, ksave = 0, ktemp, kopt = 0;

		// Set some constants
		one = 1.0;
		two = 2.0;
		zero = 0.0;
		half = 0.5;
		tol = 0.01;
		nnp = n + n + 1;
		nptm = npt - 1;
		nftest = Math.max(maxfun, 1);

		// Initialization. NF is the number of function calculations so far
		rho = rhobeg;
		rhosq = rho * rho;
		nf = 0;
		System.arraycopy(xbase, 0, x, 0, n);
		for (k = 1; k <= npt; ++k) {
			System.arraycopy(empty, 0, pl[k - 1], 0, nptm);
			j = nptm;
		}

		int flag = 30;
		while (true) {

			if (flag == 30) {

				// The branch to label 120 obtains a new value
				// of the objective function...
				BlasMath.dxpy1(n, xbase, 1, xpt[nf + 1 - 1], 1, x, 1);
				flag = 120;
			}

			if (flag == 50) {
				if (nf == 1) {
					fopt = f;
					kopt = nf;
					fbase = f;
					j = 0;
					jswitch = -1;
					ih = n;
				} else if (f < fopt) {
					fopt = f;
					kopt = nf;
				}

				// Form the gradient and diagonal second derivatives...
				if (nf <= nnp) {
					jswitch = -jswitch;
					if (jswitch > 0) {
						if (j >= 1) {
							ih += j;
							if (w[j - 1] < zero) {
								d[j - 1] = (fsave + f - two * fbase) / rhosq;
								pq[j - 1] = (fsave - f) / (two * rho);
								pl[1 - 1][ih - 1] = -two / rhosq;
								pl[nf - 1 - 1][j - 1] = half / rho;
								pl[nf - 1 - 1][ih - 1] = one / rhosq;
							} else {
								pq[j - 1] = (4.0 * fsave - 3.0 * fbase - f) / (two * rho);
								d[j - 1] = (fbase + f - two * fsave) / rhosq;
								pl[1 - 1][j - 1] = -1.5 / rho;
								pl[1 - 1][ih - 1] = one / rhosq;
								pl[nf - 1 - 1][j - 1] = two / rho;
								pl[nf - 1 - 1][ih - 1] = -two / rhosq;
							}
							pq[ih - 1] = d[j - 1];
							pl[nf - 1][j - 1] = -half / rho;
							pl[nf - 1][ih - 1] = one / rhosq;
						}

						// Pick the shift from XBASE to the next initial
						// interpolation point...
						if (j < n) {
							++j;
							xpt[nf + 1 - 1][j - 1] = rho;
						}
					} else {
						fsave = f;
						if (f < fbase) {
							w[j - 1] = rho;
							xpt[nf + 1 - 1][j - 1] = two * rho;
						} else {
							w[j - 1] = -rho;
							xpt[nf + 1 - 1][j - 1] = -rho;
						}
					}
					if (nf < nnp) {
						flag = 30;
						continue;
					}

					// Form the off-diagonal second derivatives of the
					// initial quadratic model
					ih = n;
					ip = 1;
					iq = 2;
				}

				++ih;
				if (nf > nnp) {
					temp = one / (w[ip - 1] * w[iq - 1]);
					tempa = f - fbase - w[ip - 1] * pq[ip - 1] - w[iq - 1] * pq[iq - 1];
					pq[ih - 1] = (tempa - half * rhosq * (d[ip - 1] + d[iq - 1])) * temp;
					pl[1 - 1][ih - 1] = temp;
					iw = ip + ip;
					if (w[ip - 1] < zero) {
						++iw;
					}
					pl[iw - 1][ih - 1] = -temp;
					iw = iq + iq;
					if (w[iq - 1] < zero) {
						++iw;
					}
					pl[iw - 1][ih - 1] = -temp;
					pl[nf - 1][ih - 1] = temp;

					// Pick the shift from XBASE to the next initial
					// interpolation point
					++ip;
				}
				if (ip == iq) {
					++ih;
					ip = 1;
					++iq;
				}
				if (nf < npt) {
					xpt[nf + 1 - 1][ip - 1] = w[ip - 1];
					xpt[nf + 1 - 1][iq - 1] = w[iq - 1];
					flag = 30;
					continue;
				}

				// Set parameters to begin the iterations for the current RHO
				sixthm = zero;
				delta = rho;
				tworsq = (two * rho) * (two * rho);
				rhosq = rho * rho;
				flag = 70;
			}

			if (flag == 70) {

				// Form the gradient of the quadratic model
				// at the trust region centre
				knew = 0;
				ih = n;
				System.arraycopy(xpt[kopt - 1], 0, xopt, 0, n);
				System.arraycopy(pq, 0, g, 0, n);
				for (j = 1; j <= n; ++j) {
					for (i = 1; i <= j; ++i) {
						++ih;
						g[i - 1] += (pq[ih - 1] * xopt[j - 1]);
						if (i < j) {
							g[j - 1] += (pq[ih - 1] * xopt[i - 1]);
						}
						h[i - 1][j - 1] = pq[ih - 1];
					}
				}

				// Generate the next trust region step and test its length
				trstep(n, g, h, delta, tol, d, w, 1, w, n + 1, w, 2 * n + 1, w, 3 * n + 1, w, 4 * n + 1, w, 5 * n + 1,
						evalue, empty);
				temp = BlasMath.ddotm(n, d, 1, d, 1);
				dnorm = Math.min(delta, Math.sqrt(temp));
				errtol = -one;
				if (dnorm < half * rho) {
					knew = -1;
					errtol = half * evalue[0] * rho * rho;
					if (nf <= npt + 9) {
						errtol = zero;
					}
					flag = 290;
				} else {

					// Calculate the next value of the objective function
					for (i = 1; i <= n; ++i) {
						xnew[i - 1] = xopt[i - 1] + d[i - 1];
						x[i - 1] = xbase[i - 1] + xnew[i - 1];
					}
					flag = 120;
				}
			}

			if (flag == 120) {
				if (nf >= nftest) {
					if (fopt <= f) {
						BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
					}
					return;
				}
				++nf;
				f = func.apply(x);
				++fev[0];
				if (nf <= npt) {
					flag = 50;
					continue;
				}
				if (knew == -1) {
					if (fopt <= f) {
						BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
					}
					return;
				}

				// Use the quadratic model to predict the change in F
				// due to the step D...
				vquad = zero;
				ih = n;
				for (j = 1; j <= n; ++j) {
					w[j - 1] = d[j - 1];
					vquad += (w[j - 1] * pq[j - 1]);
					for (i = 1; i <= j; ++i) {
						++ih;
						w[ih - 1] = d[i - 1] * xnew[j - 1] + d[j - 1] * xopt[i - 1];
						if (i == j) {
							w[ih - 1] *= half;
						}
						vquad += (w[ih - 1] * pq[ih - 1]);
					}
				}
				for (k = 1; k <= npt; ++k) {
					temp = BlasMath.ddotm(nptm, w, 1, pl[k - 1], 1);
					vlag[k - 1] = temp;
				}
				vlag[kopt - 1] += one;

				// Update SIXTHM...
				diff = f - fopt - vquad;
				sum = zero;
				for (k = 1; k <= npt; ++k) {
					temp = zero;
					for (i = 1; i <= n; ++i) {
						final double dst = xpt[k - 1][i - 1] - xnew[i - 1];
						temp += (dst * dst);
					}
					temp = Math.sqrt(temp);
					sum += Math.abs(temp * temp * temp * vlag[k - 1]);
				}
				sixthm = Math.max(sixthm, Math.abs(diff) / sum);

				// Update FOPT and XOPT if the new F is the least value
				// of the objective...
				fsave = fopt;
				if (f < fopt) {
					fopt = f;
					System.arraycopy(xnew, 0, xopt, 0, n);
				}
				ksave = knew;
				if (knew > 0) {
					flag = 240;
				} else {

					// Pick the next value of DELTA after a trust region step
					if (vquad >= zero) {
						if (fopt <= f) {
							BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
						}
						return;
					}
					ratio = (f - fsave) / vquad;
					if (ratio <= 0.1) {
						delta = half * dnorm;
					} else if (ratio <= 0.7) {
						delta = Math.max(half * delta, dnorm);
					} else {
						delta = Math.max(delta, Math.max(1.25 * dnorm, dnorm + rho));
					}
					if (delta <= 1.5 * rho) {
						delta = rho;
					}

					// Set KNEW to the index of the next interpolation point
					// to be deleted
					ktemp = 0;
					detrat = zero;
					if (f >= fsave) {
						ktemp = kopt;
						detrat = one;
					}
					for (k = 1; k <= npt; ++k) {
						sum = zero;
						for (i = 1; i <= n; ++i) {
							final double dst = xpt[k - 1][i - 1] - xopt[i - 1];
							sum += (dst * dst);
						}
						temp = Math.abs(vlag[k - 1]);
						if (sum > rhosq) {
							temp *= Math.pow(sum / rhosq, 1.5);
						}
						if (temp > detrat && k != ktemp) {
							detrat = temp;
							ddknew = sum;
							knew = k;
						}
					}
					if (knew == 0) {
						flag = 290;
					} else {
						flag = 240;
					}
				}
			}

			if (flag == 240) {

				// Replace the interpolation point that has index KNEW...
				System.arraycopy(xnew, 0, xpt[knew - 1], 0, n);
				temp = one / vlag[knew - 1];
				for (j = 1; j <= nptm; ++j) {
					pl[knew - 1][j - 1] = temp * pl[knew - 1][j - 1];
					pq[j - 1] += (diff * pl[knew - 1][j - 1]);
				}
				for (k = 1; k <= npt; ++k) {
					if (k != knew) {
						temp = vlag[k - 1];
						BlasMath.daxpym(nptm, -temp, pl[knew - 1], 1, pl[k - 1], 1);
					}
				}

				// Update KOPT if F is the least calculated value of the objective...
				if (f < fsave) {
					kopt = knew;
					flag = 70;
					continue;
				}
				if (ksave > 0 || dnorm > two * rho || ddknew > tworsq) {
					flag = 70;
					continue;
				}
				flag = 290;
			}

			// Alternatively, find out if the interpolation points are close
			// enough...
			for (k = 1; k <= npt; ++k) {
				w[k - 1] = zero;
				for (i = 1; i <= n; ++i) {
					final double dst = xpt[k - 1][i - 1] - xopt[i - 1];
					w[k - 1] += (dst * dst);
				}
			}

			while (true) {
				knew = -1;
				distest = tworsq;
				for (k = 1; k <= npt; ++k) {
					if (w[k - 1] > distest) {
						knew = k;
						distest = w[k - 1];
					}
				}

				// If a point is sufficiently far away,
				// then set the gradient and Hessian...
				if (knew > 0) {
					ih = n;
					sumh = zero;
					for (j = 1; j <= n; ++j) {
						g[j - 1] = pl[knew - 1][j - 1];
						for (i = 1; i <= j; ++i) {
							++ih;
							temp = pl[knew - 1][ih - 1];
							g[j - 1] += (temp * xopt[i - 1]);
							if (i < j) {
								g[i - 1] += (temp * xopt[j - 1]);
								sumh += (temp * temp);
							}
							h[i - 1][j - 1] = temp;
						}
						sumh += (half * temp * temp);
					}

					// If ERRTOL is positive, test whether to
					// replace the interpolation point...
					if (errtol > zero) {
						w[knew - 1] = zero;
						sumg = BlasMath.ddotm(n, g, 1, g, 1);
						estim = rho * (Math.sqrt(sumg) + rho * Math.sqrt(half * sumh));
						wmult = sixthm * Math.pow(distest, 1.5);
						if (wmult * estim <= errtol) {
							continue;
						}
					}

					// If the KNEW-th point may be replaced, then pick a D...
					lagmax(n, g, h, rho, d, xnew, vmax);
					if (errtol > zero) {
						if (wmult * vmax[0] <= errtol) {
							continue;
						}
					}

					// Calculate the next value of the objective function
					for (i = 1; i <= n; ++i) {
						xnew[i - 1] = xopt[i - 1] + d[i - 1];
						x[i - 1] = xbase[i - 1] + xnew[i - 1];
					}
					flag = 120;
				}
				break;
			}
			if (flag == 120) {
				continue;
			}

			if (dnorm > rho) {
				flag = 70;
				continue;
			}

			// Prepare to reduce RHO by shifting XBASE to the best point so far...
			if (rho > rhoend) {
				ih = n;
				for (j = 1; j <= n; ++j) {
					xbase[j - 1] += xopt[j - 1];
					for (k = 1; k <= npt; ++k) {
						xpt[k - 1][j - 1] -= xopt[j - 1];
					}
					for (i = 1; i <= j; ++i) {
						++ih;
						pq[i - 1] += (pq[ih - 1] * xopt[j - 1]);
						if (i < j) {
							pq[j - 1] += (pq[ih - 1] * xopt[i - 1]);
							for (k = 1; k <= npt; ++k) {
								pl[k - 1][j - 1] += (pl[k - 1][ih - 1] * xopt[i - 1]);
							}
						}
						for (k = 1; k <= npt; ++k) {
							pl[k - 1][i - 1] += (pl[k - 1][ih - 1] * xopt[j - 1]);
						}
					}
				}

				// Pick the next values of RHO and DELTA
				delta = half * rho;
				ratio = rho / rhoend;
				if (ratio <= 16.0) {
					rho = rhoend;
				} else if (ratio <= 250.0) {
					rho = Math.sqrt(ratio) * rhoend;
				} else {
					rho *= 0.1;
				}
				delta = Math.max(delta, rho);
				tworsq = (two * rho) * (two * rho);
				rhosq = rho * rho;
				flag = 70;
				continue;
			}

			// Return from the calculation, after another Newton-Raphson step...
			if (errtol >= zero) {

				// Calculate the next value of the objective function
				for (i = 1; i <= n; ++i) {
					xnew[i - 1] = xopt[i - 1] + d[i - 1];
					x[i - 1] = xbase[i - 1] + xnew[i - 1];
				}
				flag = 120;
			} else {
				break;
			}
		}

		if (fopt <= f) {
			BlasMath.dxpy1(n, xbase, 1, xopt, 1, x, 1);
		}
	}

	private static void trstep(final int n, final double[] g, final double[][] h, final double delta, final double tol,
			final double[] d, final double[] gg, final int igg, final double[] td, final int itd, final double[] tn,
			final int itn, final double[] w, final int iw, final double[] piv, final int ipiv, final double[] z,
			final int iz, final double[] evalue, final double[] empty) {

		final double one = 1.0, two = 2.0, zero = 0.0;
		double dnorm, delsq, dsq, dtg, dtz, dhd, gnorm, hnorm, gam, gsq, parl, parlest, par, paru, paruest, posdef, phi,
				phil, phiu, pivksv, pivot, scale, slope, sum, shfmin, shfmax, shift, temp, tempa, tempb, tdmin, wz, wsq,
				wwsq, zsq;
		int i, iterc, j, jp, k, kp, kpp, ksav, ksave, nm;

		// added to avoid "noninitialization"
		dsq = pivksv = phil = phiu = zero;

		// Initialization
		delsq = delta * delta;
		evalue[0] = zero;
		nm = n - 1;
		System.arraycopy(empty, 0, d, 0, n);
		for (i = 1; i <= n; ++i) {
			td[i - 1 + itd - 1] = h[i - 1][i - 1];
			for (j = 1; j <= i; ++j) {
				h[i - 1][j - 1] = h[j - 1][i - 1];
			}
		}

		// Apply Householder transformations to obtain a tridiagonal matrix...
		for (k = 1; k <= nm; ++k) {
			kp = k + 1;
			sum = zero;
			if (kp < n) {
				kpp = kp + 1;
				for (i = kpp; i <= n; ++i) {
					sum += (h[i - 1][k - 1] * h[i - 1][k - 1]);
				}
			}
			if (sum == zero) {
				tn[k - 1 + itn - 1] = h[kp - 1][k - 1];
				h[kp - 1][k - 1] = zero;
			} else {
				temp = h[kp - 1][k - 1];
				tn[k - 1 + itn - 1] = RealMath.sign(Math.sqrt(sum + temp * temp), temp);
				h[kp - 1][k - 1] = -sum / (temp + tn[k - 1 + itn - 1]);
				temp = Math.sqrt(two / (sum + h[kp - 1][k - 1] * h[kp - 1][k - 1]));
				for (i = kp; i <= n; ++i) {
					w[i - 1 + iw - 1] = temp * h[i - 1][k - 1];
					h[i - 1][k - 1] = w[i - 1 + iw - 1];
					z[i - 1 + iz - 1] = td[i - 1 + itd - 1] * w[i - 1 + iw - 1];
				}
				wz = zero;
				for (j = kp; j <= nm; ++j) {
					jp = j + 1;
					for (i = jp; i <= n; ++i) {
						z[i - 1 + iz - 1] += (h[i - 1][j - 1] * w[j - 1 + iw - 1]);
						z[j - 1 + iz - 1] += (h[i - 1][j - 1] * w[i - 1 + iw - 1]);
					}
					wz += (w[j - 1 + iw - 1] * z[j - 1 + iz - 1]);
				}
				wz += (w[n - 1 + iw - 1] * z[n - 1 + iz - 1]);
				for (j = kp; j <= n; ++j) {
					td[j - 1 + itd - 1] += (w[j - 1 + iw - 1] * (wz * w[j - 1 + iw - 1] - two * z[j - 1 + iz - 1]));
					if (j < n) {
						jp = j + 1;
						for (i = jp; i <= n; ++i) {
							h[i - 1][j - 1] = h[i - 1][j - 1] - w[i - 1 + iw - 1] * z[j - 1 + iz - 1]
									- w[j - 1 + iw - 1] * (z[i - 1 + iz - 1] - wz * w[i - 1 + iw - 1]);
						}
					}
				}
			}
		}

		// Form GG by applying the similarity transformation to G
		System.arraycopy(g, 0, gg, igg - 1, n);
		gsq = BlasMath.ddotm(n, g, 1, g, 1);
		gnorm = Math.sqrt(gsq);
		for (k = 1; k <= nm; ++k) {
			kp = k + 1;
			sum = zero;
			for (i = kp; i <= n; ++i) {
				sum += (gg[i - 1 + igg - 1] * h[i - 1][k - 1]);
			}
			for (i = kp; i <= n; ++i) {
				gg[i - 1 + igg - 1] -= (sum * h[i - 1][k - 1]);
			}
		}

		// Begin the trust region calculation with a tridiagonal matrix...
		hnorm = Math.abs(td[1 - 1 + itd - 1]) + Math.abs(tn[1 - 1 + itn - 1]);
		tdmin = td[1 - 1 + itd - 1];
		tn[n - 1 + itn - 1] = zero;
		for (i = 2; i <= n; ++i) {
			temp = Math.abs(tn[i - 1 - 1 + itn - 1]) + Math.abs(td[i - 1 + itd - 1]) + Math.abs(tn[i - 1 + itn - 1]);
			hnorm = Math.max(hnorm, temp);
			tdmin = Math.min(tdmin, td[i - 1 + itd - 1]);
		}
		if (hnorm == zero) {
			if (gnorm == zero) {

				// Return from the subroutine
				return;
			}
			scale = delta / gnorm;
			BlasMath.daxpym(n, -scale, gg, igg, d, 1);

			// Apply the inverse Householder transformations to D
			nm = n - 1;
			for (k = nm; k >= 1; --k) {
				kp = k + 1;
				sum = zero;
				for (i = kp; i <= n; ++i) {
					sum += (d[i - 1] * h[i - 1][k - 1]);
				}
				for (i = kp; i <= n; ++i) {
					d[i - 1] -= (sum * h[i - 1][k - 1]);
				}
			}

			// Return from the subroutine
			return;
		}

		// Set the initial values of PAR and its bounds
		parl = Math.max(zero, Math.max(-tdmin, gnorm / delta - hnorm));
		parlest = parl;
		par = parl;
		paru = paruest = posdef = zero;
		iterc = 0;

		while (true) {

			// Calculate the pivots of the Cholesky factorization of (H+PAR*I)
			++iterc;
			ksav = 0;
			piv[1 - 1 + ipiv - 1] = td[1 - 1 + itd - 1] + par;
			k = 1;

			boolean skipto160 = false;
			while (true) {
				if (piv[k - 1 + ipiv - 1] > zero) {
					piv[k + 1 - 1 + ipiv - 1] = td[k + 1 - 1 + itd - 1] + par
							- tn[k - 1 + itn - 1] * tn[k - 1 + itn - 1] / piv[k - 1 + ipiv - 1];
				} else {
					if (piv[k - 1 + ipiv - 1] < zero || tn[k - 1 + itn - 1] != zero) {
						skipto160 = true;
						break;
					}
					ksav = k;
					piv[k + 1 - 1 + ipiv - 1] = td[k + 1 - 1 + itd - 1] + par;
				}
				++k;
				if (k >= n) {
					break;
				}
			}

			if (skipto160 || piv[k - 1 + ipiv - 1] < zero) {
				skipto160 = true;
			}

			if (!skipto160 && piv[k - 1 + ipiv - 1] == zero) {
				ksav = k;
			}

			// Branch if all the pivots are positive...
			if (skipto160 || (ksav != 0 || gsq <= zero)) {

				boolean skipto190 = false;
				if (!skipto160) {
					if (gsq == zero) {
						if (par == zero) {
							break; // exits to 370
						}
						paru = par;
						paruest = par;
						if (ksav == 0) {
							skipto190 = true;
						}
					}
					if (!skipto190) {
						k = ksav;
					}
				}

				if (!skipto190) {

					// Set D to a direction of nonpositive curvature...
					d[k - 1] = one;
					final double temp1 = Math.abs(piv[k - 1 + ipiv - 1]);
					if (Math.abs(tn[k - 1 + itn - 1]) <= temp1) {
						dsq = one;
						dhd = piv[k - 1 + ipiv - 1];
					} else {
						temp = td[k + 1 - 1 + itd - 1] + par;
						if (temp <= Math.abs(piv[k - 1 + ipiv - 1])) {
							d[k + 1 - 1] = RealMath.sign(one, -tn[k - 1 + itn - 1]);
							dhd = piv[k - 1 + ipiv - 1] + temp - two * Math.abs(tn[k - 1 + itn - 1]);
						} else {
							d[k + 1 - 1] = -tn[k - 1 + itn - 1] / temp;
							dhd = piv[k - 1 + ipiv - 1] + tn[k - 1 + itn - 1] * d[k + 1 - 1];
						}
						dsq = one + d[k + 1 - 1] * d[k + 1 - 1];
					}

					while (k > 1) {
						--k;
						if (tn[k - 1 + itn - 1] != zero) {
							d[k - 1] = -tn[k - 1 + itn - 1] * d[k + 1 - 1] / piv[k - 1 + ipiv - 1];
							dsq += d[k - 1] * d[k - 1];
						} else {
							System.arraycopy(empty, 0, d, 0, k);
							break;
						}
					}
					parl = par;
					parlest = par - dhd / dsq;
				}

				// Terminate with D set to a multiple of the current D...
				temp = paruest;
				if (gsq == 0.0) {
					temp *= (one - tol);
				}
				if (paruest > zero && parlest >= temp) {
					dtg = BlasMath.ddotm(n, d, 1, gg, igg);
					scale = -RealMath.sign(delta / Math.sqrt(dsq), dtg);
					BlasMath.dscalm(n, scale, d, 1);
					break; // exit to 370
				}

				// Pick the value of PAR for the next iteration
				if (paru == zero) {
					par = two * parlest + gnorm / delta;
				} else {
					par = 0.5 * (parl + paru);
					par = Math.max(par, parlest);
				}
				if (paruest > zero) {
					par = Math.min(par, paruest);
				}
				continue; // continue from 140
			}

			// Calculate D for the current PAR in the positive definite case
			w[1 - 1 + iw - 1] = -gg[1 - 1 + igg - 1] / piv[1 - 1 + ipiv - 1];
			for (i = 2; i <= n; ++i) {
				w[i - 1 + iw - 1] = (-gg[i - 1 + igg - 1] - tn[i - 1 - 1 + itn - 1] * w[i - 1 - 1 + iw - 1])
						/ piv[i - 1 + ipiv - 1];
			}
			d[n - 1] = w[n - 1 + iw - 1];
			for (i = nm; i >= 1; --i) {
				d[i - 1] = w[i - 1 + iw - 1] - tn[i - 1 + itn - 1] * d[i + 1 - 1] / piv[i - 1 + ipiv - 1];
			}

			// Branch if a Newton-Raphson step is acceptable
			dsq = wsq = zero;
			for (i = 1; i <= n; ++i) {
				dsq += (d[i - 1] * d[i - 1]);
				wsq += (piv[i - 1 + ipiv - 1] * w[i - 1 + iw - 1] * w[i - 1 + iw - 1]);
			}
			if (par == zero && dsq <= delsq) {

				// Set EVALUE to the least eigenvalue of the
				// second derivative matrix...
				shfmin = zero;
				pivot = td[1 - 1 + itd - 1];
				shfmax = pivot;
				for (k = 2; k <= n; ++k) {
					pivot = td[k - 1 + itd - 1] - tn[k - 1 - 1 + itn - 1] * tn[k - 1 - 1 + itn - 1] / pivot;
					shfmax = Math.min(shfmax, pivot);
				}

				// Find EVALUE by a bisection method...
				ksave = 0;
				while (true) {
					shift = 0.5 * (shfmin + shfmax);
					k = 1;
					temp = td[1 - 1 + itd - 1] - shift;

					boolean skipto370 = false;
					while (true) {
						if (temp > zero) {
							piv[k - 1 + ipiv - 1] = temp;
							if (k < n) {
								temp = td[k + 1 - 1 + itd - 1] - shift
										- tn[k - 1 + itn - 1] * tn[k - 1 + itn - 1] / temp;
								++k;
								continue;
							} else {
								shfmin = shift;
							}
						} else {
							if (k < ksave) {
								evalue[0] = shfmin;
								skipto370 = true;
								break;
							}
							if (k == ksave) {
								if (pivksv == zero) {
									evalue[0] = shfmin;
									skipto370 = true;
									break;
								}
								if (piv[k - 1 + ipiv - 1] - temp < temp - pivksv) {
									pivksv = temp;
									shfmax = shift;
								} else {
									pivksv = zero;
									shfmax = (shift * piv[k - 1 + ipiv - 1] - shfmin * temp)
											/ (piv[k - 1 + ipiv - 1] - temp);
								}
							} else {
								ksave = k;
								pivksv = temp;
								shfmax = shift;
							}
						}
						break;
					}
					if (skipto370) {
						break;
					}
					if (shfmin > 0.99 * shfmax) {
						evalue[0] = shfmin;
						break;
					}
				}
				break; // exits to 370
			}

			// Make the usual test for acceptability of a full trust region step
			dnorm = Math.sqrt(dsq);
			phi = one / dnorm - one / delta;
			temp = tol * (one + par * dsq / wsq) - dsq * phi * phi;
			if (temp >= zero) {
				scale = delta / dnorm;
				BlasMath.dscalm(n, scale, d, 1);
				break; // exits to 370
			}
			if ((iterc >= 2 && par <= parl) || (paru > zero && par >= paru)) {
				break; // exits to 370
			}

			// Complete the iteration when PHI is negative
			if (phi < zero) {
				parlest = par;
				if (posdef == one) {
					if (phi <= phil) {
						break; // exits to 370
					}
					slope = (phi - phil) / (par - parl);
					parlest = par - phi / slope;
				}
				slope = one / gnorm;
				if (paru > zero) {
					slope = (phiu - phi) / (paru - par);
				}
				temp = par - phi / slope;
				if (paruest > zero) {
					temp = Math.min(temp, paruest);
				}
				paruest = temp;
				posdef = one;
				parl = par;
				phil = phi;

				// Pick the value of PAR for the next iteration
				if (paru == zero) {
					par = two * parlest + gnorm / delta;
				} else {
					par = 0.5 * (parl + paru);
					par = Math.max(par, parlest);
				}
				if (paruest > zero) {
					par = Math.min(par, paruest);
				}
				continue; // continue from 140
			}

			// If required, calculate Z for the alternative test for convergence
			if (posdef == zero) {
				w[1 - 1 + iw - 1] = one / piv[1 - 1 + ipiv - 1];
				for (i = 2; i <= n; ++i) {
					temp = -tn[i - 1 - 1 + itn - 1] * w[i - 1 - 1 + iw - 1];
					w[i - 1 + iw - 1] = (RealMath.sign(one, temp) + temp) / piv[i - 1 + ipiv - 1];
				}
				z[n - 1 + iz - 1] = w[n - 1 + iw - 1];
				for (i = nm; i >= 1; --i) {
					z[i - 1 + iz - 1] = w[i - 1 + iw - 1]
							- tn[i - 1 + itn - 1] * z[i + 1 - 1 + iz - 1] / piv[i - 1 + ipiv - 1];
				}
				wwsq = zsq = dtz = zero;
				for (i = 1; i <= n; ++i) {
					wwsq += (piv[i - 1 + ipiv - 1] * w[i - 1 + iw - 1] * w[i - 1 + iw - 1]);
					zsq += (z[i - 1 + iz - 1] * z[i - 1 + iz - 1]);
					dtz += (d[i - 1] * z[i - 1 + iz - 1]);
				}

				// Apply the alternative test for convergence
				tempa = Math.abs(delsq - dsq);
				tempb = Math.sqrt(dtz * dtz + tempa * zsq);
				gam = tempa / (RealMath.sign(tempb, dtz) + dtz);
				temp = tol * (wsq + par * delsq) - gam * gam * wwsq;
				if (temp >= zero) {
					BlasMath.daxpym(n, gam, z, iz, d, 1);
					break; // exits to 370
				}
				parlest = Math.max(parlest, par - wwsq / zsq);
			}

			// Complete the iteration when PHI is positive
			slope = one / gnorm;
			if (paru > zero) {
				if (phi >= phiu) {
					break; // exits to 370
				}
				slope = (phiu - phi) / (paru - par);
			}
			parlest = Math.max(parlest, par - phi / slope);
			paruest = par;
			if (posdef == one) {
				slope = (phi - phil) / (par - parl);
				paruest = par - phi / slope;
			}
			paru = par;
			phiu = phi;

			// Pick the value of PAR for the next iteration
			if (paru == zero) {
				par = two * parlest + gnorm / delta;
			} else {
				par = 0.5 * (parl + paru);
				par = Math.max(par, parlest);
			}
			if (paruest > zero) {
				par = Math.min(par, paruest);
			}
			// continue from 140
		}

		// Apply the inverse Householder transformations to D
		nm = n - 1;
		for (k = nm; k >= 1; --k) {
			kp = k + 1;
			sum = zero;
			for (i = kp; i <= n; ++i) {
				sum += (d[i - 1] * h[i - 1][k - 1]);
			}
			for (i = kp; i <= n; ++i) {
				d[i - 1] -= (sum * h[i - 1][k - 1]);
			}
		}
	}

	private static void lagmax(final int n, final double[] g, final double[][] h, final double rho, final double[] d,
			final double[] v, final double[] vmax) {

		final double half, halfrt, one, zero;
		double dd, dhd, dlin, dsq, gd, gg, ghg, gnorm, hmax, ratio, scale, sum, sumv, temp, tempa, tempb, tempc, tempd,
				tempv, vsq, vhg, vhv, vhw, vnorm, vmu, vv, vlin, wcos, wsin, whw, wsq;
		int i, j, k = 0;

		// Preliminary calculations
		half = 0.5;
		halfrt = Math.sqrt(half);
		one = 1.0;
		zero = 0.0;

		// Pick V such that ||HV|| / ||V|| is large
		hmax = zero;
		for (i = 1; i <= n; ++i) {
			for (j = 1; j <= n; ++j) {
				h[j - 1][i - 1] = h[i - 1][j - 1];
			}
			sum = BlasMath.ddotm(n, h[i - 1], 1, h[i - 1], 1);
			if (sum > hmax) {
				hmax = sum;
				k = i;
			}
		}
		System.arraycopy(h[k - 1], 0, v, 0, n);

		// Set D to a vector in the subspace spanned by V and HV that maximizes
		// |(D,HD)|/(D,D)...
		vsq = vhv = dsq = zero;
		for (i = 1; i <= n; ++i) {
			vsq += (v[i - 1] * v[i - 1]);
			d[i - 1] = BlasMath.ddotm(n, h[i - 1], 1, v, 1);
			vhv += (v[i - 1] * d[i - 1]);
			dsq += (d[i - 1] * d[i - 1]);
		}
		if (vhv * vhv <= 0.9999 * dsq * vsq) {
			temp = vhv / vsq;
			wsq = zero;
			for (i = 1; i <= n; ++i) {
				d[i - 1] -= (temp * v[i - 1]);
				wsq += (d[i - 1] * d[i - 1]);
			}
			whw = zero;
			ratio = Math.sqrt(wsq / vsq);
			for (i = 1; i <= n; ++i) {
				temp = BlasMath.ddotm(n, h[i - 1], 1, d, 1);
				whw += (temp * d[i - 1]);
				v[i - 1] *= ratio;
			}
			vhv *= (ratio * ratio);
			vhw = ratio * wsq;
			temp = half * (whw - vhv);
			temp += RealMath.sign(RealMath.hypot(temp, vhw), whw + vhv);
			for (i = 1; i <= n; ++i) {
				d[i - 1] = vhw * v[i - 1] + temp * d[i - 1];
			}
		}

		// We now turn our attention to the subspace spanned by G and D...
		gg = gd = dd = dhd = zero;
		for (i = 1; i <= n; ++i) {
			gg += (g[i - 1] * g[i - 1]);
			gd += (g[i - 1] * d[i - 1]);
			dd += (d[i - 1] * d[i - 1]);
			sum = BlasMath.ddotm(n, h[i - 1], 1, d, 1);
			dhd += (sum * d[i - 1]);
		}
		temp = gd / gg;
		vv = zero;
		scale = RealMath.sign(rho / Math.sqrt(dd), gd * dhd);
		for (i = 1; i <= n; ++i) {
			v[i - 1] = d[i - 1] - temp * g[i - 1];
			vv += (v[i - 1] * v[i - 1]);
			d[i - 1] *= scale;
		}
		gnorm = Math.sqrt(gg);
		if (gnorm * dd <= 0.5e-2 * rho * Math.abs(dhd) || vv / dd <= 1.0e-4) {
			vmax[0] = Math.abs(scale * (gd + half * scale * dhd));
			return;
		}

		// G and V are now orthogonal in the subspace spanned by G and D...
		ghg = vhg = vhv = zero;
		for (i = 1; i <= n; ++i) {
			sum = sumv = zero;
			for (j = 1; j <= n; ++j) {
				sum += (h[i - 1][j - 1] * g[j - 1]);
				sumv += (h[i - 1][j - 1] * v[j - 1]);
			}
			ghg += (sum * g[i - 1]);
			vhg += (sumv * g[i - 1]);
			vhv += (sumv * v[i - 1]);
		}
		vnorm = Math.sqrt(vv);
		ghg /= gg;
		vhg /= (vnorm * gnorm);
		vhv /= vv;
		if (Math.abs(vhg) <= 0.01 * Math.max(Math.abs(ghg), Math.abs(vhv))) {
			vmu = ghg - vhv;
			wcos = one;
			wsin = zero;
		} else {
			temp = half * (ghg - vhv);
			vmu = temp + RealMath.sign(RealMath.hypot(temp, vhg), temp);
			temp = RealMath.hypot(vmu, vhg);
			wcos = vmu / temp;
			wsin = vhg / temp;
		}
		tempa = wcos / gnorm;
		tempb = wsin / vnorm;
		tempc = wcos / vnorm;
		tempd = wsin / gnorm;
		for (i = 1; i <= n; ++i) {
			d[i - 1] = tempa * g[i - 1] + tempb * v[i - 1];
			v[i - 1] = tempc * v[i - 1] - tempd * g[i - 1];
		}

		// The final D is a multiple of the current D, V, D+V or D-V...
		dlin = wcos * gnorm / rho;
		vlin = -wsin * gnorm / rho;
		tempa = Math.abs(dlin) + half * Math.abs(vmu + vhv);
		tempb = Math.abs(vlin) + half * Math.abs(ghg - vmu);
		tempc = halfrt * (Math.abs(dlin) + Math.abs(vlin)) + 0.25 * Math.abs(ghg + vhv);
		if (tempa >= tempb && tempa >= tempc) {
			tempd = RealMath.sign(rho, dlin * (vmu + vhv));
			tempv = zero;
		} else if (tempb >= tempc) {
			tempd = zero;
			tempv = RealMath.sign(rho, vlin * (ghg - vmu));
		} else {
			tempd = RealMath.sign(halfrt * rho, dlin * (ghg + vhv));
			tempv = RealMath.sign(halfrt * rho, vlin * (ghg + vhv));
		}
		for (i = 1; i <= n; ++i) {
			d[i - 1] = tempd * d[i - 1] + tempv * v[i - 1];
		}
		vmax[0] = rho * rho * Math.max(tempa, Math.max(tempb, tempc));
	}
}
