/*
COBYLA---Constrained Optimization BY Linear Approximation.
Copyright (C) 1992 M. J. D. Powell (University of Cambridge)

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

import opt.multivariate.MultivariateOptimizerSolution;
import utils.BlasMath;

/**
 * A translation of the algorithm COBYLA for minimizing a general objective
 * function subject to non-linear inequality constraints. Originally written by
 * M. J. D. Powell.
 *
 *
 * REFERENCES:
 * 
 * [1] Fortran Code is from https://zhangzk.net/software.html
 * 
 * [2] M. J. D. Powell, "A direct search optimization method that models the
 * objective and constraint functions by linear interpolation," in Advances in
 * Optimization and Numerical Analysis, eds. S. Gomez and J.-P. Hennart (Kluwer
 * Academic: Dordrecht, 1994), p. 51-67.
 * 
 * [3] M. J. D. Powell, "Direct search algorithms for optimization
 * calculations," Acta Numerica 7, 287-336 (1998). Also available as University
 * of Cambridge, Department of Applied Mathematics and Theoretical Physics,
 * Numerical Analysis Group, Report NA1998/04 from
 * http://www.damtp.cam.ac.uk/user/na/reports.html
 */
public final class CobylaAlgorithm {

	private final double myTol, myRho0;
	private final int myMaxEvals;

	/**
	 *
	 * @param tolerance
	 * @param initialStep
	 * @param maxEvaluations
	 */
	public CobylaAlgorithm(final double tolerance, final double initialStep, final int maxEvaluations) {
		myTol = tolerance;
		myRho0 = initialStep;
		myMaxEvals = maxEvaluations;
	}

	public final MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> constr, final int m, final double[] guess) {

		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		final int[] maxfun = { myMaxEvals };

		// call main subroutine
		// TODO: check convergence
		cobyla(func, constr, n, m, x, myRho0, myTol, maxfun);
		return new MultivariateOptimizerSolution(x, maxfun[0], 0, false);
	}

	private static void cobyla(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> constr, final int n, final int m, final double[] x,
			final double rhobeg, final double rhoend, final int[] maxfun) {
		final int mpp = m + 2;
		final int iprint = 0;
		final int[] iact = new int[m + 1];
		final double[] con = new double[mpp];
		final double[] vsig = new double[n];
		final double[] veta = new double[n];
		final double[] sigbar = new double[n];
		final double[] dx = new double[n];
		final double[] w = new double[n * (3 * n + 2 * m + 11) + 4 * m + 6];
		final double[][] sim = new double[n + 1][n];
		final double[][] simi = new double[n][n];
		final double[][] datmat = new double[n + 1][mpp];
		final double[][] a = new double[m + 1][n];
		cobylb(func, constr, n, m, mpp, x, rhobeg, rhoend, iprint, maxfun, con, sim, simi, datmat, a, vsig, veta,
				sigbar, dx, w, iact);
	}

	private static void cobylb(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> constr, final int n, final int m, final int mpp,
			final double[] x, final double rhobeg, final double rhoend, final int iprint, final int[] maxfun,
			final double[] con, final double[][] sim, final double[][] simi, final double[][] datmat,
			final double[][] a, final double[] vsig, final double[] veta, final double[] sigbar, final double[] dx,
			final double[] w, final int[] iact) {
		final int[] ifull = new int[1];
		int i, ibrnch, j, jdrop, k, l, iflag = 0, nbest, np, mp, nfvals;
		double alpha, beta, gamma, delta, error, rho, resmax = 0.0, parmu, phimin, temp, tempa, f = 0.0, wsig, weta,
				parsig = 0.0, pareta, cvmaxp, cvmaxm, sum = 0.0, dxsign, resnew, barmu, prerec = 0.0, prerem = 0.0, phi,
				vmold, vmnew, trured, ratio, edgmax, denom, cmax = 0.0, cmin = 0.0;

		// Set the initial values of some parameters. The last column of SIM holds
		// the optimal vertex of the current simplex, and the preceding N columns
		// hold the displacements from the optimal vertex to the other vertices.
		// Further, SIMI holds the inverse of the matrix that is contained in the
		// first N columns of SIM.
		np = n + 1;
		mp = m + 1;
		alpha = 0.25;
		beta = 2.1;
		gamma = 0.5;
		delta = 1.1;
		rho = rhobeg;
		parmu = 0.0;
		nfvals = 0;
		temp = 1.0 / rho;
		for (i = 1; i <= n; i++) {
			sim[np - 1][i - 1] = x[i - 1];
			for (j = 1; j <= n; j++) {
				sim[j - 1][i - 1] = simi[i - 1][j - 1] = 0.0;
			}
			sim[i - 1][i - 1] = rho;
			simi[i - 1][i - 1] = temp;
		}
		jdrop = np;
		ibrnch = 0;

		int gotoflag = 40;
		while (true) {

			if (gotoflag == 40) {

				// Make the next call of the user-supplied subroutine CALCFC. These
				// instructions are also used for calling CALCFC during the
				// iterations of
				// the algorithm.
				if (nfvals >= maxfun[0] && nfvals > 0) {
					break;
				}
				++nfvals;
				f = func.apply(x);
				System.arraycopy(constr.apply(x), 0, con, 0, m);
				resmax = 0.0;
				if (m > 0) {
					for (k = 1; k <= m; ++k) {
						resmax = Math.max(resmax, -con[k - 1]);
					}
				}
				con[mp - 1] = f;
				con[mpp - 1] = resmax;
				if (ibrnch == 1) {
					gotoflag = 440;
				} else {

					// Set the recently calculated function values in a column of
					// DATMAT. This
					// array has a column for each vertex of the current simplex, the
					// entries of
					// each column being the values of the constraint functions (if
					// any)
					// followed by the objective function and the greatest constraint
					// violation
					// at the vertex.
					System.arraycopy(con, 0, datmat[jdrop - 1], 0, mpp);
					if (nfvals <= np) {

						// Exchange the new vertex of the initial simplex with the
						// optimal vertex if
						// necessary. Then, if the initial simplex is not complete,
						// pick its next
						// vertex and calculate the function values there.
						if (jdrop <= n) {
							if (datmat[np - 1][mp - 1] <= f) {
								x[jdrop - 1] = sim[np - 1][jdrop - 1];
							} else {
								sim[np - 1][jdrop - 1] = x[jdrop - 1];
								System.arraycopy(datmat[np - 1], 0, datmat[jdrop - 1], 0, mpp);
								System.arraycopy(con, 0, datmat[np - 1], 0, mpp);
								for (k = 1; k <= jdrop; k++) {
									sim[k - 1][jdrop - 1] = -rho;
									temp = 0.0;
									for (i = k; i <= jdrop; i++) {
										temp -= simi[i - 1][k - 1];
									}
									simi[jdrop - 1][k - 1] = temp;
								}
							}
						}
						if (nfvals <= n) {
							jdrop = nfvals;
							x[jdrop - 1] += rho;
							gotoflag = 40;
							continue;
						}
					}
					ibrnch = 1;
					gotoflag = 140;
				}
			}

			if (gotoflag == 140) {

				// Identify the optimal vertex of the current simplex.
				phimin = datmat[np - 1][mp - 1] + parmu * datmat[np - 1][mpp - 1];
				nbest = np;
				for (j = 1; j <= n; j++) {
					temp = datmat[j - 1][mp - 1] + parmu * datmat[j - 1][mpp - 1];
					if (temp < phimin) {
						nbest = j;
						phimin = temp;
					} else if (temp == phimin && parmu == 0.0) {
						if (datmat[j - 1][mpp - 1] < datmat[nbest - 1][mpp - 1]) {
							nbest = j;
						}
					}
				}

				// Switch the best vertex into pole position if it is not there
				// already,
				// and also update SIM, SIMI and DATMAT.
				if (nbest <= n) {
					for (i = 1; i <= mpp; i++) {
						temp = datmat[np - 1][i - 1];
						datmat[np - 1][i - 1] = datmat[nbest - 1][i - 1];
						datmat[nbest - 1][i - 1] = temp;
					}
					for (i = 1; i <= n; i++) {
						temp = sim[nbest - 1][i - 1];
						sim[nbest - 1][i - 1] = 0.0;
						sim[np - 1][i - 1] += temp;
						tempa = 0.0;
						for (k = 1; k <= n; k++) {
							sim[k - 1][i - 1] -= temp;
							tempa -= simi[k - 1][i - 1];
						}
						simi[nbest - 1][i - 1] = tempa;
					}
				}

				// Make an error return if SIGI is a poor approximation to the
				// inverse of
				// the leading N by N submatrix of SIG.
				error = 0.0;
				for (i = 1; i <= n; i++) {
					for (j = 1; j <= n; j++) {
						temp = 0.0;
						if (i == j) {
							temp -= 1.0;
						}
						temp += BlasMath.ddotm(n, simi[i - 1], 1, sim[j - 1], 1);
						error = Math.max(error, Math.abs(temp));
					}
				}
				if (error > 0.1) {
					break;
				}

				// Calculate the coefficients of the linear approximations to the
				// objective
				// and constraint functions, placing minus the objective function
				// gradient
				// after the constraint gradients in the array A. The vector W is
				// used for
				// working space.
				for (k = 1; k <= mp; k++) {
					con[k - 1] = -datmat[np - 1][k - 1];
					for (j = 1; j <= n; j++) {
						w[j - 1] = datmat[j - 1][k - 1] + con[k - 1];
					}
					for (i = 1; i <= n; i++) {
						temp = 0.0;
						for (j = 1; j <= n; j++) {
							temp += w[j - 1] * simi[j - 1][i - 1];
						}
						if (k == mp) {
							temp = -temp;
						}
						a[k - 1][i - 1] = temp;
					}
				}

				// Calculate the values of sigma and eta, and set IFLAG=0 if the
				// current
				// simplex is not acceptable.
				iflag = 1;
				parsig = alpha * rho;
				pareta = beta * rho;
				for (j = 1; j <= n; j++) {
					wsig = BlasMath.ddotm(n, simi[j - 1], 1, simi[j - 1], 1);
					weta = BlasMath.ddotm(n, sim[j - 1], 1, sim[j - 1], 1);
					vsig[j - 1] = 1.0 / Math.sqrt(wsig);
					veta[j - 1] = Math.sqrt(weta);
					if (vsig[j - 1] < parsig || veta[j - 1] > pareta) {
						iflag = 0;
					}
				}

				// If a new vertex is needed to improve acceptability, then decide
				// which
				// vertex to drop from the simplex.
				if (ibrnch != 1 && iflag != 1) {
					jdrop = 0;
					temp = pareta;
					for (j = 1; j <= n; j++) {
						if (veta[j - 1] > temp) {
							jdrop = j;
							temp = veta[j - 1];
						}
					}
					if (jdrop == 0) {
						for (j = 1; j <= n; j++) {
							if (vsig[j - 1] < temp) {
								jdrop = j;
								temp = vsig[j - 1];
							}
						}
					}

					// Calculate the step to the new vertex and its sign.
					temp = gamma * rho * vsig[jdrop - 1];
					BlasMath.dscal1(n, temp, simi[jdrop - 1], 1, dx, 1);
					cvmaxp = cvmaxm = 0.0;
					for (k = 1; k <= mp; k++) {
						sum = BlasMath.ddotm(n, a[k - 1], 1, dx, 1);
						if (k < mp) {
							temp = datmat[np - 1][k - 1];
							cvmaxp = Math.max(cvmaxp, -sum - temp);
							cvmaxm = Math.max(cvmaxm, sum - temp);
						}
					}
					dxsign = 1.0;
					if (parmu * (cvmaxp - cvmaxm) > sum + sum) {
						dxsign = -1.0;
					}

					// Update the elements of SIM and SIMI, and set the next X.
					temp = 0.0;
					for (i = 1; i <= n; i++) {
						dx[i - 1] *= dxsign;
						temp += simi[jdrop - 1][i - 1] * dx[i - 1];
					}
					System.arraycopy(dx, 0, sim[jdrop - 1], 0, n);
					BlasMath.dscalm(n, 1.0 / temp, simi[jdrop - 1], 1);
					for (j = 1; j <= n; j++) {
						if (j != jdrop) {
							temp = BlasMath.ddotm(n, simi[j - 1], 1, dx, 1);
							BlasMath.daxpym(n, -temp, simi[jdrop - 1], 1, simi[j - 1], 1);
						}
						x[j - 1] = sim[np - 1][j - 1] + dx[j - 1];
					}
					gotoflag = 40;
					continue;
				} else {
					gotoflag = 370;
				}
			}

			if (gotoflag == 370) {

				// Calculate DX=x(*)-x(0). Branch if the length of DX is less than
				// 0.5*RHO.
				int iz = 1;
				int izdota = iz + n * n;
				int ivmc = izdota + n;
				int isdirn = ivmc + mp;
				int idxnew = isdirn + n;
				int ivmd = idxnew + n;
				trstlp(n, m, a, con, rho, dx, ifull, iact, w, w, izdota, w, ivmc, w, isdirn, w, idxnew, w, ivmd);
				if (ifull[0] == 0) {
					temp = BlasMath.ddotm(n, dx, 1, dx, 1);
					if (temp < 0.25 * rho * rho) {
						ibrnch = 1;
						gotoflag = 550;
						continue;
					}
				}

				// Predict the change to F and the new maximum constraint violation
				// if the
				// variables are altered from x(0) to x(0)+DX.
				resnew = con[mp - 1] = 0.0;
				for (k = 1; k <= mp; k++) {
					sum = con[k - 1];
					sum -= BlasMath.ddotm(n, a[k - 1], 1, dx, 1);
					if (k < mp) {
						resnew = Math.max(resnew, sum);
					}
				}

				// Increase PARMU if necessary and branch back if this change alters
				// the
				// optimal vertex. Otherwise PREREM and PREREC will be set to the
				// predicted
				// reductions in the merit function and the maximum constraint
				// violation
				// respectively.
				barmu = 0.0;
				prerec = datmat[np - 1][mpp - 1] - resnew;
				if (prerec > 0.0) {
					barmu = sum / prerec;
				}
				if (parmu < 1.5 * barmu) {
					parmu = 2.0 * barmu;
					phi = datmat[np - 1][mp - 1] + parmu * datmat[np - 1][mpp - 1];
					boolean goto140 = false;
					for (j = 1; j <= n; j++) {
						temp = datmat[j - 1][mp - 1] + parmu * datmat[j - 1][mpp - 1];
						if (temp < phi) {
							goto140 = true;
							break;
						}
						if (temp == phi && parmu == 0.0) {
							if (datmat[j - 1][mpp - 1] < datmat[np - 1][mpp - 1]) {
								goto140 = true;
								break;
							}
						}
					}
					if (goto140) {
						gotoflag = 140;
						continue;
					}
				}
				prerem = parmu * prerec - sum;

				// Calculate the constraint and objective functions at x(*). Then
				// find the actual reduction in the merit function.
				BlasMath.dxpy1(n, sim[np - 1], 1, dx, 1, x, 1);
				ibrnch = 1;
				gotoflag = 40;
				continue;
			}

			if (gotoflag == 440) {
				vmold = datmat[np - 1][mp - 1] + parmu * datmat[np - 1][mpp - 1];
				vmnew = f + parmu * resmax;
				trured = vmold - vmnew;
				if (parmu == 0.0 && f == datmat[np - 1][mp - 1]) {
					prerem = prerec;
					trured = datmat[np - 1][mpp - 1] - resmax;
				}

				// Begin the operations that decide whether x(*) should replace one
				// of the
				// vertices of the current simplex, the change being mandatory if
				// TRURED is
				// positive. Firstly, JDROP is set to the index of the vertex that is
				// to be
				// replaced.
				ratio = 0.0;
				if (trured <= 0.0) {
					ratio = 1.0;
				}
				jdrop = 0;
				for (j = 1; j <= n; j++) {
					temp = BlasMath.ddotm(n, simi[j - 1], 1, dx, 1);
					temp = Math.abs(temp);
					if (temp > ratio) {
						jdrop = j;
						ratio = temp;
					}
					sigbar[j - 1] = temp * vsig[j - 1];
				}

				// Calculate the value of ell.
				edgmax = delta * rho;
				l = 0;
				for (j = 1; j <= n; j++) {
					if (sigbar[j - 1] >= parsig || sigbar[j - 1] >= vsig[j - 1]) {
						temp = veta[j - 1];
						if (trured > 0.0) {
							temp = 0.0;
							for (i = 1; i <= n; i++) {
								final double dif = dx[i - 1] - sim[j - 1][i - 1];
								temp += (dif * dif);
							}
							temp = Math.sqrt(temp);
						}
						if (temp > edgmax) {
							l = j;
							edgmax = temp;
						}
					}
				}
				if (l > 0) {
					jdrop = l;
				}
				if (jdrop != 0) {

					// Revise the simplex by updating the elements of SIM, SIMI and
					// DATMAT.
					System.arraycopy(dx, 0, sim[jdrop - 1], 0, n);
					temp = BlasMath.ddotm(n, simi[jdrop - 1], 1, dx, 1);
					BlasMath.dscalm(n, 1.0 / temp, simi[jdrop - 1], 1);
					for (j = 1; j <= n; j++) {
						if (j != jdrop) {
							temp = BlasMath.ddotm(n, simi[j - 1], 1, dx, 1);
							BlasMath.daxpym(n, -temp, simi[jdrop - 1], 1, simi[j - 1], 1);
						}
					}
					System.arraycopy(con, 0, datmat[jdrop - 1], 0, mpp);

					// Branch back for further iterations with the current RHO.
					if (trured > 0.0 && trured >= 0.1 * prerem) {
						gotoflag = 140;
						continue;
					}
				}
				gotoflag = 550;
			}

			if (gotoflag == 550) {
				if (iflag == 0) {
					ibrnch = 0;
					gotoflag = 140;
					continue;
				}

				// Otherwise reduce RHO if it is not at its least value and reset
				// PARMU.
				if (rho > rhoend) {
					rho *= 0.5;
					if (rho <= 1.5 * rhoend) {
						rho = rhoend;
					}
					if (parmu > 0.0) {
						denom = 0.0;
						for (k = 1; k <= mp; k++) {
							cmin = datmat[np - 1][k - 1];
							cmax = cmin;
							for (i = 1; i <= n; i++) {
								cmin = Math.min(cmin, datmat[i - 1][k - 1]);
								cmax = Math.max(cmax, datmat[i - 1][k - 1]);
							}
							if (k <= m && cmin < 0.5 * cmax) {
								temp = Math.max(cmax, 0.0) - cmin;
								if (denom <= 0.0) {
									denom = temp;
								} else {
									denom = Math.min(denom, temp);
								}
							}

						}
						if (denom == 0.0) {
							parmu = 0.0;
						} else if (cmax - cmin < parmu * denom) {
							parmu = (cmax - cmin) / denom;
						}
					}
					gotoflag = 140;
				} else if (ifull[0] == 1) {

					// Return the best calculated values of the variables.
					maxfun[0] = nfvals;
					return;
				} else {
					break;
				}
			}
		}
		System.arraycopy(sim[np - 1], 0, x, 0, n);
		maxfun[0] = nfvals;
	}

	private static void trstlp(final int n, final int m, final double[][] a, final double[] b, final double rho,
			final double[] dx, final int[] ifull, final int[] iact, final double[] z, final double[] zdota,
			final int iz, final double[] vmultc, final int ivmc, final double[] sdirn, final int isd,
			final double[] dxnew, final int idx, final double[] vmultd, final int ivmd) {

		int mcon, nact, nactx = 0, i, icon = 0, icount, isave, j, k, kk, kl, kp, kw;
		double resmax, optold, optnew, tot, sp, spabs, temp, acca, accb, alpha, beta, ratio, tempa, zdotv, zdvabs,
				vsave, dd, sd, ss, stpful, step, resold = 0.0, zdwabs, zdotw, sum, sumabs;

		// Initialize Z and some other variables...
		ifull[0] = 1;
		mcon = m;
		nact = 0;
		resmax = 0.0;
		for (i = 1; i <= n; i++) {
			for (j = 1; j <= n; j++) {
				z[(i - 1) + n * (j - 1)] = 0.0;
			}
			z[(i - 1) + n * (i - 1)] = 1.0;
			dx[i - 1] = 0.0;
		}
		if (m >= 1) {
			for (k = 1; k <= m; k++) {
				if (b[k - 1] > resmax) {
					resmax = b[k - 1];
					icon = k;
				}
			}
			for (k = 1; k <= m; k++) {
				iact[k - 1] = k;
				vmultc[k - 1 + ivmc - 1] = resmax - b[k - 1];
			}
		}
		if (resmax == 0.0) {
			mcon = m + 1;
			icon = mcon;
			iact[mcon - 1] = mcon;
			vmultc[mcon - 1 + ivmc - 1] = 0.0;
		} else {
			for (i = 1; i <= n; i++) {
				sdirn[i - 1 + isd - 1] = 0.0;
			}
		}

		// End the current stage of the calculation if 3 consecutive iterations
		// have either failed to reduce the best calculated value of the objective
		// function or to increase the number of active constraints since the best
		// value was calculated. This strategy prevents cycling, but there is a
		// remote possibility that it will cause premature termination.
		optold = 0.0;
		icount = 0;

		while (true) {

			if (mcon == m) {
				optnew = resmax;
			} else {
				optnew = -BlasMath.ddotm(n, dx, 1, a[mcon - 1], 1);
			}
			if (icount == 0 || optnew < optold) {
				optold = optnew;
				nactx = nact;
				icount = 3;
			} else if (nact > nactx) {
				nactx = nact;
				icount = 3;
			} else {
				--icount;
				if (icount == 0) {

					// We employ any freedom that may be available to reduce the
					// objective
					// function before returning a DX whose length is less than RHO.
					if (mcon == m) {
						mcon = m + 1;
						icon = mcon;
						iact[mcon - 1] = mcon;
						vmultc[mcon - 1 + ivmc - 1] = optold = 0.0;
						icount = 0;
						continue;
					} else {
						ifull[0] = 0;
						return;
					}
				}
			}

			// If ICON exceeds NACT, then we add the constraint with index IACT(ICON)
			// to
			// the active set. Apply Givens rotations so that the last N-NACT-1
			// columns
			// of Z are orthogonal to the gradient of the new constraint, a scalar
			// product being set to zero if its nonzero value could be due to
			// computer
			// rounding errors. The array DXNEW is used for working space.
			if (icon <= nact) {

				// Delete the constraint that has the index IACT(ICON) from the
				// active set.
				if (icon < nact) {
					isave = iact[icon - 1];
					vsave = vmultc[icon - 1 + ivmc - 1];
					k = icon;

					while (true) {

						kp = k + 1;
						kk = iact[kp - 1];
						sp = BlasMath.ddotm(n, z, n * (k - 1) + 1, a[kk - 1], 1);
						temp = Math.sqrt(sp * sp + zdota[kp - 1 + iz - 1] * zdota[kp - 1 + iz - 1]);
						alpha = zdota[kp - 1 + iz - 1] / temp;
						beta = sp / temp;
						zdota[kp - 1 + iz - 1] = alpha * zdota[k - 1 + iz - 1];
						zdota[k - 1 + iz - 1] = temp;
						for (i = 1; i <= n; i++) {
							temp = alpha * z[(i - 1) + n * (kp - 1)] + beta * z[(i - 1) + n * (k - 1)];
							z[i - 1 + n * (kp - 1)] = alpha * z[(i - 1) + n * (k - 1)] - beta * z[i - 1 + n * (kp - 1)];
							z[i - 1 + n * (k - 1)] = temp;
						}
						iact[k - 1] = kk;
						vmultc[k - 1 + ivmc - 1] = vmultc[kp - 1 + ivmc - 1];
						k = kp;
						if (k >= nact) {
							break;
						}
					}
					iact[k - 1] = isave;
					vmultc[k - 1 + ivmc - 1] = vsave;
				}
				--nact;

				// If stage one is in progress, then set SDIRN to the direction of
				// the next
				// change to the current vector of variables.
				if (mcon > m) {

					// Pick the next search direction of stage two.
					temp = 1.0 / zdota[nact - 1 + iz - 1];
					BlasMath.dscal1(n, temp, z, n * (nact - 1) + 1, sdirn, isd);
				} else {
					temp = BlasMath.ddotm(n, sdirn, isd, z, n * nact + 1);
					BlasMath.daxpym(n, -temp, z, n * nact + 1, sdirn, isd);
				}
			} else {
				kk = iact[icon - 1];
				System.arraycopy(a[kk - 1], 0, dxnew, idx - 1, n);
				tot = 0.0;
				k = n;

				while (k > nact) {
					sp = spabs = 0.0;
					for (i = 1; i <= n; i++) {
						temp = z[i - 1 + n * (k - 1)] * dxnew[i - 1 + idx - 1];
						sp += temp;
						spabs += Math.abs(temp);
					}
					acca = spabs + 0.1 * Math.abs(sp);
					accb = spabs + 0.2 * Math.abs(sp);
					if (spabs >= acca || acca >= accb) {
						sp = 0.0;
					}
					if (tot == 0.0) {
						tot = sp;
					} else {
						kp = k + 1;
						temp = Math.sqrt(sp * sp + tot * tot);
						alpha = sp / temp;
						beta = tot / temp;
						tot = temp;
						for (i = 1; i <= n; i++) {
							temp = alpha * z[i - 1 + n * (k - 1)] + beta * z[i - 1 + n * (kp - 1)];
							z[i - 1 + n * (kp - 1)] = alpha * z[i - 1 + n * (kp - 1)] - beta * z[i - 1 + n * (k - 1)];
							z[i - 1 + n * (k - 1)] = temp;
						}
					}
					--k;
				}

				// Add the new constraint if this can be done without a deletion from
				// the
				// active set.
				if (tot != 0.0) {
					nact = nact + 1;
					zdota[nact - 1 + iz - 1] = tot;
					vmultc[icon - 1 + ivmc - 1] = vmultc[nact - 1 + ivmc - 1];
					vmultc[nact - 1 + ivmc - 1] = 0.0;
				} else {

					// The next instruction is reached if a deletion has to be made
					// from the
					// active set in order to make room for the new active
					// constraint, because
					// the new constraint gradient is a linear combination of the
					// gradients of
					// the old active constraints. Set the elements of VMULTD to the
					// multipliers
					// of the linear combination. Further, set IOUT to the index of
					// the
					// constraint to be deleted, but branch if no suitable index can
					// be found.
					ratio = -1.0;
					k = nact;

					while (true) {

						zdotv = zdvabs = 0.0;
						for (i = 1; i <= n; i++) {
							temp = z[i - 1 + n * (k - 1)] * dxnew[i - 1 + idx - 1];
							zdotv += temp;
							zdvabs += Math.abs(temp);
						}
						acca = zdvabs + 0.1 * Math.abs(zdotv);
						accb = zdvabs + 0.2 * Math.abs(zdotv);
						if (zdvabs < acca && acca < accb) {
							temp = zdotv / zdota[k - 1 + iz - 1];
							if (temp > 0.0 && iact[k - 1] <= m) {
								tempa = vmultc[k - 1 + ivmc - 1] / temp;
								if (ratio < 0.0 || tempa < ratio) {
									ratio = tempa;
									// iout = k;
								}
							}
							if (k >= 2) {
								kw = iact[k - 1];
								BlasMath.daxpym(n, -temp, a[kw - 1], 1, dxnew, idx);
							}
							vmultd[k - 1 + ivmd - 1] = temp;
						} else {
							vmultd[k - 1 + ivmd - 1] = 0.0;
						}
						--k;
						if (k <= 0) {
							break;
						}
					}

					if (ratio < 0.0) {

						// We employ any freedom that may be available to reduce the
						// objective
						// function before returning a DX whose length is less than
						// RHO.
						if (mcon == m) {
							mcon = m + 1;
							icon = mcon;
							iact[mcon - 1] = mcon;
							vmultc[mcon - 1 + ivmc - 1] = optold = 0.0;
							icount = 0;
							continue;
						} else {
							ifull[0] = 0;
							return;
						}
					}

					// Revise the Lagrange multipliers and reorder the active
					// constraints so
					// that the one to be replaced is at the end of the list. Also
					// calculate the
					// new value of ZDOTA(NACT) and branch if it is not acceptable.
					for (k = 1; k <= nact; k++) {
						vmultc[k - 1 + ivmc - 1] = Math.max(0.0,
								vmultc[k - 1 + ivmc - 1] - ratio * vmultd[k - 1 + ivmd - 1]);
					}
					if (icon < nact) {
						isave = iact[icon - 1];
						vsave = vmultc[icon - 1 + ivmc - 1];
						k = icon;

						while (true) {

							kp = k + 1;
							kw = iact[kp - 1];
							sp = BlasMath.ddotm(n, z, n * (k - 1) + 1, a[kw - 1], 1);
							temp = Math.sqrt(sp * sp + zdota[kp - 1 + iz - 1] * zdota[kp - 1 + iz - 1]);
							alpha = zdota[kp - 1 + iz - 1] / temp;
							beta = sp / temp;
							zdota[kp - 1 + iz - 1] = alpha * zdota[k - 1 + iz - 1];
							zdota[k - 1 + iz - 1] = temp;
							for (i = 1; i <= n; i++) {
								temp = alpha * z[i - 1 + n * (kp - 1)] + beta * z[i - 1 + n * (k - 1)];
								z[i - 1 + n * (kp - 1)] = alpha * z[i - 1 + n * (k - 1)]
										- beta * z[i - 1 + n * (kp - 1)];
								z[i - 1 + n * (k - 1)] = temp;
							}
							iact[k - 1] = kw;
							vmultc[k - 1 + ivmc - 1] = vmultc[kp - 1 + ivmc - 1];
							k = kp;
							if (k >= nact) {
								break;
							}
						}
						iact[k - 1] = isave;
						vmultc[k - 1 + ivmc - 1] = vsave;
					}
					temp = BlasMath.ddotm(n, z, n * (nact - 1) + 1, a[kk - 1], 1);
					if (temp == 0.0) {

						// We employ any freedom that may be available to reduce the
						// objective
						// function before returning a DX whose length is less than
						// RHO.
						if (mcon == m) {
							mcon = m + 1;
							icon = mcon;
							iact[mcon - 1] = mcon;
							vmultc[mcon - 1 + ivmc - 1] = optold = 0.0;
							icount = 0;
							continue;
						} else {
							ifull[0] = 0;
							return;
						}
					}
					zdota[nact - 1 + iz - 1] = temp;
					vmultc[icon - 1 + ivmc - 1] = 0.0;
					vmultc[nact - 1 + ivmc - 1] = ratio;
				}

				// Update IACT and ensure that the objective function continues to be
				// treated as the last active constraint when MCON>M.
				iact[icon - 1] = iact[nact - 1];
				iact[nact - 1] = kk;
				if (mcon > m && kk != mcon) {
					k = nact - 1;
					sp = BlasMath.ddotm(n, z, n * (k - 1) + 1, a[kk - 1], 1);
					temp = Math.sqrt(sp * sp + zdota[nact - 1 + iz - 1] * zdota[nact - 1 + iz - 1]);
					alpha = zdota[nact - 1 + iz - 1] / temp;
					beta = sp / temp;
					zdota[nact - 1 + iz - 1] = alpha * zdota[k - 1 + iz - 1];
					zdota[k - 1 + iz - 1] = temp;
					for (i = 1; i <= n; i++) {
						temp = alpha * z[i - 1 + n * (nact - 1)] + beta * z[i - 1 + n * (k - 1)];
						z[i - 1 + n * (nact - 1)] = alpha * z[i - 1 + n * (k - 1)] - beta * z[i - 1 + n * (nact - 1)];
						z[i - 1 + n * (k - 1)] = temp;
					}
					iact[nact - 1] = iact[k - 1];
					iact[k - 1] = kk;
					temp = vmultc[k - 1 + ivmc - 1];
					vmultc[k - 1 + ivmc - 1] = vmultc[nact - 1 + ivmc - 1];
					vmultc[nact - 1 + ivmc - 1] = temp;
				}

				// If stage one is in progress, then set SDIRN to the direction of
				// the next
				// change to the current vector of variables.
				if (mcon > m) {

					// Pick the next search direction of stage two.
					temp = 1.0 / zdota[nact - 1 + iz - 1];
					BlasMath.dscal1(n, temp, z, n * (nact - 1) + 1, sdirn, isd);
				} else {
					kk = iact[nact - 1];
					temp = BlasMath.ddotm(n, sdirn, isd, a[kk - 1], 1);
					temp -= 1.0;
					temp /= zdota[nact - 1 + iz - 1];
					BlasMath.daxpym(n, -temp, z, n * (nact - 1) + 1, sdirn, isd);
				}
			}

			// Calculate the step to the boundary of the trust region or take the
			// step
			// that reduces RESMAX to zero. The two statements below that include the
			// factor 1.0E-6 prevent some harmless underflows that occurred in a test
			// calculation. Further, we skip the step if it could be zero within a
			// reasonable tolerance for computer rounding errors.
			dd = rho * rho;
			sd = ss = 0.0;
			for (i = 1; i <= n; i++) {
				if (Math.abs(dx[i - 1]) >= 1.0e-6 * rho) {
					dd -= dx[i - 1] * dx[i - 1];
				}
				sd += dx[i - 1] * sdirn[i - 1 + isd - 1];
				ss += sdirn[i - 1 + isd - 1] * sdirn[i - 1 + isd - 1];
			}
			if (dd <= 0.0) {

				// We employ any freedom that may be available to reduce the
				// objective
				// function before returning a DX whose length is less than RHO.
				if (mcon == m) {
					mcon = m + 1;
					icon = mcon;
					iact[mcon - 1] = mcon;
					vmultc[mcon - 1 + ivmc - 1] = optold = 0.0;
					icount = 0;
					continue;
				} else {
					ifull[0] = 0;
					return;
				}
			}
			temp = Math.sqrt(ss * dd);
			if (Math.abs(sd) >= 1.0e-6 * temp) {
				temp = Math.sqrt(ss * dd + sd * sd);
			}
			stpful = dd / (temp + sd);
			step = stpful;
			if (mcon == m) {
				acca = step + 0.1 * resmax;
				accb = step + 0.2 * resmax;
				if (step >= acca || acca >= accb) {
					mcon = m + 1;
					icon = mcon;
					iact[mcon - 1] = mcon;
					vmultc[mcon - 1 + ivmc - 1] = optold = 0.0;
					icount = 0;
					continue;
				}
				step = Math.min(step, resmax);
			}

			// Set DXNEW to the new variables if STEP is the steplength, and reduce
			// RESMAX to the corresponding maximum residual if stage one is being
			// done.
			// Because DXNEW will be changed during the calculation of some Lagrange
			// multipliers, it will be restored to the following value later.
			BlasMath.daxpy1(n, step, sdirn, isd, dx, 1, dxnew, idx);
			if (mcon == m) {
				resold = resmax;
				resmax = 0.0;
				for (k = 1; k <= nact; k++) {
					kk = iact[k - 1];
					temp = b[kk - 1];
					temp -= BlasMath.ddotm(n, a[kk - 1], 1, dxnew, idx);
					resmax = Math.max(resmax, temp);
				}
			}

			// Set VMULTD to the VMULTC vector that would occur if DX became DXNEW. A
			// device is included to force VMULTD(K)=0.0 if deviations from this
			// value
			// can be attributed to computer rounding errors. First calculate the new
			// Lagrange multipliers.
			k = nact;
			while (true) {
				zdotw = zdwabs = 0.0;
				for (i = 1; i <= n; i++) {
					temp = z[i - 1 + n * (k - 1)] * dxnew[i - 1 + idx - 1];
					zdotw += temp;
					zdwabs += Math.abs(temp);
				}
				acca = zdwabs + 0.1 * Math.abs(zdotw);
				accb = zdwabs + 0.2 * Math.abs(zdotw);
				if (zdwabs >= acca || acca >= accb) {
					zdotw = 0.0;
				}
				vmultd[k - 1 + ivmd - 1] = zdotw / zdota[k - 1 + iz - 1];
				if (k >= 2) {
					kk = iact[k - 1];
					BlasMath.daxpym(n, -vmultd[k - 1 + ivmd - 1], a[kk - 1], 1, dxnew, idx);
					--k;
				} else {
					break;
				}
			}
			if (mcon > m) {
				vmultd[nact - 1 + ivmd - 1] = Math.max(0.0, vmultd[nact - 1 + ivmd - 1]);
			}

			// Complete VMULTC by finding the new constraint residuals.
			BlasMath.daxpy1(n, step, sdirn, isd, dx, 1, dxnew, idx);
			if (mcon > nact) {
				kl = nact + 1;
				for (k = kl; k <= mcon; k++) {
					kk = iact[k - 1];
					sum = resmax - b[kk - 1];
					sumabs = resmax + Math.abs(b[kk - 1]);
					for (i = 1; i <= n; i++) {
						temp = a[kk - 1][i - 1] * dxnew[i - 1 + idx - 1];
						sum += temp;
						sumabs += Math.abs(temp);
					}
					acca = sumabs + 0.1 * Math.abs(sum);
					accb = sumabs + 0.2 * Math.abs(sum);
					if (sumabs >= acca || acca >= accb) {
						sum = 0.0;
					}
					vmultd[k - 1 + ivmd - 1] = sum;
				}
			}

			// Calculate the fraction of the step from DX to DXNEW that will be
			// taken.
			ratio = 1.0;
			icon = 0;
			for (k = 1; k <= mcon; k++) {
				if (vmultd[k - 1 + ivmd - 1] < 0.0) {
					temp = vmultc[k - 1 + ivmc - 1] / (vmultc[k - 1 + ivmc - 1] - vmultd[k - 1 + ivmd - 1]);
					if (temp < ratio) {
						ratio = temp;
						icon = k;
					}
				}
			}

			// Update DX, VMULTC and RESMAX.
			temp = 1.0 - ratio;
			for (i = 1; i <= n; i++) {
				dx[i - 1] = temp * dx[i - 1] + ratio * dxnew[i - 1 + idx - 1];
			}
			for (k = 1; k <= mcon; k++) {
				vmultc[k - 1 + ivmc - 1] = Math.max(0.0,
						temp * vmultc[k - 1 + ivmc - 1] + ratio * vmultd[k - 1 + ivmd - 1]);
			}
			if (mcon == m) {
				resmax = resold + ratio * (resmax - resold);
			}

			// If the full step is not acceptable then begin another iteration.
			// Otherwise switch to stage two or end the calculation.
			if (icon > 0) {
			} else if (step == stpful) {
				return;
			} else {
				mcon = m + 1;
				icon = mcon;
				iact[mcon - 1] = mcon;
				vmultc[mcon - 1 + ivmc - 1] = optold = 0.0;
				icount = 0;
			}
		}
	}
}
