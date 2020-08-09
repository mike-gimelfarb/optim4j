/*
Condition for Use: This software is freely available, but we expect that all publications 
describing work using this software, or all commercial products using it, quote at least
one of the references given below.
 
[1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization,
(1995), SIAM Journal on Scientific and Statistical Computing , 16, 5, pp. 1190-1208.

[2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines 
for large scale bound constrained optimization (1997), ACM Transactions on Mathematical
Software, Vol 23, Num. 4, pp. 550 - 560.

[3] J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines 
for large scale bound constrained optimization (2011), to appear in ACM Transactions on
Mathematical Software. 

This software is released under the "New BSD License" (aka "Modified BSD License" or "3-clause license").
 */
package opt.multivariate.unconstrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.GradientOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.BlasMath;
import utils.Constants;

/**
 *
 * @author Michael
 */
public final class LBFGSBAlgorithm extends GradientOptimizer {

	private final int mySize;

	/**
	 *
	 * @param tolerance
	 * @param memorySize
	 */
	public LBFGSBAlgorithm(final double tolerance, final int memorySize) {
		super(tolerance);
		mySize = memorySize;
	}

	@Override
	public final MultivariateOptimizerSolution optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess) {

		// prepare variables
		final double[] l = new double[guess.length];
		final double[] u = new double[guess.length];
		final int[] nbd = new int[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			l[i] = 1.0e-60;
			u[i] = 1.0e+60;
			nbd[i] = 0;
		}
		final int[] fev = new int[1];

		// call main subroutine
		final double[] result = lbfgsb(f, df, guess, l, u, nbd, mySize, 10.0, myTol, fev);
		// TODO: check convergence
		return new MultivariateOptimizerSolution(result, fev[0], fev[0], false);
	}

	/**
	 *
	 * @param f
	 * @param df
	 * @param guess
	 * @param lb
	 * @param ub
	 * @return
	 */
	public final MultivariateOptimizerSolution optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess, final double[] lb, final double[] ub) {

		// prepare variables
		final int[] nbd = new int[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			nbd[i] = 2;
		}
		final int[] fev = new int[1];

		// call main subroutine
		// TODO: check convergence
		final double[] result = lbfgsb(f, df, guess, lb, ub, nbd, mySize, 10.0, myTol, fev);
		return new MultivariateOptimizerSolution(result, fev[0], fev[0], false);
	}

	private static double[] lbfgsb(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final double[] guess, final double[] l, final double[] u,
			final int[] nbd, final int m, final double factr, final double pgtol, final int[] fev) {

		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);

		// call main subroutine
		fev[0] = 0;
		driver(func, dfunc, n, m, x, l, u, nbd, factr, pgtol, fev);
		return x;
	}

	private static void driver(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final int n, final int m, final double[] x,
			final double[] l, final double[] u, final int[] nbd, final double factr, final double pgtol,
			final int[] fev) {

		final double[][] ws = new double[n][m], wy = new double[n][m], sy = new double[m][m], ss = new double[m][m],
				wt = new double[m][m], wn = new double[2 * m][2 * m], snd = new double[2 * m][2 * m];
		final double[] z = new double[n], r = new double[n], d = new double[n], t = new double[n], xp = new double[n],
				wa = new double[8 * m], dsave = new double[29], f = new double[1], g = new double[n];
		final int[] iwa = new int[3 * n], isave = new int[44];
		final String[] task = new String[1], csave = new String[1];
		final boolean[] lsave = new boolean[4];

		// We start the iteration by initializing task.
		task[0] = "START";

		// ------- the beginning of the loop ----------
		while (true) {

			// This is the call to the L-BFGS-B code.
			setulb(n, m, x, l, u, nbd, f, g, factr, pgtol, ws, wy, sy, ss, wt, wn, snd, z, r, d, t, xp, wa, iwa, task,
					-1, csave, lsave, isave, dsave);

			if ("FG".equals(task[0].substring(0, 2))) {

				// the minimization routine has returned to request the
				// function f and gradient g values at the current x.
				f[0] = func.apply(x);
				System.arraycopy(dfunc.apply(x), 0, g, 0, n);
				++fev[0];

				// go back to the minimization routine.
				continue;
			}

			if (!"NEW_X".equals(task[0].substring(0, 5))) {
				break;
			}
		}

		// the minimization routine has returned with a new iterate,
		// and we have opted to continue the iteration.
		// If task is neither FG nor NEW_X we terminate execution.
	}

	private static void setulb(final int n, final int m, final double[] x, final double[] l, final double[] u,
			final int[] nbd, final double[] f, final double[] g, final double factr, final double pgtol,
			final double[][] ws, final double[][] wy, final double[][] sy, final double[][] ss, final double[][] wt,
			final double[][] wn, final double[][] snd, final double[] z, final double[] r, final double[] d,
			final double[] t, final double[] xp, final double[] wa, final int[] iwa, final String[] task,
			final int iprint, final String[] csave, final boolean[] lsave, final int[] isave, final double[] dsave) {

		if ("START".equals(task[0].substring(0, 5))) {
			isave[1 - 1] = m * n;
			isave[2 - 1] = m * m;
			isave[3 - 1] = 4 * m * m;
			isave[4 - 1] = 1;
			isave[5 - 1] = isave[4 - 1] + isave[1 - 1];
			isave[6 - 1] = isave[5 - 1] + isave[1 - 1];
			isave[7 - 1] = isave[6 - 1] + isave[2 - 1];
			isave[8 - 1] = isave[7 - 1] + isave[2 - 1];
			isave[9 - 1] = isave[8 - 1] + isave[2 - 1];
			isave[10 - 1] = isave[9 - 1] + isave[3 - 1];
			isave[11 - 1] = isave[10 - 1] + isave[3 - 1];
			isave[12 - 1] = isave[11 - 1] + n;
			isave[13 - 1] = isave[12 - 1] + n;
			isave[14 - 1] = isave[13 - 1] + n;
			isave[15 - 1] = isave[14 - 1] + n;
			isave[16 - 1] = isave[15 - 1] + n;
		}

		final int[] index = new int[n];
		final int[] iwhere = new int[n];
		final int[] indx2 = new int[n];
		System.arraycopy(iwa, 0, index, 0, n);
		System.arraycopy(iwa, n, iwhere, 0, n);
		System.arraycopy(iwa, 2 * n, indx2, 0, n);
		mainlb(n, m, x, l, u, nbd, f, g, factr, pgtol, ws, wy, sy, ss, wt, wn, snd, z, r, d, t, xp, wa, index, iwhere,
				indx2, task, iprint, csave, lsave, isave, dsave);
		System.arraycopy(index, 0, iwa, 0, n);
		System.arraycopy(iwhere, 0, iwa, n, n);
		System.arraycopy(indx2, 0, iwa, 2 * n, n);
	}

	private static void mainlb(final int n, final int m, final double[] x, final double[] l, final double[] u,
			final int[] nbd, final double[] f, final double[] g, final double factr, final double pgtol,
			final double[][] ws, final double[][] wy, final double[][] sy, final double[][] ss, final double[][] wt,
			final double[][] wn, final double[][] snd, final double[] z, final double[] r, final double[] d,
			final double[] t, final double[] xp, final double[] wa, final int[] index, final int[] iwhere,
			final int[] indx2, final String[] task, final int iprint, final String[] csave, final boolean[] lsave,
			final int[] isave, final double[] dsave) {

		final double[] sbgnrm = new double[1], fold = new double[1], stp = new double[1], dnorm = new double[1],
				gd = new double[1], gdold = new double[1], dtd = new double[1], xstep = new double[1],
				stpmx = new double[1], theta = new double[1];
		final boolean[] prjctd = new boolean[1], cnstnd = new boolean[1], boxed = new boolean[1], wrk = new boolean[1];
		final int[] info = new int[1], k = new int[1], nseg = new int[1], nfree = new int[1], nenter = new int[1],
				ileave = new int[1], iword = new int[1], ifun = new int[1], iback = new int[1], nfgv = new int[1],
				itail = new int[1], col = new int[1], head = new int[1];
		boolean updatd;
		// String word;
		int i, nintol, itfile, nskip, iter, iupdat, nact;
		double dr, rr, tol, ddum, epsmch;
		final double one = 1.0, zero = 0.0;

		int gotoflag;
		if ("START".equals(task[0].substring(0, 5))) {

			// Initialize counters and scalars when task='START'.
			// for the limited memory BFGS matrices:
			epsmch = Constants.EPSILON;
			col[0] = 0;
			head[0] = 1;
			theta[0] = one;
			iupdat = 0;
			updatd = false;
			iback[0] = itail[0] = iword[0] = nact = ileave[0] = nenter[0] = 0;
			fold[0] = dnorm[0] = gd[0] = stpmx[0] = zero;
			sbgnrm[0] = stp[0] = gdold[0] = dtd[0] = zero;

			// for operation counts:
			iter = nfgv[0] = nseg[0] = nintol = nskip = 0;
			nfree[0] = n;
			ifun[0] = 0;

			// for stopping tolerance:
			tol = factr * epsmch;

			// 'word' records the status of subspace solutions.
			// word = "---";
			// 'info' records the termination information.
			info[0] = 0;
			itfile = 8;

			// Check the input arguments for errors.
			errclb(n, m, factr, l, u, nbd, task, info, k);
			if ("ERROR".equals(task[0].substring(0, 5))) {

				// call prn3lb(n,x,f,task,iprint,info,itfile,...
				return;
			}
			// call prn1lb(n,m,l,u,x,iprint,itfile,epsmch)

			// Initialize iwhere & project x onto the feasible set.
			active(n, l, u, nbd, x, iwhere, iprint, prjctd, cnstnd, boxed);

			// The end of the initialization.
			// Compute f0 and g0.
			task[0] = "FG_START";

			// return to the driver to calculate f and g; reenter at 111.
			gotoflag = 1000;
		} else {

			// restore local variables.
			prjctd[0] = lsave[1 - 1];
			cnstnd[0] = lsave[2 - 1];
			boxed[0] = lsave[3 - 1];
			updatd = lsave[4 - 1];

			nintol = isave[1 - 1];
			itfile = isave[3 - 1];
			iback[0] = isave[4 - 1];
			nskip = isave[5 - 1];
			head[0] = isave[6 - 1];
			col[0] = isave[7 - 1];
			itail[0] = isave[8 - 1];
			iter = isave[9 - 1];
			iupdat = isave[10 - 1];
			nseg[0] = isave[12 - 1];
			nfgv[0] = isave[13 - 1];
			info[0] = isave[14 - 1];
			ifun[0] = isave[15 - 1];
			iword[0] = isave[16 - 1];
			nfree[0] = isave[17 - 1];
			nact = isave[18 - 1];
			ileave[0] = isave[19 - 1];
			nenter[0] = isave[20 - 1];

			theta[0] = dsave[1 - 1];
			fold[0] = dsave[2 - 1];
			tol = dsave[3 - 1];
			dnorm[0] = dsave[4 - 1];
			epsmch = dsave[5 - 1];
			gd[0] = dsave[11 - 1];
			stpmx[0] = dsave[12 - 1];
			sbgnrm[0] = dsave[13 - 1];
			stp[0] = dsave[14 - 1];
			gdold[0] = dsave[15 - 1];
			dtd[0] = dsave[16 - 1];

			// After returning from the driver go to the point where execution
			// is to resume.
			if ("FG_LN".equals(task[0].substring(0, 5))) {
				gotoflag = 666;
			} else if ("NEW_X".equals(task[0].substring(0, 5))) {
				gotoflag = 777;
			} else if ("FG_ST".equals(task[0].substring(0, 5))) {

				nfgv[0] = 1;

				// Compute the infinity norm of the (-) projected gradient.
				projgr(n, l, u, nbd, x, g, sbgnrm);
				if (sbgnrm[0] <= pgtol) {

					// terminate the algorithm.
					task[0] = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL";
					gotoflag = 1000;
				} else {
					gotoflag = 222;
				}
			} else if ("STOP".equals(task[0].substring(0, 4))) {
				if ("CPU".equals(task[0].substring(6, 9))) {

					// restore the previous iterate.
					System.arraycopy(t, 0, x, 0, n);
					System.arraycopy(r, 0, g, 0, n);
					f[0] = fold[0];
				}
				gotoflag = 1000;
			} else {

				// Compute f0 and g0.
				task[0] = "FG_START";

				// return to the driver to calculate f and g; reenter at 111.
				gotoflag = 1000;
			}
		}

		while (gotoflag != 1000) {

			// ----------------- the beginning of the loop --------------------------
			if (gotoflag == 222) {

				iword[0] = -1;
				if (!cnstnd[0] && col[0] > 0) {

					// skip the search for GCP.
					System.arraycopy(x, 0, z, 0, n);
					wrk[0] = updatd;
					nseg[0] = 0;
				} else {

					// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
					// c
					// c Compute the Generalized Cauchy Point (GCP).
					// c
					// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
					cauchy(n, x, l, u, nbd, g, indx2, iwhere, t, d, z, m, wy, ws, sy, wt, theta[0], col[0], head[0], wa,
							1, wa, 2 * m + 1, wa, 4 * m + 1, wa, 6 * m + 1, nseg, iprint, sbgnrm[0], info, epsmch);
					if (info[0] != 0) {

						// singular triangular system detected; refresh the lbfgs
						// memory.
						info[0] = col[0] = 0;
						head[0] = 1;
						theta[0] = one;
						iupdat = 0;
						updatd = false;
						gotoflag = 222;
						continue;
					}
					nintol += nseg[0];

					// Count the entering and leaving variables for iter > 0;
					// find the index set of free and active variables at the GCP.
					freev(n, nfree, index, nenter, ileave, indx2, iwhere, wrk, updatd, cnstnd[0], iprint, iter);
					nact = n - nfree[0];
				}

				// If there are no free variables or B=theta*I, then
				// skip the subspace minimization.
				if (nfree[0] != 0 && col[0] != 0) {

					// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
					// c
					// c Subspace minimization.
					// c
					// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
					// Form the LEL^T factorization of the indefinite
					// matrix K = [-D -Y'ZZ'Y/theta L_a'-R_z' ]
					// [L_a -R_z theta*S'AA'S ]
					// where E = [-I 0]
					// [ 0 I]
					if (wrk[0]) {
						formk(n, nfree[0], index, nenter[0], ileave[0], indx2, iupdat, updatd, wn, snd, m, ws, wy, sy,
								theta[0], col[0], head[0], info);
					}
					if (info[0] != 0) {

						// nonpositive definiteness in Cholesky factorization;
						// refresh the lbfgs memory and restart the iteration.
						info[0] = col[0] = 0;
						head[0] = 1;
						theta[0] = one;
						iupdat = 0;
						updatd = false;
						gotoflag = 222;
						continue;
					}

					// compute r=-Z'B(xcp-xk)-Z'g (using wa(2m+1)=W'(xcp-x)
					// from 'cauchy').
					cmprlb(n, m, x, g, ws, wy, sy, wt, z, r, wa, index, theta[0], col[0], head[0], nfree[0], cnstnd[0],
							info);
					if (info[0] == 0) {

						// c-jlm-jn call the direct method.
						subsm(n, m, nfree[0], index, l, u, nbd, z, r, xp, ws, wy, theta[0], x, g, col[0], head[0],
								iword, wa, wn, iprint, info);
					}
					if (info[0] != 0) {

						// singular triangular system detected;
						// refresh the lbfgs memory and restart the iteration.
						info[0] = col[0] = 0;
						head[0] = 1;
						theta[0] = one;
						iupdat = 0;
						updatd = false;
						gotoflag = 222;
						continue;
					}
				}

				// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
				// c
				// c Line search and optimality tests.
				// c
				// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
				// Generate the search direction d:=z-x.
				for (i = 1; i <= n; ++i) {
					d[i - 1] = z[i - 1] - x[i - 1];
				}
				gotoflag = 666;
			}

			if (gotoflag == 666) {

				lnsrlb(n, l, u, nbd, x, f, fold, gd, gdold, g, d, r, t, z, stp, dnorm, dtd, xstep, stpmx, iter, ifun,
						iback, nfgv, info, task, boxed[0], cnstnd[0], csave, isave, 22, dsave, 17);
				if (info[0] != 0 || iback[0] >= 20) {

					// restore the previous iterate.
					System.arraycopy(t, 0, x, 0, n);
					System.arraycopy(r, 0, g, 0, n);
					f[0] = fold[0];
					if (col[0] == 0) {

						// abnormal termination.
						if (info[0] == 0) {
							info[0] = -9;

							// restore the actual number of f and g evaluations etc.
							--nfgv[0];
							--ifun[0];
							--iback[0];
						}
						task[0] = "ABNORMAL_TERMINATION_IN_LNSRCH";
						++iter;
						gotoflag = 1000;
						continue;
					} else {

						// refresh the lbfgs memory and restart the iteration.
						if (info[0] == 0) {
							--nfgv[0];
						}
						info[0] = col[0] = 0;
						head[0] = 1;
						theta[0] = one;
						iupdat = 0;
						updatd = false;
						task[0] = "RESTART_FROM_LNSRCH";
						gotoflag = 222;
						continue;
					}
				} else if ("FG_LN".equals(task[0].substring(0, 5))) {

					// return to the driver for calculating f and g; reenter at 666.
					gotoflag = 1000;
					continue;
				} else {

					// calculate and print out the quantities related to the new X.
					++iter;

					// Compute the infinity norm of the projected (-)gradient.
					projgr(n, l, u, nbd, x, g, sbgnrm);

					// Print iteration information.
					// call prn2lb(n,x,f,g,iprint,itfile,iter,nfgv,nact,
					// sbgnrm,nseg,word,iword,iback,stp,xstep)
					gotoflag = 1000;
					continue;
				}
			}

			if (gotoflag == 777) {

				// Test for termination.
				if (sbgnrm[0] <= pgtol) {

					// terminate the algorithm.
					task[0] = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL";
					gotoflag = 1000;
					continue;
				}
				ddum = Math.max(Math.abs(fold[0]), Math.max(Math.abs(f[0]), one));
				if ((fold[0] - f[0]) <= tol * ddum) {

					// terminate the algorithm.
					task[0] = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH";
					if (iback[0] >= 10) {
						info[0] = -5;
					}

					// i.e., to issue a warning if iback>10 in the line search.
					gotoflag = 1000;
					continue;
				}

				// Compute d=newx-oldx, r=newg-oldg, rr=y'y and dr=y's.
				for (i = 1; i <= n; ++i) {
					r[i - 1] = g[i - 1] - r[i - 1];
				}
				rr = BlasMath.ddotm(n, r, 1, r, 1);
				if (stp[0] == one) {
					dr = gd[0] - gdold[0];
					ddum = -gdold[0];
				} else {
					dr = (gd[0] - gdold[0]) * stp[0];
					BlasMath.dscalm(n, stp[0], d, 1);
					ddum = -gdold[0] * stp[0];
				}
				if (dr <= epsmch * ddum) {

					// skip the L-BFGS update.
					++nskip;
					updatd = false;
					gotoflag = 222;
					continue;
				}

				// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
				// c
				// c Update the L-BFGS matrix.
				// c
				// cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
				updatd = true;
				++iupdat;

				// Update matrices WS and WY and form the middle matrix in B.
				matupd(n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head, theta, rr, dr, stp[0], dtd[0]);

				// Form the upper half of the pds T = theta*SS + L*D^(-1)*L';
				// Store T in the upper triangular of the array wt;
				// Cholesky factorize T to J*J' with
				// J' stored in the upper triangular of wt.
				formt(m, wt, sy, ss, col[0], theta[0], info);
				if (info[0] != 0) {

					// nonpositive definiteness in Cholesky factorization;
					// refresh the lbfgs memory and restart the iteration.
					info[0] = col[0] = 0;
					head[0] = 1;
					theta[0] = one;
					iupdat = 0;
					updatd = false;
				}
				gotoflag = 222;
			}

			// Now the inverse of the middle matrix in B is
			// [ D^(1/2) O ] [ -D^(1/2) D^(-1/2)*L' ]
			// [ -L*D^(-1/2) J ] [ 0 J' ]
			// -------------------- the end of the loop -----------------------------
		}

		// Save local variables.
		lsave[1 - 1] = prjctd[0];
		lsave[2 - 1] = cnstnd[0];
		lsave[3 - 1] = boxed[0];
		lsave[4 - 1] = updatd;

		isave[1 - 1] = nintol;
		isave[3 - 1] = itfile;
		isave[4 - 1] = iback[0];
		isave[5 - 1] = nskip;
		isave[6 - 1] = head[0];
		isave[7 - 1] = col[0];
		isave[8 - 1] = itail[0];
		isave[9 - 1] = iter;
		isave[10 - 1] = iupdat;
		isave[12 - 1] = nseg[0];
		isave[13 - 1] = nfgv[0];
		isave[14 - 1] = info[0];
		isave[15 - 1] = ifun[0];
		isave[16 - 1] = iword[0];
		isave[17 - 1] = nfree[0];
		isave[18 - 1] = nact;
		isave[19 - 1] = ileave[0];
		isave[20 - 1] = nenter[0];

		dsave[1 - 1] = theta[0];
		dsave[2 - 1] = fold[0];
		dsave[3 - 1] = tol;
		dsave[4 - 1] = dnorm[0];
		dsave[5 - 1] = epsmch;
		dsave[11 - 1] = gd[0];
		dsave[12 - 1] = stpmx[0];
		dsave[13 - 1] = sbgnrm[0];
		dsave[14 - 1] = stp[0];
		dsave[15 - 1] = gdold[0];
		dsave[16 - 1] = dtd[0];
	}

	private static void active(final int n, final double[] l, final double[] u, final int[] nbd, final double[] x,
			final int[] iwhere, final int iprint, final boolean[] prjctd, final boolean[] cnstnd,
			final boolean[] boxed) {

		int nbdd, i;
		final double zero = 0.0;

		// Initialize nbdd, prjctd, cnstnd and boxed.
		nbdd = 0;
		prjctd[0] = cnstnd[0] = false;
		boxed[0] = true;

		// Project the initial x to the easible set if necessary.
		for (i = 1; i <= n; ++i) {
			if (nbd[i - 1] > 0) {
				if (nbd[i - 1] <= 2 && x[i - 1] <= l[i - 1]) {
					if (x[i - 1] < l[i - 1]) {
						prjctd[0] = true;
						x[i - 1] = l[i - 1];
					}
					++nbdd;
				} else if (nbd[i - 1] >= 2 && x[i - 1] >= u[i - 1]) {
					if (x[i - 1] > u[i - 1]) {
						prjctd[0] = true;
						x[i - 1] = u[i - 1];
					}
					++nbdd;
				}
			}
		}

		// Initialize iwhere and assign values to cnstnd and boxed.
		for (i = 1; i <= n; ++i) {
			if (nbd[i - 1] != 2) {
				boxed[0] = false;
			}
			if (nbd[i - 1] == 0) {

				// this variable is always free
				iwhere[i - 1] = -1;

				// otherwise set x(i)=mid(x(i), u(i), l(i)).
			} else {
				cnstnd[0] = true;
				if (nbd[0] == 2 && u[i - 1] - l[i - 1] <= zero) {

					// this variable is always fixed
					iwhere[i - 1] = 3;
				} else {
					iwhere[i - 1] = 0;
				}
			}
		}
	}

	private static void cauchy(final int n, final double[] x, final double[] l, final double[] u, final int[] nbd,
			final double[] g, final int[] iorder, final int[] iwhere, final double[] t, final double[] d,
			final double[] xcp, final int m, final double[][] wy, final double[][] ws, final double[][] sy,
			final double[][] wt, final double theta, final int col, final int head, final double[] p, final int ip,
			final double[] c, final int ic, final double[] wbp, final int iwbp, final double[] v, final int iv,
			final int[] nseg, final int iprint, final double sbgnrm, final int[] info, final double epsmch) {

		boolean xlower, xupper, bnded;
		int i, j, col2, nfree, nbreak, pointr, ibp, nleft, ibkmin, iter;
		double f1, f2, dt, dtm, tsum, dibp, zibp, dibp2, bkmin, tu = 0.0, tl = 0.0, wmc, wmp, wmw, tj, tj0, neggi,
				f2_org;
		final double one = 1.0, zero = 0.0;

		// Check the status of the variables, reset iwhere(i) if necessary;
		// compute the Cauchy direction d and the breakpoints t; initialize
		// the derivative f1 and the vector p = W'd (for theta = 1).
		if (sbgnrm <= zero) {
			System.arraycopy(x, 0, xcp, 0, n);
			return;
		}
		bnded = true;
		nfree = n + 1;
		nbreak = ibkmin = 0;
		bkmin = zero;
		col2 = 2 * col;
		f1 = zero;

		// We set p to zero and build it up as we determine d.
		for (i = 1; i <= col2; ++i) {
			p[i - 1 + ip - 1] = zero;
		}

		// In the following loop we determine for each variable its bound
		// status and its breakpoint, and update p accordingly.
		// Smallest breakpoint is identified.
		for (i = 1; i <= n; ++i) {
			neggi = -g[i - 1];
			if (iwhere[i - 1] != 3 && iwhere[i - 1] != -1) {

				// if x(i) is not a constant and has bounds,
				// compute the difference between x(i) and its bounds.
				if (nbd[i - 1] <= 2) {
					tl = x[i - 1] - l[i - 1];
				}
				if (nbd[i - 1] >= 2) {
					tu = u[i - 1] - x[i - 1];
				}

				// If a variable is close enough to a bound
				// we treat it as at bound.
				xlower = nbd[i - 1] <= 2 && tl <= zero;
				xupper = nbd[i - 1] >= 2 && tu <= zero;

				// reset iwhere(i).
				iwhere[i - 1] = 0;
				if (xlower) {
					if (neggi <= zero) {
						iwhere[i - 1] = 1;
					}
				} else if (xupper) {
					if (neggi >= zero) {
						iwhere[i - 1] = 2;
					}
				} else if (Math.abs(neggi) <= zero) {
					iwhere[i - 1] = -3;
				}
			}
			pointr = head;
			if (iwhere[i - 1] != 0 && iwhere[i - 1] != -1) {
				d[i - 1] = zero;
			} else {
				d[i - 1] = neggi;
				f1 -= neggi * neggi;

				// calculate p := p - W'e_i* (g_i).
				for (j = 1; j <= col; ++j) {
					p[j - 1 + ip - 1] += wy[i - 1][pointr - 1] * neggi;
					p[col + j - 1 + ip - 1] += ws[i - 1][pointr - 1] * neggi;
					pointr = (pointr % m) + 1;
				}
				if (nbd[i - 1] <= 2 && nbd[i - 1] != 0 && neggi < zero) {

					// x(i) + d(i) is bounded; compute t(i).
					++nbreak;
					iorder[nbreak - 1] = i;
					t[nbreak - 1] = tl / (-neggi);
					if (nbreak == 1 || t[nbreak - 1] < bkmin) {
						bkmin = t[nbreak - 1];
						ibkmin = nbreak;
					}
				} else if (nbd[i - 1] >= 2 && neggi > zero) {

					// x(i) + d(i) is bounded; compute t(i).
					++nbreak;
					iorder[nbreak - 1] = i;
					t[nbreak - 1] = tu / neggi;
					if (nbreak == 1 || t[nbreak - 1] < bkmin) {
						bkmin = t[nbreak - 1];
						ibkmin = nbreak;
					}
				} else {

					// x(i) + d(i) is not bounded.
					--nfree;
					iorder[nfree - 1] = i;
					if (Math.abs(neggi) > zero) {
						bnded = false;
					}
				}
			}
		}

		// The indices of the nonzero components of d are now stored
		// in iorder(1),...,iorder(nbreak) and iorder(nfree),...,iorder(n).
		// The smallest of the nbreak breakpoints is in t(ibkmin)=bkmin.
		if (theta != one) {

			// complete the initialization of p for theta not= one.
			BlasMath.dscalm(col, theta, p, col + 1 + ip - 1);
		}

		// Initialize GCP xcp = x.
		System.arraycopy(x, 0, xcp, 0, n);
		if (nbreak == 0 && nfree == n + 1) {

			// is a zero vector, return with the initial xcp as GCP.
			return;
		}

		// Initialize c = W'(xcp - x) = 0.
		for (j = 1; j <= col2; ++j) {
			c[j - 1 + ic - 1] = zero;
		}

		// Initialize derivative f2.
		f2 = -theta * f1;
		f2_org = f2;
		if (col > 0) {
			bmv(m, sy, wt, col, p, ip, v, iv, info);
			if (info[0] != 0) {
				return;
			}
			f2 -= BlasMath.ddotm(col2, v, iv, p, ip);
		}
		dtm = -f1 / f2;
		tsum = zero;
		nseg[0] = 1;

		// If there are no breakpoints, locate the GCP and return.
		if (nbreak != 0) {
			nleft = nbreak;
			iter = 1;
			tj = zero;

			// ------------------- the beginning of the loop
			// -------------------------
			while (true) {

				// Find the next smallest breakpoint;
				// compute dt = t(nleft) - t(nleft + 1).
				tj0 = tj;
				if (iter == 1) {

					// Since we already have the smallest breakpoint we need not do
					// heapsort yet. Often only one breakpoint is used and the
					// cost of heapsort is avoided.
					tj = bkmin;
					ibp = iorder[ibkmin - 1];
				} else {
					if (iter == 2) {

						// Replace the already used smallest breakpoint with the
						// breakpoint numbered nbreak > nlast, before heapsort call.
						if (ibkmin != nbreak) {
							t[ibkmin - 1] = t[nbreak - 1];
							iorder[ibkmin - 1] = iorder[nbreak - 1];
						}

						// Update heap structure of breakpoints
						// (if iter=2, initialize heap).
					}
					hpsolb(nleft, t, iorder, iter - 2);
					tj = t[nleft - 1];
					ibp = iorder[nleft - 1];
				}
				dt = tj - tj0;

				// If a minimizer is within this interval, locate the GCP and return.
				if (dtm < dt) {
					break;
				}

				// Otherwise fix one variable and
				// reset the corresponding component of d to zero.
				tsum += dt;
				--nleft;
				++iter;
				dibp = d[ibp - 1];
				d[ibp - 1] = zero;
				if (dibp > zero) {
					zibp = u[ibp - 1] - x[ibp - 1];
					xcp[ibp - 1] = u[ibp - 1];
					iwhere[ibp - 1] = 2;
				} else {
					zibp = l[ibp - 1] - x[ibp - 1];
					xcp[ibp - 1] = l[ibp - 1];
					iwhere[ibp - 1] = 1;
				}
				if (nleft == 0 && nbreak == n) {

					// all n variables are fixed, return with xcp as GCP.
					dtm = dt;

					// Update c = c + dtm*p = W'(x^c - x)
					// which will be used in computing r = Z'(B(x^c - x) + g).
					if (col > 0) {
						BlasMath.daxpym(col2, dtm, p, ip, c, ic);
					}
					return;
				}

				// Update the derivative information.
				++nseg[0];
				dibp2 = dibp * dibp;

				// Update f1 and f2.
				// temporarily set f1 and f2 for col=0.
				f1 += (dt * f2 + dibp2 - theta * dibp * zibp);
				f2 -= (theta * dibp2);
				if (col > 0) {

					// update c = c + dt*p.
					BlasMath.daxpym(col2, dt, p, ip, c, ic);

					// choose wbp,
					// the row of W corresponding to the breakpoint encountered.
					pointr = head;
					for (j = 1; j <= col; ++j) {
						wbp[j - 1 + iwbp - 1] = wy[ibp - 1][pointr - 1];
						wbp[col + j - 1 + iwbp - 1] = theta * ws[ibp - 1][pointr - 1];
						pointr = (pointr % m) + 1;
					}

					// compute (wbp)Mc, (wbp)Mp, and (wbp)M(wbp)'.
					bmv(m, sy, wt, col, wbp, iwbp, v, iv, info);
					if (info[0] != 0) {
						return;
					}
					wmc = BlasMath.ddotm(col2, c, ic, v, iv);
					wmp = BlasMath.ddotm(col2, p, ip, v, iv);
					wmw = BlasMath.ddotm(col2, wbp, iwbp, v, iv);

					// update p = p - dibp*wbp.
					BlasMath.daxpym(col2, -dibp, wbp, iwbp, p, ip);

					// complete updating f1 and f2 while col > 0.
					f1 += dibp * wmc;
					f2 += (2.0 * dibp * wmp - dibp2 * wmw);
				}

				f2 = Math.max(epsmch * f2_org, f2);
				if (nleft > 0) {
					dtm = -f1 / f2;

					// to repeat the loop for unsearched intervals.
				} else if (bnded) {
					// f1 = f2 = zero;
					dtm = zero;
					break;
				} else {
					dtm = -f1 / f2;
					break;
				}
			}
		}

		// ------------------- the end of the loop -------------------------------
		if (dtm <= zero) {
			dtm = zero;
		}
		tsum += dtm;

		// Move free variables (i.e., the ones w/o breakpoints) and
		// the variables whose breakpoints haven't been reached.
		BlasMath.daxpym(n, tsum, d, 1, xcp, 1);

		// Update c = c + dtm*p = W'(x^c - x)
		// which will be used in computing r = Z'(B(x^c - x) + g).
		if (col > 0) {
			BlasMath.daxpym(col2, dtm, p, ip, c, ic);
		}
	}

	private static void cmprlb(final int n, final int m, final double[] x, final double[] g, final double[][] ws,
			final double[][] wy, final double[][] sy, final double[][] wt, final double[] z, final double[] r,
			final double[] wa, final int[] index, final double theta, final int col, final int head, final int nfree,
			final boolean cnstnd, final int[] info) {

		int i, j, k, pointr;
		double a1, a2;

		if (!cnstnd && col > 0) {
			for (i = 1; i <= n; ++i) {
				r[i - 1] = -g[i - 1];
			}
		} else {
			for (i = 1; i <= nfree; ++i) {
				k = index[i - 1];
				r[i - 1] = -theta * (z[k - 1] - x[k - 1]) - g[k - 1];
			}
			bmv(m, sy, wt, col, wa, 2 * m + 1, wa, 1, info);
			if (info[0] != 0) {
				info[0] = -8;
				return;
			}
			pointr = head;
			for (j = 1; j <= col; ++j) {
				a1 = wa[j - 1];
				a2 = theta * wa[col + j - 1];
				for (i = 1; i <= nfree; ++i) {
					k = index[i - 1];
					r[i - 1] += wy[k - 1][pointr - 1] * a1 + ws[k - 1][pointr - 1] * a2;
				}
				pointr = (pointr % m) + 1;
			}
		}
	}

	private static void bmv(final int m, final double[][] sy, final double[][] wt, final int col, final double[] v,
			final int iv, final double[] p, final int ip, final int[] info) {

		int i, k, i2;
		double sum;

		if (col == 0) {
			return;
		}

		// PART I: solve [ D^(1/2) O ] [ p1 ] = [ v1 ]
		// [ -L*D^(-1/2) J ] [ p2 ] [ v2 ].
		//
		// solve Jp2=v2+LD^(-1)v1.
		p[col + 1 - 1 + ip - 1] = v[col + 1 - 1 + iv - 1];
		for (i = 2; i <= col; ++i) {
			i2 = col + i;
			sum = 0.0;
			for (k = 1; k <= i - 1; ++k) {
				sum += sy[i - 1][k - 1] * v[k - 1 + iv - 1] / sy[k - 1][k - 1];
			}
			p[i2 - 1 + ip - 1] = v[i2 - 1 + iv - 1] + sum;
		}

		// Solve the triangular system
		dtrsl(wt, m, col, p, col + 1 + ip - 1, 11, info);
		if (info[0] != 0) {
			return;
		}

		// solve D^(1/2)p1=v1.
		for (i = 1; i <= col; ++i) {
			p[i - 1 + ip - 1] = v[i - 1 + iv - 1] / Math.sqrt(sy[i - 1][i - 1]);
		}

		// PART II: solve [ -D^(1/2) D^(-1/2)*L' ] [ p1 ] = [ p1 ]
		// [ 0 J' ] [ p2 ] [ p2 ].
		//
		// solve J^Tp2=p2.
		dtrsl(wt, m, col, p, col + 1 + ip - 1, 01, info);
		if (info[0] != 0) {
			return;
		}

		// compute p1=-D^(-1/2)(p1-D^(-1/2)L'p2) =-D^(-1/2)p1+D^(-1)L'p2.
		for (i = 1; i <= col; ++i) {
			p[i - 1 + ip - 1] = -p[i - 1 + ip - 1] / Math.sqrt(sy[i - 1][i - 1]);
		}
		for (i = 1; i <= col; ++i) {
			sum = 0.0;
			for (k = i + 1; k <= col; ++k) {
				sum += sy[k - 1][i - 1] * p[col + k - 1 + ip - 1] / sy[i - 1][i - 1];
			}
			p[i - 1 + ip - 1] += sum;
		}
	}

	private static void errclb(final int n, final int m, final double factr, final double[] l, final double[] u,
			final int[] nbd, final String[] task, final int[] info, final int[] k) {

		int i;
		final double one = 1.0, zero = 0.0;

		// Check the input arguments for errors.
		if (n <= 0) {
			task[0] = "ERROR: N .LE. 0";
		}
		if (m <= 0) {
			task[0] = "ERROR: M .LE. 0";
		}
		if (factr < zero) {
			task[0] = "ERROR: FACTR .LT. 0";
		}

		// Check the validity of the arrays nbd(i), u(i), and l(i).
		for (i = 1; i <= n; ++i) {
			if (nbd[i - 1] < 0 || nbd[i - 1] > 3) {

				// return
				task[0] = "ERROR: INVALID NBD";
				info[0] = -6;
				k[0] = i;
			}
			if (nbd[i - 1] == 2) {
				if (l[i - 1] > u[i - 1]) {

					// return
					task[0] = "ERROR: NO FEASIBLE SOLUTION";
					info[0] = -7;
					k[0] = i;
				}
			}
		}
	}

	private static void formk(final int n, final int nsub, final int[] ind, final int nenter, final int ileave,
			final int[] indx2, final int iupdat, final boolean updatd, final double[][] wn, final double[][] wn1,
			final int m, final double[][] ws, final double[][] wy, final double[][] sy, final double theta,
			final int col, final int head, final int[] info) {

		final double[] aux = new double[wn.length];
		int m2, ipntr, jpntr, iy, is, jy, js, is1, js1, k1, i, k, col2, pbegin, pend, dbegin, dend, upcl, jaux;
		double temp1, temp2, temp3, temp4;
		final double one = 1.0, zero = 0.0;

		// Form the lower triangular part of
		// WN1 = [Y' ZZ'Y L_a'+R_z']
		// [L_a+R_z S'AA'S ]
		// where L_a is the strictly lower triangular part of S'AA'Y
		// R_z is the upper triangular part of S'ZZ'Y.
		if (updatd) {
			if (iupdat > m) {

				// shift old part of WN1.
				for (jy = 1; jy <= m - 1; ++jy) {
					js = m + jy;

					// call dcopy(m-jy,wn1(jy+1,jy+1),1,wn1(jy,jy),1)
					// call dcopy(m-jy,wn1(js+1,js+1),1,wn1(js,js),1)
					for (jaux = 1; jaux <= m - jy; ++jaux) {
						wn1[jy - 1 + jaux - 1][jy - 1] = wn1[jy + 1 - 1 + jaux - 1][jy + 1 - 1];
						wn1[js - 1 + jaux - 1][js - 1] = wn1[js + 1 - 1 + jaux - 1][js + 1 - 1];
					}

					// call dcopy(m-1,wn1(m+2,jy+1),1,wn1(m+1,jy),1)
					for (jaux = 1; jaux <= m - 1; ++jaux) {
						wn1[m + 1 - 1 + jaux - 1][jy - 1] = wn1[m + 2 - 1 + jaux - 1][jy + 1 - 1];
					}
				}
			}

			// put new rows in blocks (1,1), (2,1) and (2,2).
			pbegin = 1;
			pend = nsub;
			dbegin = nsub + 1;
			dend = n;
			iy = col;
			is = m + col;
			ipntr = head + col - 1;
			if (ipntr > m) {
				ipntr -= m;
			}
			jpntr = head;
			for (jy = 1; jy <= col; ++jy) {
				js = m + jy;
				temp1 = temp2 = temp3 = zero;

				// compute element jy of row 'col' of Y'ZZ'Y
				for (k = pbegin; k <= pend; ++k) {
					k1 = ind[k - 1];
					temp1 += wy[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
				}

				// compute elements jy of row 'col' of L_a and S'AA'S
				for (k = dbegin; k <= dend; ++k) {
					k1 = ind[k - 1];
					temp2 += ws[k1 - 1][ipntr - 1] * ws[k1 - 1][jpntr - 1];
					temp3 += ws[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
				}
				wn1[iy - 1][jy - 1] = temp1;
				wn1[is - 1][js - 1] = temp2;
				wn1[is - 1][jy - 1] = temp3;
				jpntr = (jpntr % m) + 1;
			}

			// put new column in block (2,1).
			jy = col;
			jpntr = head + col - 1;
			if (jpntr > m) {
				jpntr -= m;
			}
			ipntr = head;
			for (i = 1; i <= col; ++i) {
				is = m + i;
				temp3 = zero;

				// compute element i of column 'col' of R_z
				for (k = pbegin; k <= pend; ++k) {
					k1 = ind[k - 1];
					temp3 += ws[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
				}
				ipntr = (ipntr % m) + 1;
				wn1[is - 1][jy - 1] = temp3;
			}
			upcl = col - 1;
		} else {
			upcl = col;
		}

		// modify the old parts in blocks (1,1) and (2,2) due to changes
		// in the set of free variables.
		ipntr = head;
		for (iy = 1; iy <= upcl; ++iy) {
			is = m + iy;
			jpntr = head;
			for (jy = 1; jy <= iy; ++jy) {
				js = m + jy;
				temp1 = temp2 = temp3 = temp4 = zero;
				for (k = 1; k <= nenter; ++k) {
					k1 = indx2[k - 1];
					temp1 += wy[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
					temp2 += ws[k1 - 1][ipntr - 1] * ws[k1 - 1][jpntr - 1];
				}
				for (k = ileave; k <= n; ++k) {
					k1 = indx2[k - 1];
					temp3 += wy[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
					temp4 += ws[k1 - 1][ipntr - 1] * ws[k1 - 1][jpntr - 1];
				}
				wn1[iy - 1][jy - 1] += (temp1 - temp3);
				wn1[is - 1][js - 1] += (-temp2 + temp4);
				jpntr = (jpntr % m) + 1;
			}
			ipntr = (ipntr % m) + 1;
		}

		// modify the old parts in block (2,1).
		ipntr = head;
		for (is = m + 1; is <= m + upcl; ++is) {
			jpntr = head;
			for (jy = 1; jy <= upcl; ++jy) {
				temp1 = temp3 = zero;
				for (k = 1; k <= nenter; ++k) {
					k1 = indx2[k - 1];
					temp1 += ws[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
				}
				for (k = ileave; k <= n; ++k) {
					k1 = indx2[k - 1];
					temp3 += ws[k1 - 1][ipntr - 1] * wy[k1 - 1][jpntr - 1];
				}
				if (is <= jy + m) {
					wn1[is - 1][jy - 1] += (temp1 - temp3);
				} else {
					wn1[is - 1][jy - 1] += (-temp1 + temp3);
				}
				jpntr = (jpntr % m) + 1;
			}
			ipntr = (ipntr % m) + 1;
		}

		// Form the upper triangle of WN = [D+Y' ZZ'Y/theta -L_a'+R_z' ]
		// [-L_a +R_z S'AA'S*theta]
		m2 = 2 * m;
		for (iy = 1; iy <= col; ++iy) {
			is = col + iy;
			is1 = m + iy;
			for (jy = 1; jy <= iy; ++jy) {
				js = col + jy;
				js1 = m + jy;
				wn[jy - 1][iy - 1] = wn1[iy - 1][jy - 1] / theta;
				wn[js - 1][is - 1] = wn1[is1 - 1][js1 - 1] * theta;
			}
			for (jy = 1; jy <= iy - 1; ++jy) {
				wn[jy - 1][is - 1] = -wn1[is1 - 1][jy - 1];
			}
			for (jy = iy; jy <= col; ++jy) {
				wn[jy - 1][is - 1] = wn1[is1 - 1][jy - 1];
			}
			wn[iy - 1][iy - 1] += sy[iy - 1][iy - 1];
		}

		// Form the upper triangle of WN= [ LL' L^-1(-L_a'+R_z')]
		// [(-L_a +R_z)L'^-1 S'AA'S*theta ]
		// first Cholesky factor (1,1) block of wn to get LL'
		// with L' stored in the upper triangle of wn.
		dpofa(wn, 1, 1, m2, col, info);
		if (info[0] != 0) {
			info[0] = -1;
			return;
		}

		// then form L^-1(-L_a'+R_z') in the (1,2) block.
		col2 = 2 * col;
		for (js = col + 1; js <= col2; ++js) {
			for (jaux = 1; jaux <= wn.length; ++jaux) {
				aux[jaux - 1] = wn[1 - 1 + jaux - 1][js - 1];
			}
			dtrsl(wn, m2, col, aux, 1, 11, info);
			for (jaux = 1; jaux <= wn.length; ++jaux) {
				wn[1 - 1 + jaux - 1][js - 1] = aux[jaux - 1];
			}
		}

		// Form S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z') in the
		// upper triangle of (2,2) block of wn.
		for (is = col + 1; is <= col2; ++is) {
			for (js = is; js <= col2; ++js) {

				// wn(is,js) = wn(is,js) + ddot(col,wn(1,is),1,wn(1,js),1)
				double dot = 0.0;
				for (jaux = 1; jaux <= col; ++jaux) {
					dot += wn[1 - 1 + jaux - 1][is - 1] * wn[1 - 1 + jaux - 1][js - 1];
				}
				wn[is - 1][js - 1] += dot;
			}
		}

		// Cholesky factorization of (2,2) block of wn.
		dpofa(wn, col + 1, col + 1, m2, col, info);
		if (info[0] != 0) {
			info[0] = -2;
		}
	}

	private static void formt(final int m, final double[][] wt, final double[][] sy, final double[][] ss, final int col,
			final double theta, final int[] info) {
		int i, j, k, k1;
		double ddum;
		final double zero = 0.0;

		// Form the upper half of T = theta*SS + L*D^(-1)*L',
		// store T in the upper triangle of the array wt.
		BlasMath.dscal1(col, theta, ss[1 - 1], 1, wt[1 - 1], 1);
		for (i = 2; i <= col; ++i) {
			for (j = i; j <= col; ++j) {
				k1 = Math.min(i, j) - 1;
				ddum = zero;
				for (k = 1; k <= k1; ++k) {
					ddum += sy[i - 1][k - 1] * sy[j - 1][k - 1] / sy[k - 1][k - 1];
				}
				wt[i - 1][j - 1] = ddum + theta * ss[i - 1][j - 1];
			}
		}

		// Cholesky factorize T to J*J' with
		// J' stored in the upper triangle of wt.
		dpofa(wt, 1, 1, m, col, info);
		if (info[0] != 0) {
			info[0] = -3;
		}
	}

	private static void freev(final int n, final int[] nfree, final int[] index, final int[] nenter, final int[] ileave,
			final int[] indx2, final int[] iwhere, final boolean[] wrk, final boolean updatd, final boolean cnstnd,
			final int iprint, final int iter) {

		int iact, i, k;

		nenter[0] = 0;
		ileave[0] = n + 1;
		if (iter > 0 && cnstnd) {

			// count the entering and leaving variables.
			for (i = 1; i <= nfree[0]; ++i) {
				k = index[i - 1];
				if (iwhere[k - 1] > 0) {
					--ileave[0];
					indx2[ileave[0] - 1] = k;
				}
			}
			for (i = 1 + nfree[0]; i <= n; ++i) {
				k = index[i - 1];
				if (iwhere[k - 1] <= 0) {
					++nenter[0];
					indx2[nenter[0] - 1] = k;
				}
			}
		}
		wrk[0] = (ileave[0] < n + 1) || (nenter[0] > 0) || updatd;

		// Find the index set of free and active variables at the GCP.
		nfree[0] = 0;
		iact = n + 1;
		for (i = 1; i <= n; ++i) {
			if (iwhere[i - 1] <= 0) {
				++nfree[0];
				index[nfree[0] - 1] = i;
			} else {
				--iact;
				index[iact - 1] = i;
			}
		}
	}

	private static void hpsolb(final int n, final double[] t, final int[] iorder, final int iheap) {

		int i, j, k, indxin, indxou;
		double ddum, out;

		if (iheap == 0) {

			// Rearrange the elements t(1) to t(n) to form a heap.
			for (k = 2; k <= n; ++k) {
				ddum = t[k - 1];
				indxin = iorder[k - 1];

				// Add ddum to the heap.
				i = k;
				while (i > 1) {
					j = i / 2;
					if (ddum < t[j - 1]) {
						t[i - 1] = t[j - 1];
						iorder[i - 1] = iorder[j - 1];
						i = j;
					} else {
						break;
					}
				}
				t[i - 1] = ddum;
				iorder[i - 1] = indxin;
			}
		}

		// Assign to 'out' the value of t(1), the least member of the heap,
		// and rearrange the remaining members to form a heap as
		// elements 1 to n-1 of t.
		if (n > 1) {
			i = 1;
			out = t[1 - 1];
			indxou = iorder[1 - 1];
			ddum = t[n - 1];
			indxin = iorder[n - 1];

			// Restore the heap
			while (true) {
				j = i + i;
				if (j <= n - 1) {
					if (t[j + 1 - 1] < t[j - 1]) {
						++j;
					}
					if (t[j - 1] < ddum) {
						t[i - 1] = t[j - 1];
						iorder[i - 1] = iorder[j - 1];
						i = j;
					} else {
						break;
					}
				} else {
					break;
				}
			}
			t[i - 1] = ddum;
			iorder[i - 1] = indxin;

			// Put the least member in t(n).
			t[n - 1] = out;
			iorder[n - 1] = indxou;
		}
	}

	private static void lnsrlb(final int n, final double[] l, final double[] u, final int[] nbd, final double[] x,
			final double[] f, final double[] fold, final double[] gd, final double[] gdold, final double[] g,
			final double[] d, final double[] r, final double[] t, final double[] z, final double[] stp,
			final double[] dnorm, final double[] dtd, final double[] xstep, final double[] stpmx, final int iter,
			final int[] ifun, final int[] iback, final int[] nfgv, final int[] info, final String[] task,
			final boolean boxed, final boolean cnstnd, final String[] csave, final int[] isave, final int iisave,
			final double[] dsave, final int idsave) {

		int i;
		double a1, a2;
		final double one = 1.0, zero = 0.0, big = 1.0e10, ftol = 1.0e-3, gtol = 0.9, xtol = 0.1;

		if (!"FG_LN".equals(task[0].substring(0, 5))) {

			dtd[0] = BlasMath.ddotm(n, d, 1, d, 1);
			dnorm[0] = Math.sqrt(dtd[0]);

			// Determine the maximum step length.
			stpmx[0] = big;
			if (cnstnd) {
				if (iter == 0) {
					stpmx[0] = one;
				} else {
					for (i = 1; i <= n; ++i) {
						a1 = d[i - 1];
						if (nbd[i - 1] != 0) {
							if (a1 < zero && nbd[i - 1] <= 2) {
								a2 = l[i - 1] - x[i - 1];
								if (a2 >= zero) {
									stpmx[0] = zero;
								} else if (a1 * stpmx[0] < a2) {
									stpmx[0] = a2 / a1;
								}
							} else if (a1 > zero && nbd[i - 1] >= 2) {
								a2 = u[i - 1] - x[i - 1];
								if (a2 <= zero) {
									stpmx[0] = zero;
								} else if (a1 * stpmx[0] > a2) {
									stpmx[0] = a2 / a1;
								}
							}
						}
					}
				}
			}

			if (iter == 0 && !boxed) {
				stp[0] = Math.min(one / dnorm[0], stpmx[0]);
			} else {
				stp[0] = one;
			}
			System.arraycopy(x, 0, t, 0, n);
			System.arraycopy(g, 0, r, 0, n);
			fold[0] = f[0];
			ifun[0] = iback[0] = 0;
			csave[0] = "START";
		}

		gd[0] = BlasMath.ddotm(n, g, 1, d, 1);
		if (ifun[0] == 0) {
			gdold[0] = gd[0];
			if (gd[0] >= zero) {

				// the directional derivative >=0. Line search is impossible.
				info[0] = -4;
				return;
			}
		}

		dcsrch(f, gd, stp, ftol, gtol, xtol, zero, stpmx[0], csave, isave, iisave, dsave, idsave);
		xstep[0] = stp[0] * dnorm[0];
		if (!(csave[0].length() >= 4 && "CONV".equals(csave[0].substring(0, 4)))
				&& !(csave[0].length() >= 4 && "WARN".equals(csave[0].substring(0, 4)))) {
			task[0] = "FG_LNSRCH";
			++ifun[0];
			++nfgv[0];
			iback[0] = ifun[0] - 1;
			if (stp[0] == one) {
				System.arraycopy(z, 0, x, 0, n);
			} else {
				BlasMath.daxpy1(n, stp[0], d, 1, t, 1, x, 1);
			}
		} else {
			task[0] = "NEW_X";
		}
	}

	private static void matupd(final int n, final int m, final double[][] ws, final double[][] wy, final double[][] sy,
			final double[][] ss, final double[] d, final double[] r, final int[] itail, final int iupdat,
			final int[] col, final int[] head, final double[] theta, final double rr, final double dr, final double stp,
			final double dtd) {

		int j, jaux, pointr;
		final double one = 1.0;

		// Set pointers for matrices WS and WY.
		if (iupdat <= m) {
			col[0] = iupdat;
			itail[0] = ((head[0] + iupdat - 2) % m) + 1;
		} else {
			itail[0] = (itail[0] % m) + 1;
			head[0] = (head[0] % m) + 1;
		}

		// Update matrices WS and WY.
		// call dcopy(n,d,1,ws(1,itail),1)
		// call dcopy(n,r,1,wy(1,itail),1)
		for (jaux = 1; jaux <= n; ++jaux) {
			ws[1 - 1 + jaux - 1][itail[0] - 1] = d[1 - 1 + jaux - 1];
			wy[1 - 1 + jaux - 1][itail[0] - 1] = r[1 - 1 + jaux - 1];
		}

		// Set theta=yy/ys.
		theta[0] = rr / dr;

		// Form the middle matrix in B.
		// update the upper triangle of SS, and the lower triangle of SY:
		if (iupdat > m) {

			// move old information
			for (j = 1; j <= col[0] - 1; ++j) {

				// call dcopy(j,ss(2,j+1),1,ss(1,j),1)
				for (jaux = 1; jaux <= j; ++jaux) {
					ss[1 - 1 + jaux - 1][j - 1] = ss[2 - 1 + jaux - 1][j + 1 - 1];
				}

				// call dcopy(col-j,sy(j+1,j+1),1,sy(j,j),1)
				for (jaux = 1; jaux <= col[0] - j; ++jaux) {
					sy[j - 1 + jaux - 1][j - 1] = sy[j + 1 - 1 + jaux - 1][j + 1 - 1];
				}
			}
		}

		// add new information: the last row of SY and the last column of SS:
		pointr = head[0];
		for (j = 1; j <= col[0] - 1; ++j) {

			// sy(col,j) = ddot(n,d,1,wy(1,pointr),1)
			// ss(j,col) = ddot(n,ws(1,pointr),1,d,1)
			double dot1 = 0.0;
			double dot2 = 0.0;
			for (jaux = 1; jaux <= n; ++jaux) {
				dot1 += d[1 - 1 + jaux - 1] * wy[1 - 1 + jaux - 1][pointr - 1];
				dot2 += ws[1 - 1 + jaux - 1][pointr - 1] * d[1 - 1 + jaux - 1];
			}
			sy[col[0] - 1][j - 1] = dot1;
			ss[j - 1][col[0] - 1] = dot2;
			pointr = (pointr % m) + 1;
		}
		if (stp == one) {
			ss[col[0] - 1][col[0] - 1] = dtd;
		} else {
			ss[col[0] - 1][col[0] - 1] = stp * stp * dtd;
		}
		sy[col[0] - 1][col[0] - 1] = dr;
	}

	private static void projgr(final int n, final double[] l, final double[] u, final int[] nbd, final double[] x,
			final double[] g, final double[] sbgnrm) {

		int i;
		double gi;
		final double one = 1.0, zero = 0.0;

		sbgnrm[0] = zero;
		for (i = 1; i <= n; ++i) {
			gi = g[i - 1];
			if (nbd[i - 1] != 0) {
				if (gi < zero) {
					if (nbd[i - 1] >= 2) {
						gi = Math.max(x[i - 1] - u[i - 1], gi);
					}
				} else if (nbd[i - 1] <= 2) {
					gi = Math.min(x[i - 1] - l[i - 1], gi);
				}
			}
			sbgnrm[0] = Math.max(sbgnrm[0], Math.abs(gi));
		}
	}

	private static void subsm(final int n, final int m, final int nsub, final int[] ind, final double[] l,
			final double[] u, final int[] nbd, final double[] x, final double[] d, final double[] xp,
			final double[][] ws, final double[][] wy, final double theta, final double[] xx, final double[] gg,
			final int col, final int head, final int[] iword, final double[] wv, final double[][] wn, final int iprint,
			final int[] info) {

		int pointr, m2, col2, ibd, jy, js, i, j, k;
		double alpha, xk, dk, temp1, temp2, dd_p;
		final double one = 1.0, zero = 0.0;

		if (nsub <= 0) {
			return;
		}

		// Compute wv = W'Zd.
		pointr = head;
		for (i = 1; i <= col; ++i) {
			temp1 = temp2 = zero;
			for (j = 1; j <= nsub; ++j) {
				k = ind[j - 1];
				temp1 += wy[k - 1][pointr - 1] * d[j - 1];
				temp2 += ws[k - 1][pointr - 1] * d[j - 1];
			}
			wv[i - 1] = temp1;
			wv[col + i - 1] = theta * temp2;
			pointr = (pointr % m) + 1;
		}

		// Compute wv:=K^(-1)wv.
		m2 = 2 * m;
		col2 = 2 * col;
		dtrsl(wn, m2, col2, wv, 1, 11, info);
		if (info[0] != 0) {
			return;
		}
		for (i = 1; i <= col; ++i) {
			wv[i - 1] = -wv[i - 1];
		}
		dtrsl(wn, m2, col2, wv, 1, 01, info);
		if (info[0] != 0) {
			return;
		}

		// Compute d = (1/theta)d + (1/theta**2)Z'W wv.
		pointr = head;
		for (jy = 1; jy <= col; ++jy) {
			js = col + jy;
			for (i = 1; i <= nsub; ++i) {
				k = ind[i - 1];
				d[i - 1] += wy[k - 1][pointr - 1] * wv[jy - 1] / theta + ws[k - 1][pointr - 1] * wv[js - 1];
			}
			pointr = (pointr % m) + 1;
		}
		BlasMath.dscalm(nsub, one / theta, d, 1);

		// -----------------------------------------------------------------
		// Let us try the projection, d is the Newton direction
		iword[0] = 0;
		System.arraycopy(x, 0, xp, 0, n);
		for (i = 1; i <= nsub; ++i) {
			k = ind[i - 1];
			dk = d[i - 1];
			xk = x[k - 1];
			if (nbd[k - 1] != 0) {

				switch (nbd[k - 1]) {
				case 1:

					// ! lower bounds only
					x[k - 1] = Math.max(l[k - 1], xk + dk);
					if (x[k - 1] == l[k - 1]) {
						iword[0] = 1;
					}
					break;
				case 2:

					// ! upper and lower bounds
					xk = Math.max(l[k - 1], xk + dk);
					x[k - 1] = Math.min(u[k - 1], xk);
					if (x[k - 1] == l[k - 1] || x[k - 1] == u[k - 1]) {
						iword[0] = 1;
					}
					break;
				case 3:

					// ! upper bounds only
					x[k - 1] = Math.min(u[k - 1], xk + dk);
					if (x[k - 1] == u[k - 1]) {
						iword[0] = 1;
					}
					break;
				default:
					break;
				}
			} else {

				// ! free variables
				x[k - 1] = xk + dk;
			}
		}

		if (iword[0] == 0) {
			return;
		}

		// check sign of the directional derivative
		dd_p = zero;
		for (i = 1; i <= n; ++i) {
			dd_p += (x[i - 1] - xx[i - 1]) * gg[i - 1];
		}
		if (dd_p > zero) {
			System.arraycopy(xp, 0, x, 0, n);
		} else {
			return;
		}

		// -----------------------------------------------------------------
		alpha = one;
		temp1 = alpha;
		ibd = 0;
		for (i = 1; i <= nsub; ++i) {
			k = ind[i - 1];
			dk = d[i - 1];
			if (nbd[k - 1] != 0) {
				if (dk < zero && nbd[k - 1] <= 2) {
					temp2 = l[k - 1] - x[k - 1];
					if (temp2 >= zero) {
						temp1 = zero;
					} else if (dk * alpha < temp2) {
						temp1 = temp2 / dk;
					}
				} else if (dk > zero && nbd[k - 1] >= 2) {
					temp2 = u[k - 1] - x[k - 1];
					if (temp2 <= zero) {
						temp1 = zero;
					} else if (dk * alpha > temp2) {
						temp1 = temp2 / dk;
					}
				}
				if (temp1 < alpha) {
					alpha = temp1;
					ibd = i;
				}
			}
		}
		if (alpha < one) {
			dk = d[ibd - 1];
			k = ind[ibd - 1];
			if (dk > zero) {
				x[k - 1] = u[k - 1];
				d[ibd - 1] = zero;
			} else if (dk < zero) {
				x[k - 1] = l[k - 1];
				d[ibd - 1] = zero;
			}
		}
		for (i = 1; i <= nsub; ++i) {
			k = ind[i - 1];
			x[k - 1] += alpha * d[i - 1];
		}
	}

	private static void dcsrch(final double[] f, final double[] g, final double[] stp, final double ftol,
			final double gtol, final double xtol, final double stpmin, final double stpmax, final String[] task,
			final int[] isave, final int iisave, final double[] dsave, final int idsave) {

		final double[] stx = new double[1], fxm = new double[1], fgm = new double[1], sty = new double[1],
				fym = new double[1], gym = new double[1], gxm = new double[1], fx = new double[1], gx = new double[1],
				fy = new double[1], gy = new double[1];
		final boolean[] brackt = new boolean[1];
		double finit, ginit, ftest, gtest, width, width1, stmin, stmax, fm, gm;
		int stage;
		final double zero = 0.0, p5 = 0.5, p66 = 0.66, xtrapl = 1.1, xtrapu = 4.0;

		// Initialization block.
		if (task[0].length() >= 5 && "START".equals(task[0].substring(0, 5))) {

			// Check the input arguments for errors.
			if (stp[0] < stpmin) {
				task[0] = "ERROR: STP .LT. STPMIN";
			}
			if (stp[0] > stpmax) {
				task[0] = "ERROR: STP .GT. STPMAX";
			}
			if (g[0] >= zero) {
				task[0] = "ERROR: INITIAL G .GE. ZERO";
			}
			if (ftol < zero) {
				task[0] = "ERROR: FTOL .LT. ZERO";
			}
			if (gtol < zero) {
				task[0] = "ERROR: GTOL .LT. ZERO";
			}
			if (xtol < zero) {
				task[0] = "ERROR: XTOL .LT. ZERO";
			}
			if (stpmin < zero) {
				task[0] = "ERROR: STPMIN .LT. ZERO";
			}
			if (stpmax < stpmin) {
				task[0] = "ERROR: STPMAX .LT. STPMIN";
			}

			// Exit if there are errors on input.
			if ("ERROR".equals(task[0].substring(0, 5))) {
				return;
			}

			// Initialize local variables.
			brackt[0] = false;
			stage = 1;
			finit = f[0];
			ginit = g[0];
			gtest = ftol * ginit;
			width = stpmax - stpmin;
			width1 = width / p5;

			// The variables stx, fx, gx contain the values of the step,
			// function, and derivative at the best step.
			// The variables sty, fy, gy contain the value of the step,
			// function, and derivative at sty.
			// The variables stp, f, g contain the values of the step,
			// function, and derivative at stp.
			stx[0] = zero;
			fx[0] = finit;
			gx[0] = ginit;
			sty[0] = zero;
			fy[0] = finit;
			gy[0] = ginit;
			stmin = zero;
			stmax = stp[0] + xtrapu * stp[0];
			task[0] = "FG";

			// Save local variables.
			if (brackt[0]) {
				isave[1 - 1 + iisave - 1] = 1;
			} else {
				isave[1 - 1 + iisave - 1] = 0;
			}
			isave[2 - 1 + iisave - 1] = stage;
			dsave[1 - 1 + idsave - 1] = ginit;
			dsave[2 - 1 + idsave - 1] = gtest;
			dsave[3 - 1 + idsave - 1] = gx[0];
			dsave[4 - 1 + idsave - 1] = gy[0];
			dsave[5 - 1 + idsave - 1] = finit;
			dsave[6 - 1 + idsave - 1] = fx[0];
			dsave[7 - 1 + idsave - 1] = fy[0];
			dsave[8 - 1 + idsave - 1] = stx[0];
			dsave[9 - 1 + idsave - 1] = sty[0];
			dsave[10 - 1 + idsave - 1] = stmin;
			dsave[11 - 1 + idsave - 1] = stmax;
			dsave[12 - 1 + idsave - 1] = width;
			dsave[13 - 1 + idsave - 1] = width1;
			return;
		} else {

			// Restore local variables.
			brackt[0] = isave[1 - 1 + iisave - 1] == 1;
			stage = isave[2 - 1 + iisave - 1];
			ginit = dsave[1 - 1 + idsave - 1];
			gtest = dsave[2 - 1 + idsave - 1];
			gx[0] = dsave[3 - 1 + idsave - 1];
			gy[0] = dsave[4 - 1 + idsave - 1];
			finit = dsave[5 - 1 + idsave - 1];
			fx[0] = dsave[6 - 1 + idsave - 1];
			fy[0] = dsave[7 - 1 + idsave - 1];
			stx[0] = dsave[8 - 1 + idsave - 1];
			sty[0] = dsave[9 - 1 + idsave - 1];
			stmin = dsave[10 - 1 + idsave - 1];
			stmax = dsave[11 - 1 + idsave - 1];
			width = dsave[12 - 1 + idsave - 1];
			width1 = dsave[13 - 1 + idsave - 1];
		}

		// If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
		// algorithm enters the second stage.
		ftest = finit + stp[0] * gtest;
		if (stage == 1 && f[0] <= ftest && g[0] >= zero) {
			stage = 2;
		}

		// Test for warnings.
		if (brackt[0] && (stp[0] <= stmin || stp[0] >= stmax)) {
			task[0] = "WARNING: ROUNDING ERRORS PREVENT PROGRESS";
		}
		if (brackt[0] && stmax - stmin <= xtol * stmax) {
			task[0] = "WARNING: XTOL TEST SATISFIED";
		}
		if (stp[0] == stpmax && f[0] <= ftest && g[0] <= gtest) {
			task[0] = "WARNING: STP = STPMAX";
		}
		if (stp[0] == stpmin && (f[0] > ftest || g[0] >= gtest)) {
			task[0] = "WARNING: STP = STPMIN";
		}

		// Test for convergence.
		if (f[0] <= ftest && Math.abs(g[0]) <= gtol * (-ginit)) {
			task[0] = "CONVERGENCE";
		}

		// Test for termination.
		if (task[0].length() >= 4
				&& ("WARN".equals(task[0].substring(0, 4)) || "CONV".equals(task[0].substring(0, 4)))) {

			// Save local variables.
			if (brackt[0]) {
				isave[1 - 1 + iisave - 1] = 1;
			} else {
				isave[1 - 1 + iisave - 1] = 0;
			}
			isave[2 - 1 + iisave - 1] = stage;
			dsave[1 - 1 + idsave - 1] = ginit;
			dsave[2 - 1 + idsave - 1] = gtest;
			dsave[3 - 1 + idsave - 1] = gx[0];
			dsave[4 - 1 + idsave - 1] = gy[0];
			dsave[5 - 1 + idsave - 1] = finit;
			dsave[6 - 1 + idsave - 1] = fx[0];
			dsave[7 - 1 + idsave - 1] = fy[0];
			dsave[8 - 1 + idsave - 1] = stx[0];
			dsave[9 - 1 + idsave - 1] = sty[0];
			dsave[10 - 1 + idsave - 1] = stmin;
			dsave[11 - 1 + idsave - 1] = stmax;
			dsave[12 - 1 + idsave - 1] = width;
			dsave[13 - 1 + idsave - 1] = width1;
			return;
		}

		// A modified function is used to predict the step during the
		// first stage if a lower function value has been obtained but
		// the decrease is not sufficient.
		if (stage == 1 && f[0] <= fx[0] && f[0] > ftest) {

			// Define the modified function and derivative values.
			fm = f[0] - stp[0] * gtest;
			fxm[0] = fx[0] - stx[0] * gtest;
			fym[0] = fy[0] - sty[0] * gtest;
			gm = g[0] - gtest;
			gxm[0] = gx[0] - gtest;
			gym[0] = gy[0] - gtest;

			// Call dcstep to update stx, sty, and to compute the new step.
			dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);

			// Reset the function and derivative values for f.
			fx[0] = fxm[0] + stx[0] * gtest;
			fy[0] = fym[0] + sty[0] * gtest;
			gx[0] = gxm[0] + gtest;
			gy[0] = gym[0] + gtest;
		} else {

			// Call dcstep to update stx, sty, and to compute the new step.
			dcstep(stx, fx, gx, sty, fy, gy, stp, f[0], g[0], brackt, stmin, stmax);
		}

		// Decide if a bisection step is needed.
		if (brackt[0]) {
			if (Math.abs(sty[0] - stx[0]) >= p66 * width1) {
				stp[0] = stx[0] + p5 * (sty[0] - stx[0]);
			}
			width1 = width;
			width = Math.abs(sty[0] - stx[0]);
		}

		// Set the minimum and maximum steps allowed for stp.
		if (brackt[0]) {
			stmin = Math.min(stx[0], sty[0]);
			stmax = Math.max(stx[0], sty[0]);
		} else {
			stmin = stp[0] + xtrapl * (stp[0] - stx[0]);
			stmax = stp[0] + xtrapu * (stp[0] - stx[0]);
		}

		// Force the step to be within the bounds stpmax and stpmin.
		stp[0] = Math.max(stp[0], stpmin);
		stp[0] = Math.min(stp[0], stpmax);

		// If further progress is not possible, let stp be the best
		// point obtained during the search.
		if (brackt[0] && (stp[0] <= stmin || stp[0] >= stmax) || (brackt[0] && stmax - stmin <= xtol * stmax)) {
			stp[0] = stx[0];
		}

		// Obtain another function and derivative.
		task[0] = "FG";

		// Save local variables.
		if (brackt[0]) {
			isave[1 - 1 + iisave - 1] = 1;
		} else {
			isave[1 - 1 + iisave - 1] = 0;
		}
		isave[2 - 1 + iisave - 1] = stage;
		dsave[1 - 1 + idsave - 1] = ginit;
		dsave[2 - 1 + idsave - 1] = gtest;
		dsave[3 - 1 + idsave - 1] = gx[0];
		dsave[4 - 1 + idsave - 1] = gy[0];
		dsave[5 - 1 + idsave - 1] = finit;
		dsave[6 - 1 + idsave - 1] = fx[0];
		dsave[7 - 1 + idsave - 1] = fy[0];
		dsave[8 - 1 + idsave - 1] = stx[0];
		dsave[9 - 1 + idsave - 1] = sty[0];
		dsave[10 - 1 + idsave - 1] = stmin;
		dsave[11 - 1 + idsave - 1] = stmax;
		dsave[12 - 1 + idsave - 1] = width;
		dsave[13 - 1 + idsave - 1] = width1;
	}

	private static void dcstep(final double[] stx, final double[] fx, final double[] dx, final double[] sty,
			final double[] fy, final double[] dy, final double[] stp, final double fp, final double dp,
			final boolean[] brackt, final double stpmin, final double stpmax) {

		double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;
		final double zero = 0.0, p66 = 0.66, two = 2.0, three = 3.0;

		sgnd = dp * (dx[0] / Math.abs(dx[0]));

		if (fp > fx[0]) {

			// First case: A higher function value. The minimum is bracketed.
			// If the cubic step is closer to stx than the quadratic step, the
			// cubic step is taken, otherwise the average of the cubic and
			// quadratic steps is taken.
			theta = three * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			s = Math.max(Math.max(Math.abs(theta), Math.abs(dx[0])), Math.abs(dp));
			gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] < stx[0]) {
				gamma = -gamma;
			}
			p = (gamma - dx[0]) + theta;
			q = ((gamma - dx[0]) + gamma) + dp;
			r = p / q;
			stpc = stx[0] + r * (stp[0] - stx[0]);
			stpq = stx[0] + ((dx[0] / ((fx[0] - fp) / (stp[0] - stx[0]) + dx[0])) / two) * (stp[0] - stx[0]);
			if (Math.abs(stpc - stx[0]) < Math.abs(stpq - stx[0])) {
				stpf = stpc;
			} else {
				stpf = stpc + (stpq - stpc) / two;
			}
			brackt[0] = true;
		} else if (sgnd < zero) {

			// Second case: A lower function value and derivatives of opposite
			// sign. The minimum is bracketed. If the cubic step is farther from
			// stp than the secant step, the cubic step is taken, otherwise the
			// secant step is taken.
			theta = three * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			s = Math.max(Math.max(Math.abs(theta), Math.abs(dx[0])), Math.abs(dp));
			gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] > stx[0]) {
				gamma = -gamma;
			}
			p = (gamma - dp) + theta;
			q = ((gamma - dp) + gamma) + dx[0];
			r = p / q;
			stpc = stp[0] + r * (stx[0] - stp[0]);
			stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0]);
			if (Math.abs(stpc - stp[0]) > Math.abs(stpq - stp[0])) {
				stpf = stpc;
			} else {
				stpf = stpq;
			}
			brackt[0] = true;
		} else if (Math.abs(dp) < Math.abs(dx[0])) {

			// Third case: A lower function value, derivatives of the same sign,
			// and the magnitude of the derivative decreases.
			//
			// The cubic step is computed only if the cubic tends to infinity
			// in the direction of the step or if the minimum of the cubic
			// is beyond stp. Otherwise the cubic step is defined to be the
			// secant step.
			theta = three * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			s = Math.max(Math.max(Math.abs(theta), Math.abs(dx[0])), Math.abs(dp));

			// The case gamma = 0 only arises if the cubic does not tend
			// to infinity in the direction of the step.
			gamma = s * Math.sqrt(Math.max(zero, (theta / s) * (theta / s) - (dx[0] / s) * (dp / s)));
			if (stp[0] > stx[0]) {
				gamma = -gamma;
			}
			p = (gamma - dp) + theta;
			q = (gamma + (dx[0] - dp)) + gamma;
			r = p / q;
			if (r < zero && gamma != zero) {
				stpc = stp[0] + r * (stx[0] - stp[0]);
			} else if (stp[0] > stx[0]) {
				stpc = stpmax;
			} else {
				stpc = stpmin;
			}
			stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0]);

			if (brackt[0]) {

				// A minimizer has been bracketed. If the cubic step is
				// closer to stp than the secant step, the cubic step is
				// taken, otherwise the secant step is taken.
				if (Math.abs(stpc - stp[0]) < Math.abs(stpq - stp[0])) {
					stpf = stpc;
				} else {
					stpf = stpq;
				}
				if (stp[0] > stx[0]) {
					stpf = Math.min(stp[0] + p66 * (sty[0] - stp[0]), stpf);
				} else {
					stpf = Math.max(stp[0] + p66 * (sty[0] - stp[0]), stpf);
				}
			} else {

				// A minimizer has not been bracketed. If the cubic step is
				// farther from stp than the secant step, the cubic step is
				// taken, otherwise the secant step is taken.
				if (Math.abs(stpc - stp[0]) > Math.abs(stpq - stp[0])) {
					stpf = stpc;
				} else {
					stpf = stpq;
				}
				stpf = Math.min(stpmax, stpf);
				stpf = Math.max(stpmin, stpf);
			}
		} else if (brackt[0]) {

			// Fourth case: A lower function value, derivatives of the same sign,
			// and the magnitude of the derivative does not decrease. If the
			// minimum is not bracketed, the step is either stpmin or stpmax,
			// otherwise the cubic step is taken.
			theta = three * (fp - fy[0]) / (sty[0] - stp[0]) + dy[0] + dp;
			s = Math.max(Math.max(Math.abs(theta), Math.abs(dy[0])), Math.abs(dp));
			gamma = s * Math.sqrt((theta / s) * (theta / s) - (dy[0] / s) * (dp / s));
			if (stp[0] > sty[0]) {
				gamma = -gamma;
			}
			p = (gamma - dp) + theta;
			q = ((gamma - dp) + gamma) + dy[0];
			r = p / q;
			stpc = stp[0] + r * (sty[0] - stp[0]);
			stpf = stpc;
		} else if (stp[0] > stx[0]) {
			stpf = stpmax;
		} else {
			stpf = stpmin;
		}

		// Update the interval which contains a minimizer.
		if (fp > fx[0]) {
			sty[0] = stp[0];
			fy[0] = fp;
			dy[0] = dp;
		} else {
			if (sgnd < zero) {
				sty[0] = stx[0];
				fy[0] = fx[0];
				dy[0] = dx[0];
			}
			stx[0] = stp[0];
			fx[0] = fp;
			dx[0] = dp;
		}

		// Compute the new step.
		stp[0] = stpf;
	}

	private static void dtrsl(final double[][] t, final int ldt, final int n, final double[] b, final int ib,
			final int job, final int[] info) {

		// internal variables
		double temp;
		int cas, j, jj, jaux;

		// check for zero diagonal elements.
		for (info[0] = 1; info[0] <= n; ++info[0]) {
			if (t[info[0] - 1][info[0] - 1] == 0.0) {
				return;
			}
		}
		info[0] = 0;

		// determine the task and go to it.
		cas = 1;
		if ((job % 10) != 0) {
			cas = 2;
		}
		if ((job % 100) / 10 != 0) {
			cas += 2;
		}

		switch (cas) {
		case 1:

			// solve t*x=b for t lower triangular
			b[1 - 1 + ib - 1] /= t[1 - 1][1 - 1];
			if (n >= 2) {
				for (j = 2; j <= n; ++j) {
					temp = -b[j - 1 - 1 + ib - 1];

					// call daxpy(n-j+1,temp,t(j,j-1),1,b(j),1)
					for (jaux = 1; jaux <= n - j + 1; ++jaux) {
						b[j - 1 + jaux - 1 + ib - 1] += temp * t[j - 1 + jaux - 1][j - 1 - 1];
					}
					b[j - 1 + ib - 1] /= t[j - 1][j - 1];
				}
			}
			break;
		case 2:

			// solve t*x=b for t upper triangular.
			b[n - 1 + ib - 1] /= t[n - 1][n - 1];
			if (n >= 2) {
				for (jj = 2; jj <= n; ++jj) {
					j = n - jj + 1;
					temp = -b[j + 1 - 1 + ib - 1];

					// call daxpy(j,temp,t(1,j+1),1,b(1),1)
					for (jaux = 1; jaux <= j; ++jaux) {
						b[1 - 1 + jaux - 1 + ib - 1] += temp * t[1 - 1 + jaux - 1][j + 1 - 1];
					}
					b[j - 1 + ib - 1] /= t[j - 1][j - 1];
				}
			}
			break;
		case 3:

			// solve trans(t)*x=b for t lower triangular.
			b[n - 1 + ib - 1] /= t[n - 1][n - 1];
			if (n >= 2) {
				for (jj = 2; jj <= n; ++jj) {
					j = n - jj + 1;

					// b(j) = b(j) - ddot(jj-1,t(j+1,j),1,b(j+1),1)
					for (jaux = 1; jaux <= jj - 1; ++jaux) {
						b[j - 1 + ib - 1] -= t[j + 1 - 1 + jaux - 1][j - 1] * b[j + 1 - 1 + jaux - 1 + ib - 1];
					}
					b[j - 1 + ib - 1] /= t[j - 1][j - 1];
				}
			}
			break;
		default:

			// solve trans(t)*x=b for t upper triangular.
			b[1 - 1 + ib - 1] /= t[1 - 1][1 - 1];
			if (n >= 2) {
				for (j = 2; j <= n; ++j) {

					// b(j) = b(j) - ddot(j-1,t(1,j),1,b(1),1)
					for (jaux = 1; jaux <= j - 1; ++jaux) {
						b[j - 1 + ib - 1] -= t[1 - 1 + jaux - 1][j - 1] * b[1 - 1 + jaux - 1 + ib - 1];
					}
					b[j - 1 + ib - 1] /= t[j - 1][j - 1];
				}
			}
			break;
		}
	}

	private static void dpofa(final double[][] a, final int ia, final int ja, final int lda, final int n,
			final int[] info) {

		double t, s;
		int j, jm1, jaux, k;

		for (j = 1; j <= n; ++j) {
			info[0] = j;
			s = 0.0;
			jm1 = j - 1;
			if (jm1 >= 1) {
				for (k = 1; k <= jm1; ++k) {

					// t = a(k,j) - ddot(k-1,a(1,k),1,a(1,j),1)
					t = a[k - 1 + ia - 1][j - 1 + ja - 1];
					for (jaux = 1; jaux <= k - 1; ++jaux) {
						t -= a[1 - 1 + jaux - 1 + ia - 1][k - 1 + ja - 1]
								* a[1 - 1 + jaux - 1 + ia - 1][j - 1 + ja - 1];
					}
					t /= a[k - 1 + ia - 1][k - 1 + ja - 1];
					a[k - 1 + ia - 1][j - 1 + ja - 1] = t;
					s += t * t;
				}
			}
			s = a[j - 1 + ia - 1][j - 1 + ja - 1] - s;
			if (s <= 0.0) {
				return;
			}
			a[j - 1 + ia - 1][j - 1 + ja - 1] = Math.sqrt(s);
		}
		info[0] = 0;
	}
}
