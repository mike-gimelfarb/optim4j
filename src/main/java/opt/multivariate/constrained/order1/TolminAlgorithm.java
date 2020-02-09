/*
TOLMIN---a TOlerant Linearly-constrained MINimization algorithm.
Copyright (C) 1990 M. J. D. Powell (University of Cambridge)

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
package opt.multivariate.constrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import utils.BlasMath;
import utils.IntMath;
import utils.RealMath;

/**
 * A translation of the algorithm TOLMIN for minimizing a general differentiable
 * non-linear function subject to linear constraints. Originally written by M.
 * J. D. Powell.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Code found at: https://zhangzk.net/software.html
 * 
 * [2] Powell, M. J. D. "A tolerant algorithm for linearly constrained
 * optimization calculations." Mathematical Programming 45.1-3 (1989): 547-566.
 */
public final class TolminAlgorithm {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	private interface Fg {

		double fgcalc(int n, double[] x, double[] g);
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double[] fsave = new double[1];
	private final double myTol;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 */
	public TolminAlgorithm(final double tolerance) {
		myTol = tolerance;
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param func
	 * @param dfunc
	 * @param meq
	 * @param a
	 * @param b
	 * @param xl
	 * @param xu
	 * @param guess
	 * @return
	 */
	public final double[] optimize(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final int meq, final double[][] a, final double[] b,
			final double[] xl, final double[] xu, final double[] guess) {

		// prepare work variables
		final int n = guess.length;
		final int m = a.length;
		final double[] x = Arrays.copyOf(guess, n);
		final double[] par = new double[n];
		final int[] iact = new int[m + 2 * n];
		final int[] nact = new int[1];
		final int[] info = new int[1];

		final Fg fg = (pn, px, pg) -> {
			System.arraycopy(dfunc.apply(px), 0, pg, 0, pn);
			return func.apply(px);
		};

		// call main subroutine
		getmin(fg, n, m, meq, a, b, xl, xu, x, myTol, iact, nact, par, 0, info);
		return info[0] == 1 ? x : null;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void getmin(final Fg fg, final int n, final int m, final int meq, final double[][] a, final double[] b,
			final double[] xl, final double[] xu, final double[] x, final double acc, final int[] iact,
			final int[] nact, final double[] par, final int iprint, final int[] info) {

		final double[] wa1 = new double[n], wa2 = new double[n * n], wa3 = new double[n], wa4 = new double[n],
				wa5 = new double[n], wa6 = new double[m + n + n], wa7 = new double[n], wa8 = new double[n],
				wa9 = new double[n], wa10 = new double[n], wa11 = new double[n];
		minflc(fg, n, m, meq, a, b, xl, xu, x, acc, iact, nact, par, iprint, info, wa1, wa2, wa3, wa4, wa5, wa6, wa7,
				wa8, wa9, wa10, wa11);
	}

	private void minflc(final Fg fg, final int n, final int m, final int meq, final double[][] a, final double[] b,
			final double[] xl, final double[] xu, final double[] x, final double acc, final int[] iact,
			final int[] nact, final double[] par, final int iprint, final int[] info, final double[] g,
			final double[] z, final double[] u, final double[] xbig, final double[] reskt, final double[] bres,
			final double[] d, final double[] ztg, final double[] gm, final double[] xs, final double[] gs) {

		int i, k, nfmax, mp, mtot;
		final int[] meql = new int[1], msat = new int[1], iterc = new int[1], nfvals = new int[1];
		final double[] relacc = new double[1], tol = new double[1], zznorm = new double[1];

		// Initialize ZZNORM, ITERC, NFVALS and NFMAX.
		zznorm[0] = -1.0;
		iterc[0] = nfvals[0] = nfmax = 0;
		if (info[0] > 0) {
			nfmax = info[0];
		}

		// Check the bounds on N, M and MEQ.
		info[0] = 4;
		if (Math.max(1 - n, Math.max(-m, meq * (meq - m))) > 0) {
			return;
		}

		// Initialize RELACC, Z, U and TOL.
		initzu(n, m, xl, xu, x, iact, meql, info, z, u, xbig, relacc);
		tol[0] = Math.max(0.01, 10.0 * relacc[0]);
		if (info[0] == 4) {
			return;
		}

		// Add any equality constraints to the active set.
		if (meq > 0) {
			eqcons(n, m, meq, a, b, xu, iact, meql, info, z, u, relacc[0], xs, gs);
			if (info[0] == 5) {
				return;
			}
		}
		nact[0] = msat[0] = meql[0];

		// Add the bounds to the list of constraints.
		mtot = nact[0];
		for (i = 1; i <= n; ++i) {
			if (xl[i - 1] < xu[i - 1]) {
				mtot += 2;
				iact[mtot - 1 - 1] = m + i;
				iact[mtot - 1] = m + n + i;
			}
		}

		// Try to satisfy the bound constraints.
		getfes(n, m, a, b, xl, xu, x, iact, nact, par, info, g, z, u, xbig, relacc[0], tol, meql[0], msat, mtot, bres,
				d, ztg, gm, reskt, xs, gs);
		if (msat[0] < mtot) {
			info[0] = 6;
			return;
		}

		// Add the ordinary inequalities to the list of constraints.
		if (m > meq) {
			mp = meq + 1;
			for (k = mp; k <= m; ++k) {
				++mtot;
				iact[mtot - 1] = k;
			}
		}

		while (true) {

			// Correct any constraint violations.
			getfes(n, m, a, b, xl, xu, x, iact, nact, par, info, g, z, u, xbig, relacc[0], tol, meql[0], msat, mtot,
					bres, d, ztg, gm, reskt, xs, gs);
			if (msat[0] < mtot) {
				info[0] = 7;
				return;
			} else if (meql[0] == n) {
				return;
			}

			// Minimize the objective function in the case when constraints are
			// treated as degenerate if their residuals are less than TOL.
			minfun(fg, n, m, a, b, xl, xu, x, acc, iact, nact, par, iprint, info, g, z, u, xbig, relacc[0], zznorm,
					tol[0], meql[0], mtot, iterc, nfvals, nfmax, reskt, bres, d, ztg, gm, xs, gs);

			// Reduce TOL if necessary.
			if (tol[0] > relacc[0] && nact[0] > 0) {
				if (nfvals[0] != nfmax) {
					adjtol(n, m, a, b, xl, xu, x, iact, nact[0], xbig, relacc[0], tol, meql[0]);
					continue;
				} else {
					info[0] = 8;
				}
			}
			break;
		}
	}

	private void minfun(final Fg fg, final int n, final int m, final double[][] a, final double[] b, final double[] xl,
			final double[] xu, final double[] x, final double acc, final int[] iact, final int[] nact,
			final double[] par, final int iprint, final int[] info, final double[] g, final double[] z,
			final double[] u, final double[] xbig, final double relacc, final double[] zznorm, final double tol,
			final int meql, final int mtot, final int[] iterc, final int[] nfvals, final int nfmax,
			final double[] reskt, final double[] bres, final double[] d, final double[] ztg, final double[] gm,
			final double[] xs, final double[] gs) {

		int i, iterp, k, nfvalk, iterk;
		double fprev, diff = 0.0, sum;
		final double[] stepcb = new double[1], ddotg = new double[1], relaxf = new double[1], ssqkt = new double[1],
				step = new double[1];
		final int[] indxbd = new int[1], msat = new int[1];

		// f is saved in fsave accross invocations
		// Initialize the minimization calculation.
		msat[0] = mtot;
		iterk = iterc[0];
		nfvalk = nfvals[0];
		if (nfvals[0] == 0 || info[0] == 1) {
			fsave[0] = fg.fgcalc(n, x, g);
			++nfvals[0];
		}
		fprev = Math.abs(fsave[0] + fsave[0] + 1.0);
		iterp = -1;
		if (iprint != 0) {
			iterp = iterc[0] + IntMath.abs(iprint);
			if (iterc[0] == 0) {
				iterp = 0;
			}
		}

		while (true) {

			// Calculate the next search direction.
			conres(n, m, a, b, xl, xu, x, iact, nact, par, g, z, u, xbig, bres, d, ztg, relacc, tol, stepcb, ddotg,
					meql, msat, mtot, indxbd, gm, reskt, xs, gs);

			// Calculate the Kuhn Tucker residual vector.
			ktvec(n, m, a, iact, nact[0], par, g, reskt, z, u, bres, relaxf, meql, ssqkt, xs, gs);

			boolean skipto40 = false;

			// Test for convergence.
			if (ssqkt[0] <= acc * acc) {
				info[0] = 1;

				// Printing from the subroutine.
				if (iprint == 0) {
					return;
				} else {
					iterk = -1;
					iterp = iterc[0] + IntMath.abs(iprint);
					if (iterk >= 0) {
						skipto40 = true;
					} else {
						return;
					}
				}
			}

			if (!skipto40) {
				if (ddotg[0] >= 0.0) {
					info[0] = 2;

					// Printing from the subroutine.
					if (iprint == 0) {
						return;
					} else {
						iterk = -1;
						iterp = iterc[0] + IntMath.abs(iprint);
						if (iterk >= 0) {
							skipto40 = true;
						} else {
							return;
						}
					}
				}
			}

			if (!skipto40) {

				// Test for termination due to no decrease in F.
				if (fsave[0] >= fprev) {
					if (!(tol == relacc || nact[0] == 0) || diff <= 0.0) {
						info[0] = 3;

						// Printing from the subroutine.
						if (iprint == 0) {
							return;
						} else {
							iterk = -1;
							iterp = iterc[0] + IntMath.abs(iprint);
							if (iterk >= 0) {
								skipto40 = true;
							} else {
								return;
							}
						}
					}
				}
			}

			if (!skipto40) {
				diff = fprev - fsave[0];
				fprev = fsave[0];

				// Test that more calls of FGCALC are allowed.
				if (nfvals[0] == nfmax) {
					info[0] = 8;

					// Printing from the subroutine.
					if (iprint == 0) {
						return;
					} else {
						iterk = -1;
						iterp = iterc[0] + IntMath.abs(iprint);
						if (iterk >= 0) {
							skipto40 = true;
						} else {
							return;
						}
					}
				}
			}

			if (!skipto40) {

				// Test whether to reduce TOL and to provide printing.
				if (tol > relacc && iterc[0] > iterk && 0.1 * relaxf[0] >= Math.max(diff, -0.5 * ddotg[0])) {

					// Printing from the subroutine.
					if (iprint == 0) {
						return;
					} else {
						iterk = -1;
						iterp = iterc[0] + IntMath.abs(iprint);
						if (iterk >= 0) {
							skipto40 = true;
						} else {
							return;
						}
					}
				}
			}

			if (!skipto40) {
				if (iterp == iterc[0]) {
					iterp = iterc[0] + IntMath.abs(iprint);
					if (iterk > 0) {
						return;
					}
				}
			}

			while (true) {

				// Calculate the step along the search direction.
				++iterc[0];
				lsrch(fg, n, x, g, d, xs, gs, relacc, stepcb[0], ddotg[0], fsave, step, nfvals, nfmax, bres);
				if (step[0] == 0.0) {
					info[0] = 3;
					sum = 0.0;
					for (i = 1; i <= n; ++i) {
						sum += Math.abs(d[i - 1] * gs[i - 1]);
					}
					if (ddotg[0] + relacc * sum >= 0.0) {
						info[0] = 2;
					}

					// Printing from the subroutine.
					if (iprint == 0) {
						return;
					} else {
						iterk = -1;
						iterp = iterc[0] + IntMath.abs(iprint);
						if (iterk < 0) {
							return;
						}
					}
				} else {
					break;
				}
			}

			// Revise XBIG.
			for (i = 1; i <= n; ++i) {
				xbig[i - 1] = Math.max(xbig[i - 1], Math.abs(x[i - 1]));
			}

			// Revise the second derivative approximation.
			zbfgs(n, x, nact[0], g, z, ztg, xs, gs, zznorm);

			// Add a constraint to the active set if it restricts the step.
			if (step[0] == stepcb[0]) {
				k = iact[indxbd[0] - 1];
				if (k > m) {
					k -= m;
					if (k <= n) {
						x[k - 1] = xl[k - 1];
					} else {
						x[k - n - 1] = xu[k - n - 1];
					}
				}
				addcon(n, m, a, iact, nact, z, u, relacc, indxbd[0], xs, gs);
			}
		}
	}

	private static void conres(final int n, final int m, final double[][] a, final double[] b, final double[] xl,
			final double[] xu, final double[] x, final int[] iact, final int[] nact, final double[] par,
			final double[] g, final double[] z, final double[] u, final double[] xbig, final double[] bres,
			final double[] d, final double[] ztg, final double relacc, final double tol, final double[] stepcb,
			final double[] sumres, final int meql, final int[] msat, final int mtot, final int[] indxbd,
			final double[] gm, final double[] gmnew, final double[] parnew, final double[] cgrad) {

		int i, idiff, j, jm, k, msatk, mdeg, kl;
		double res, resabs, sum, temp;
		final double[] ddotg = new double[1];

		idiff = mtot - msat[0];

		// Calculate and partition the residuals of the inactive constraints,
		// and set the gradient vector when seeking feasibility.
		if (idiff > 0.0) {
			Arrays.fill(g, 0, n, 0.0);
			sumres[0] = 0.0;
		}
		msatk = msat[0];
		mdeg = msat[0] = nact[0];
		kl = meql + 1;

		for (k = kl; k <= mtot; ++k) {
			j = iact[k - 1];

			// Calculate the residual of the current constraint.
			if (j <= m) {
				res = b[j - 1];
				resabs = Math.abs(b[j - 1]);
				for (i = 1; i <= n; ++i) {
					res -= (x[i - 1] * a[j - 1][i - 1]);
					resabs += Math.abs(xbig[i - 1] * a[j - 1][i - 1]);
				}
			} else {
				jm = j - m;
				if (jm <= n) {
					res = x[jm - 1] - xl[jm - 1];
					resabs = Math.abs(xbig[jm - 1]) + Math.abs(xl[jm - 1]);
				} else {
					jm -= n;
					res = xu[jm - 1] - x[jm - 1];
					resabs = Math.abs(xbig[jm - 1]) + Math.abs(xu[jm - 1]);
				}
			}
			bres[j - 1] = res;

			// Set TEMP to the relative residual.
			temp = 0.0;
			if (resabs != 0.0) {
				temp = res / resabs;
			}
			if (k > msatk && temp < 0.0) {
				if (temp + relacc >= 0.0) {
					if (j <= m) {
						sum = Math.abs(b[j - 1]);
						for (i = 1; i <= n; ++i) {
							sum += Math.abs(x[i - 1] * a[j - 1][i - 1]);
						}
					} else {
						jm = j - m;
						if (jm <= n) {
							sum = Math.abs(x[jm - 1]) + Math.abs(xl[jm - 1]);
						} else {
							sum = Math.abs(x[jm - n - 1]) + Math.abs(xu[jm - n - 1]);
						}
					}
					if (Math.abs(res) <= sum * relacc) {
						temp = 0.0;
					}
				}
			}

			// Place the residual in the appropriate position.
			if (k <= nact[0]) {
				continue;
			}
			if (k <= msatk || temp >= 0.0) {
				++msat[0];
				if (msat[0] < k) {
					iact[k - 1] = iact[msat[0] - 1];
				}
				if (temp > tol) {
					iact[msat[0] - 1] = j;
				} else {
					++mdeg;
					iact[msat[0] - 1] = iact[mdeg - 1];
					iact[mdeg - 1] = j;
				}
			} else {

				// Update the gradient and SUMRES if the constraint is violated when
				// seeking feasibility.
				if (j <= m) {
					BlasMath.dxpym(n, a[j - 1], 1, g, 1);
				} else {
					j -= m;
					if (j <= n) {
						g[j - 1] -= 1.0;
					} else {
						g[j - n - 1] += 1.0;
					}
				}
				sumres[0] += Math.abs(res);
			}
		}

		// Seek the next search direction unless CONRES was called from GETFES
		// and feasibility has been achieved.
		stepcb[0] = 0.0;
		if (idiff > 0 && msat[0] == mtot) {
			return;
		}
		getd(n, m, a, iact, nact, par, g, z, u, d, ztg, relacc, ddotg, meql, mdeg, gm, gmnew, parnew, cgrad);

		// Calculate the (bound on the) step-length due to the constraints.
		if (ddotg[0] < 0.0) {
			stepbd(n, m, a, iact, bres, d, stepcb, ddotg, mdeg, msat, mtot, indxbd);
		}
		if (idiff == 0) {
			sumres[0] = ddotg[0];
		}
	}

	private static void newcon(final int n, final int m, final double[][] a, final int[] iact, final int[] nact,
			final double[] z, final double[] u, final double[] d, final double relacc, final int mdeg,
			final double[] zzdiag, final double[] gmnew, final double[] cgrad) {

		int i, iadd = 0, iz, j, jm, jmv, k, khigh, np;
		double cvmax, cviol, sum, sumabs, sumd, temp, savsum = 0.0, savabs = 0.0;

		// Initialization.
		np = nact[0] + 1;
		khigh = mdeg;
		iz = 0;
		for (i = 1; i <= n; ++i) {
			zzdiag[i - 1] = 0.0;
			for (j = np; j <= n; ++j) {
				zzdiag[i - 1] += (z[iz + j - 1] * z[iz + j - 1]);
			}
			iz += n;
		}

		while (true) {

			// Calculate the scalar products of D with its constraints.
			cvmax = 0.0;
			for (k = np; k <= khigh; ++k) {
				j = iact[k - 1];
				if (j <= m) {
					sum = sumabs = sumd = 0.0;
					for (i = 1; i <= n; ++i) {
						temp = d[i - 1] * a[j - 1][i - 1];
						sum += temp;
						sumabs += Math.abs(temp);
						sumd += (zzdiag[i - 1] * a[j - 1][i - 1] * a[j - 1][i - 1]);
					}
				} else {
					jm = j - m;
					if (jm <= n) {
						sum = -d[jm - 1];
					} else {
						jm -= n;
						sum = d[jm - 1];
					}
					sumabs = Math.abs(sum);
					sumd = zzdiag[jm - 1];
				}

				// Pick out the most violated constraint, or return if the
				// violation is negligible.
				if (sum > relacc * sumabs) {
					cviol = sum * sum / sumd;
					if (cviol > cvmax) {
						cvmax = cviol;
						iadd = k;
						savsum = sum;
						savabs = sumabs;
					}
				}
			}
			if (cvmax <= 0.0) {
				return;
			}
			if (nact[0] == 0) {
				k = nact[0];
				addcon(n, m, a, iact, nact, z, u, relacc, iadd, gmnew, cgrad);
				if (nact[0] > k) {
					return;
				}

				// Seek another constraint violation.
				iadd = np;
			} else {

				// Set GMNEW to the gradient of the most violated constraint.
				j = iact[iadd - 1];
				if (j <= m) {
					jmv = 0;
					System.arraycopy(a[j - 1], 0, gmnew, 0, n);
				} else {
					jmv = j - m;
					Arrays.fill(gmnew, 0, n, 0.0);
					if (jmv <= n) {
						gmnew[jmv - 1] = -1.0;
					} else {
						jmv -= n;
						gmnew[jmv - 1] = 1.0;
					}
				}

				// Modify GMNEW for the next active constraint.
				k = nact[0];

				while (true) {

					temp = 0.0;
					iz = k;
					for (i = 1; i <= n; ++i) {
						temp += (z[iz - 1] * gmnew[i - 1]);
						iz += n;
					}
					temp *= u[k - 1];
					j = iact[k - 1];
					if (j <= m) {
						BlasMath.daxpym(n, -temp, a[j - 1], 1, gmnew, 1);
					} else {
						jm = j - m;
						if (jm <= n) {
							gmnew[jm - 1] += temp;
						} else {
							gmnew[jm - n - 1] -= temp;
						}
					}

					// Revise the values of SAVSUM and SAVABS.
					sum = sumabs = 0.0;
					for (i = 1; i <= n; ++i) {
						temp = d[i - 1] * gmnew[i - 1];
						sum += temp;
						sumabs += Math.abs(temp);
					}
					savsum = Math.min(savsum, sum);
					savabs = Math.max(savabs, sumabs);
					--k;
					if (k < 1) {
						break;
					}
				}

				// Add the new constraint to the active set if the constraint
				// violation is still significant.
				if (jmv > 0) {
					d[jmv - 1] = 0.0;
				}
				if (savsum > relacc * savabs) {
					k = nact[0];
					addcon(n, m, a, iact, nact, z, u, relacc, iadd, gmnew, cgrad);
					if (nact[0] > k) {
						return;
					}

					// Seek another constraint violation.
					iadd = np;
				}
			}

			if (np < khigh) {
				k = iact[khigh - 1];
				iact[khigh - 1] = iact[iadd - 1];
				iact[iadd - 1] = k;
				--khigh;
			} else {
				return;
			}
		}
	}

	private static void addcon(final int n, final int m, final double[][] a, final int[] iact, final int[] nact,
			final double[] z, final double[] u, final double relacc, final int indxbd, final double[] ztc,
			final double[] cgrad) {

		int i, ipiv = 0, iz, j, jp, icon, inewbd, iznbd = 0, np;
		double temp, wcos, wsin, wpiv, tempa, tempb, sum, sumabs;

		np = nact[0] + 1;
		icon = iact[indxbd - 1];
		iact[indxbd - 1] = iact[np - 1];
		iact[np - 1] = icon;

		// Form ZTC when the new constraint is a bound.
		if (icon > m) {
			inewbd = icon - m;
			if (inewbd <= n) {
				temp = -1.0;
			} else {
				inewbd -= n;
				temp = 1.0;
			}
			iznbd = inewbd * n - n;
			BlasMath.dscal1(n, temp, z, iznbd + 1, ztc, 1);
		} else {

			// Else form ZTC for an ordinary constraint.
			System.arraycopy(a[icon - 1], 0, cgrad, 0, n);
			for (j = 1; j <= n; ++j) {
				ztc[j - 1] = 0.0;
				iz = j;
				for (i = 1; i <= n; ++i) {
					ztc[j - 1] += (z[iz - 1] * cgrad[i - 1]);
					iz += n;
				}
			}
		}

		// Find any Givens rotations to apply to the last columns of Z.
		j = n;
		while (true) {

			jp = j;
			--j;
			if (j > nact[0]) {
				if (ztc[jp - 1] == 0.0) {
					continue;
				}
				if (Math.abs(ztc[jp - 1]) <= relacc * Math.abs(ztc[j - 1])) {
					temp = Math.abs(ztc[j - 1]);
				} else if (Math.abs(ztc[j - 1]) <= relacc * Math.abs(ztc[jp - 1])) {
					temp = Math.abs(ztc[jp - 1]);
				} else {
					temp = Math.abs(ztc[jp - 1]) * RealMath.hypot(1.0, ztc[j - 1] / ztc[jp - 1]);
				}
				wcos = ztc[j - 1] / temp;
				wsin = ztc[jp - 1] / temp;
				ztc[j - 1] = temp;

				// Apply the rotation when the new constraint is a bound.
				iz = j;
				if (icon > m) {
					for (i = 1; i <= n; ++i) {
						temp = wcos * z[iz + 1 - 1] - wsin * z[iz - 1];
						z[iz - 1] = wcos * z[iz - 1] + wsin * z[iz + 1 - 1];
						z[iz + 1 - 1] = temp;
						iz += n;
					}
					z[iznbd + jp - 1] = 0.0;
				} else {

					// Else apply the rotation for an ordinary constraint.
					wpiv = 0.0;
					for (i = 1; i <= n; ++i) {
						tempa = wcos * z[iz + 1 - 1];
						tempb = wsin * z[iz - 1];
						temp = Math.abs(cgrad[i - 1]) * (Math.abs(tempa) + Math.abs(tempb));
						if (temp > wpiv) {
							wpiv = temp;
							ipiv = i;
						}
						z[iz - 1] = wcos * z[iz - 1] + wsin * z[iz + 1 - 1];
						z[iz + 1 - 1] = tempa - tempb;
						iz += n;
					}

					// Ensure orthogonality of Z(.,JP) to CGRAD.
					sum = 0.0;
					iz = jp;
					for (i = 1; i <= n; ++i) {
						sum += (z[iz - 1] * cgrad[i - 1]);
						iz += n;
					}
					if (sum != 0.0) {
						iz = ipiv * n - n + jp;
						z[iz - 1] -= (sum / cgrad[ipiv - 1]);
					}
				}
			} else {
				break;
			}
		}

		// Test for linear independence in the proposed new active set.
		if (ztc[np - 1] == 0.0) {
			return;
		}
		if (icon <= m) {
			sum = sumabs = 0.0;
			iz = np;
			for (i = 1; i <= n; ++i) {
				temp = z[iz - 1] * cgrad[i - 1];
				sum += temp;
				sumabs += Math.abs(temp);
				iz += n;
			}
			if (Math.abs(sum) <= relacc * sumabs) {
				return;
			}
		}

		// Set the new diagonal element of U and return.
		u[np - 1] = 1.0 / ztc[np - 1];
		nact[0] = np;
	}

	private static void lsrch(final Fg fg, final int n, final double[] x, final double[] g, final double[] d,
			final double[] xs, final double[] gs, final double relacc, final double stepcb, final double ddotg,
			final double[] f, final double[] step, final int[] nfvals, final int nfmax, final double[] gopt) {

		int icount, i;
		double relint, ratio, temp, sbase, stpmin, fbase, ddotgb, stplow, flow, dglow, stphgh, stpopt, fopt, dgopt,
				dgmid, fhgh = 0.0, dghgh = 0.0, dgknot;

		// Initialization.
		relint = 0.9;
		icount = 0;
		ratio = -1.0;
		for (i = 1; i <= n; ++i) {
			xs[i - 1] = x[i - 1];
			gs[i - 1] = g[i - 1];
			gopt[i - 1] = g[i - 1];
			if (d[i - 1] != 0.0) {
				temp = Math.abs(x[i - 1] / d[i - 1]);
				if (ratio < 0.0 || temp < ratio) {
					ratio = temp;
				}
			}
		}
		step[0] = Math.min(1.0, stepcb);

		// The following number 1.0D-12 is independent of the working
		// accuracy of the computer arithmetic.
		stpmin = Math.max(relacc * ratio, 1.0e-12 * step[0]);
		step[0] = Math.max(stpmin, step[0]);
		sbase = 0.0;
		fbase = f[0];
		ddotgb = ddotg;
		stplow = 0.0;
		flow = f[0];
		dglow = ddotg;
		stphgh = stpopt = 0.0;
		fopt = f[0];
		dgopt = Math.abs(ddotg);

		while (true) {

			// Calculate another function and gradient value.
			BlasMath.daxpy1(n, step[0], d, 1, xs, 1, x, 1);
			f[0] = fg.fgcalc(n, x, g);
			++icount;
			dgmid = BlasMath.ddotm(n, d, 1, g, 1);
			if (f[0] <= fopt) {
				if (f[0] < fopt || Math.abs(dgmid) < dgopt) {
					stpopt = step[0];
					fopt = f[0];
					System.arraycopy(g, 0, gopt, 0, n);
					dgopt = Math.abs(dgmid);
				}
			}
			if (nfvals[0] + icount == nfmax) {
				break;
			}

			// Modify the bounds on the steplength or convergence.
			boolean skipto60 = false;
			if (f[0] >= fbase + 0.1 * (step[0] - sbase) * ddotgb) {
				if (stphgh > 0.0 || f[0] > fbase || dgmid > 0.5 * ddotg) {
					stphgh = step[0];
					fhgh = f[0];
					dghgh = dgmid;
					skipto60 = true;
				} else {
					sbase = step[0];
					fbase = f[0];
					ddotgb = dgmid;
				}
			}

			if (!skipto60) {
				if (dgmid >= 0.7 * ddotgb) {
					break;
				}
				stplow = step[0];
				flow = f[0];
				dglow = dgmid;
			}
			if (stphgh > 0.0 && stplow >= relint * stphgh) {
				break;
			}

			// Calculate the next step length or end the iterations.
			if (stphgh == 0.0) {
				if (step[0] == stepcb) {
					break;
				}
				temp = 10.0;
				if (dgmid > 0.9 * ddotg) {
					temp = ddotg / (ddotg - dgmid);
				}
				step[0] = Math.min(temp * step[0], stepcb);
			} else if (icount == 1 || stplow > 0.0) {
				dgknot = 2.0 * (fhgh - flow) / (stphgh - stplow) - 0.5 * (dglow + dghgh);
				if (dgknot >= 0.0) {
					ratio = Math.max(0.1, 0.5 * dglow / (dglow - dgknot));
				} else {
					ratio = (0.5 * dghgh - dgknot) / (dghgh - dgknot);
				}
				step[0] = stplow + ratio * (stphgh - stplow);
			} else {
				step[0] *= 0.1;
				if (step[0] < stpmin) {
					break;
				}
			}
		}

		// Return from subroutine.
		if (step[0] != stpopt) {
			step[0] = stpopt;
			f[0] = fopt;
			for (i = 1; i <= n; ++i) {
				x[i - 1] = xs[i - 1] + step[0] * d[i - 1];
				g[i - 1] = gopt[i - 1];
			}
		}
		nfvals[0] += icount;
	}

	private static void sdegen(final int n, final int m, final double[][] a, final int[] iact, final int[] nact,
			final double[] par, final double[] z, final double[] u, final double[] d, final double[] ztg,
			final double[] gm, final double relacc, final double[] ddotgm, final int meql, final int mdeg,
			final double[] gmnew, final double[] parnew, final double[] cgrad) {

		int i, idrop = 0, itest = 0, iz, j, jm, k, ku, mp, np;
		double dtest, sum, ratio, temp, theta;

		mp = meql + 1;
		dtest = 0.0;

		while (true) {

			// Calculate the search direction and branch if it is not downhill.
			sdirn(n, nact[0], z, d, ztg, gm, relacc, ddotgm);
			if (ddotgm[0] == 0.0) {
				return;
			}

			// Branch if there is no need to consider any degenerate constraints.
			if (nact[0] == mdeg) {
				return;
			}
			np = nact[0] + 1;
			sum = 0.0;
			for (j = np; j <= n; ++j) {
				sum += (ztg[j - 1] * ztg[j - 1]);
			}
			if (dtest > 0.0 && sum >= dtest) {
				if (itest == 1) {
					return;
				}
				itest = 1;
			} else {
				dtest = sum;
				itest = 0;
			}

			// Add a constraint to the active set if there are any significant
			// violations of degenerate constraints.
			k = nact[0];
			newcon(n, m, a, iact, nact, z, u, d, relacc, mdeg, gmnew, parnew, cgrad);
			if (nact[0] == k) {
				return;
			}
			par[nact[0] - 1] = 0.0;

			while (true) {

				// Calculate the new reduced gradient and Lagrange parameters.
				System.arraycopy(gm, 0, gmnew, 0, n);
				k = nact[0];

				while (true) {
					temp = 0.0;
					iz = k;
					for (i = 1; i <= n; ++i) {
						temp += (z[iz - 1] * gmnew[i - 1]);
						iz += n;
					}
					temp *= u[k - 1];
					parnew[k - 1] = par[k - 1] + temp;
					if (k == nact[0]) {
						parnew[k - 1] = Math.min(parnew[k - 1], 0.0);
					}
					j = iact[k - 1];
					if (j <= m) {
						BlasMath.daxpym(n, -temp, a[j - 1], 1, gmnew, 1);
					} else {
						jm = j - m;
						if (jm <= n) {
							gmnew[jm - 1] += temp;
						} else {
							gmnew[jm - n - 1] -= temp;
						}
					}
					--k;
					if (k <= meql) {
						break;
					}
				}

				// Set RATIO for linear interpolation between PAR and PARNEW.
				ratio = 0.0;
				if (mp < nact[0]) {
					ku = nact[0] - 1;
					for (k = mp; k <= ku; ++k) {
						if (parnew[k - 1] > 0.0) {
							ratio = parnew[k - 1] / (parnew[k - 1] - par[k - 1]);
							idrop = k;
						}
					}
				}

				// Apply the linear interpolation.
				theta = 1.0 - ratio;
				for (k = mp; k <= nact[0]; ++k) {
					par[k - 1] = Math.min(theta * parnew[k - 1] + ratio * par[k - 1], 0.0);
				}
				for (i = 1; i <= n; ++i) {
					gm[i - 1] = theta * gmnew[i - 1] + ratio * gm[i - 1];
				}

				// Drop a constraint if RATIO is positive.
				if (ratio > 0.0) {
					delcon(n, m, a, iact, nact, z, u, relacc, idrop);
					for (k = idrop; k <= nact[0]; ++k) {
						par[k - 1] = par[k + 1 - 1];
					}
				} else {
					break;
				}
			}

			// Return if there is no freedom for a new search direction.
			if (nact[0] >= n) {
				break;
			}
		}
		ddotgm[0] = 0.0;
	}

	private static void ktvec(final int n, final int m, final double[][] a, final int[] iact, final int nact,
			final double[] par, final double[] g, final double[] reskt, final double[] z, final double[] u,
			final double[] bres, final double[] relaxf, final int meql, final double[] ssqkt, final double[] parw,
			final double[] resktw) {

		int i, icase, iz, j, jm, k, kk, kl;
		double temp, ssqktw = 0.0;

		// Calculate the Lagrange parameters and the residual vector.
		System.arraycopy(g, 0, reskt, 0, n);

		if (nact > 0) {

			icase = 0;
			while (true) {
				for (kk = 1; kk <= nact; ++kk) {
					k = nact + 1 - kk;
					j = iact[k - 1];
					temp = 0.0;
					iz = k;
					for (i = 1; i <= n; ++i) {
						temp += (z[iz - 1] * reskt[i - 1]);
						iz += n;
					}
					temp *= u[k - 1];
					if (icase == 0) {
						par[k - 1] = 0.0;
					}
					if (k <= meql || par[k - 1] + temp < 0.0) {
						par[k - 1] += temp;
					} else {
						temp = -par[k - 1];
						par[k - 1] = 0.0;
					}
					if (temp != 0.0) {
						if (j <= m) {
							BlasMath.daxpym(n, -temp, a[j - 1], 1, reskt, 1);
						} else {
							jm = j - m;
							if (jm <= n) {
								reskt[jm - 1] += temp;
							} else {
								reskt[jm - n - 1] -= temp;
							}
						}
					}
				}

				// Calculate the sum of squares of the KT residual vector.
				ssqkt[0] = 0.0;
				if (nact == n) {
					return;
				}
				ssqkt[0] += BlasMath.ddotm(n, reskt, 1, reskt, 1);

				// Apply iterative refinement to the residual vector.
				if (icase == 0) {
					icase = 1;
					System.arraycopy(par, 0, parw, 0, nact);
					System.arraycopy(reskt, 0, resktw, 0, n);
					ssqktw = ssqkt[0];
				} else {
					break;
				}
			}

			// Undo the iterative refinement if it does not reduce SSQKT.
			if (ssqktw < ssqkt[0]) {
				System.arraycopy(parw, 0, par, 0, nact);
				System.arraycopy(resktw, 0, reskt, 0, n);
				ssqkt[0] = ssqktw;
			}
		} else {

			// Calculate SSQKT when there are no active constraints.
			ssqkt[0] = BlasMath.ddotm(n, g, 1, g, 1);
		}

		// Predict the reduction in F if one corrects any positive residuals
		// of active inequality constraints.
		relaxf[0] = 0.0;
		if (meql < nact) {
			kl = meql + 1;
			for (k = kl; k <= nact; ++k) {
				j = iact[k - 1];
				if (bres[j - 1] > 0.0) {
					relaxf[0] -= (par[k - 1] * bres[j - 1]);
				}
			}
		}
	}

	private static void stepbd(final int n, final int m, final double[][] a, final int[] iact, final double[] bres,
			final double[] d, final double[] stepcb, final double[] ddotg, final int mdeg, final int[] msat,
			final int mtot, final int[] indxbd) {

		int iflag, j, jm, k, kl;
		double sp, temp;

		// Set steps to constraint boundaries and find the least positive one.
		iflag = 0;
		stepcb[0] = 0.0;
		indxbd[0] = 0;
		k = mdeg;

		boolean do10 = true;
		while (true) {

			if (do10) {
				++k;
				if (k > mtot) {

					// Try to pass through the boundary of a violated constraint.
					if (indxbd[0] <= msat[0]) {
						return;
					}
					iflag = 1;
					k = indxbd[0];
				}
			}

			// Form the scalar product of D with the current constraint normal.
			j = iact[k - 1];
			if (j <= m) {
				sp = BlasMath.ddotm(n, d, 1, a[j - 1], 1);
			} else {
				jm = j - m;
				if (jm <= n) {
					sp = -d[jm - 1];
				} else {
					sp = d[jm - n - 1];
				}
			}

			// The next branch is taken if label 20 was reached via label 50.
			if (iflag != 1) {

				// Set BRES(J) to indicate the status of the j-th constraint.
				if (sp * bres[j - 1] <= 0.0) {
					bres[j - 1] = 0.0;
				} else {
					bres[j - 1] /= sp;
					if (stepcb[0] == 0.0 || bres[j - 1] < stepcb[0]) {
						stepcb[0] = bres[j - 1];
						indxbd[0] = k;
					}
				}
				do10 = true;
				continue;
			}

			++msat[0];
			iact[indxbd[0] - 1] = iact[msat[0] - 1];
			iact[msat[0] - 1] = j;
			bres[j - 1] = 0.0;
			indxbd[0] = msat[0];
			ddotg[0] -= sp;
			if (ddotg[0] < 0.0 && msat[0] < mtot) {

				// Seek the next constraint boundary along the search direction.
				temp = 0.0;
				kl = mdeg + 1;
				for (k = kl; k <= mtot; ++k) {
					j = iact[k - 1];
					if (bres[j - 1] > 0.0) {
						if (temp == 0.0 || bres[j - 1] < temp) {
							temp = bres[j - 1];
							indxbd[0] = k;
						}
					}
				}
				if (temp > 0.0) {
					stepcb[0] = temp;

					// Try to pass through the boundary of a violated constraint.
					if (indxbd[0] <= msat[0]) {
						return;
					}
					iflag = 1;
					k = indxbd[0];
					do10 = false;
					continue;
				}
			}
			break;
		}
	}

	private static void delcon(final int n, final int m, final double[][] a, final int[] iact, final int[] nact,
			final double[] z, final double[] u, final double relacc, final int idrop) {

		int i, ibd, ipiv = 0, izbd = 0, iz, isave, icon, j, jp, nm;
		double rjjp, ujp, sum, temp, tempa, tempb, denom, wcos, wsin, wpiv;

		nm = nact[0] - 1;
		if (idrop == nact[0]) {
			nact[0] = nm;
			return;
		}
		isave = iact[idrop - 1];

		// Cycle through the constraint exchanges that are needed.
		for (j = idrop; j <= nm; ++j) {
			jp = j + 1;
			icon = iact[jp - 1];
			iact[j - 1] = icon;

			// Calculate the (J,JP) element of R.
			if (icon <= m) {
				rjjp = 0.0;
				iz = j;
				for (i = 1; i <= n; ++i) {
					rjjp += (z[iz - 1] * a[icon - 1][i - 1]);
					iz += n;
				}
			} else {
				ibd = icon - m;
				if (ibd <= n) {
					izbd = ibd * n - n;
					rjjp = -z[izbd + j - 1];
				} else {
					ibd -= n;
					izbd = ibd * n - n;
					rjjp = z[izbd + j - 1];
				}
			}

			// Calculate the parameters of the next rotation.
			ujp = u[jp - 1];
			temp = rjjp * ujp;
			denom = Math.abs(temp);
			if (denom * relacc < 1.0) {
				denom = RealMath.hypot(1.0, denom);
			}
			wcos = temp / denom;
			wsin = 1.0 / denom;

			// Rotate Z when a bound constraint is promoted.
			iz = j;
			if (icon > m) {
				for (i = 1; i <= n; ++i) {
					temp = wcos * z[iz + 1 - 1] - wsin * z[iz - 1];
					z[iz - 1] = wcos * z[iz - 1] + wsin * z[iz + 1 - 1];
					z[iz + 1 - 1] = temp;
					iz += n;
				}
				z[izbd + jp - 1] = 0.0;
			} else {

				// Rotate Z when an ordinary constraint is promoted.
				wpiv = 0.0;
				for (i = 1; i <= n; ++i) {
					tempa = wcos * z[iz + 1 - 1];
					tempb = wsin * z[iz - 1];
					temp = Math.abs(a[icon - 1][i - 1]) * (Math.abs(tempa) + Math.abs(tempb));
					if (temp > wpiv) {
						wpiv = temp;
						ipiv = i;
					}
					z[iz - 1] = wcos * z[iz - 1] + wsin * z[iz + 1 - 1];
					z[iz + 1 - 1] = tempa - tempb;
					iz += n;
				}

				// Ensure orthogonality to promoted constraint.
				sum = 0.0;
				iz = jp;
				for (i = 1; i <= n; ++i) {
					sum += (z[iz - 1] * a[icon - 1][i - 1]);
					iz += n;
				}
				if (sum != 0.0) {
					iz = ipiv * n - n + jp;
					z[iz - 1] -= (sum / a[icon - 1][ipiv - 1]);
				}
			}

			// Set the new diagonal elements of U.
			u[jp - 1] = -denom * u[j - 1];
			u[j - 1] = ujp / denom;
		}

		// Return.
		iact[nact[0] - 1] = isave;
		nact[0] = nm;
	}

	private static void satact(final int n, final int m, final double[][] a, final double[] b, final double[] xl,
			final double[] xu, final double[] x, final int[] iact, final int[] nact, final int[] info, final double[] z,
			final double[] u, final double[] xbig, final double relacc, final double tol, final int meql) {

		int i, idrop, iz, j, jx = 0, k;
		double res, resabs, resbig, savex = 0.0, scale, temp, tempa;

		if (nact[0] == 0) {
			return;
		}
		for (k = 1; k <= nact[0]; ++k) {

			// Calculate the next constraint residual.
			j = iact[k - 1];
			if (j <= m) {
				res = b[j - 1];
				resabs = Math.abs(b[j - 1]);
				resbig = resabs;
				for (i = 1; i <= n; ++i) {
					tempa = a[j - 1][i - 1];
					temp = tempa * x[i - 1];
					res -= temp;
					resabs += Math.abs(temp);
					resbig += (Math.abs(tempa) * xbig[i - 1]);
				}
			} else {
				jx = j - m;
				if (jx <= n) {
					res = x[jx - 1] - xl[jx - 1];
					resabs = Math.abs(x[jx - 1]) + Math.abs(xl[jx - 1]);
					resbig = xbig[jx - 1] + Math.abs(xl[jx - 1]);
					savex = xl[jx - 1];
				} else {
					jx -= n;
					res = xu[jx - 1] - x[jx - 1];
					resabs = Math.abs(x[jx - 1]) + Math.abs(xu[jx - 1]);
					resbig = xbig[jx - 1] + Math.abs(xu[jx - 1]);
					savex = xu[jx - 1];
				}
			}

			// Shift X if necessary.
			if (res != 0.0) {
				temp = res / resabs;
				if (k <= meql) {
					temp = -Math.abs(temp);
				}
				if (tol == relacc || temp + relacc < 0.0) {
					info[0] = 1;
					scale = res * u[k - 1];
					iz = k;
					for (i = 1; i <= n; ++i) {
						x[i - 1] += (scale * z[iz - 1]);
						iz += n;
						xbig[i - 1] = Math.max(xbig[i - 1], Math.abs(x[i - 1]));
					}
					if (j > m) {
						x[jx - 1] = savex;
					}
				} else if (res / resbig > tol) {

					// Else flag a constraint deletion if necessary.
					iact[k - 1] = -iact[k - 1];
				}
			}
		}

		// Delete any flagged constraints and then return.
		idrop = nact[0];
		while (true) {
			if (iact[idrop - 1] < 0) {
				iact[idrop - 1] = -iact[idrop - 1];
				delcon(n, m, a, iact, nact, z, u, relacc, idrop);
			}
			--idrop;
			if (idrop <= meql) {
				break;
			}
		}
	}

	private static void zbfgs(final int n, final double[] x, final int nact, final double[] g, final double[] z,
			final double[] ztg, final double[] xs, final double[] gs, final double[] zznorm) {

		int i, iz, k, km, kp, np;
		double dd, dg, temp, wcos, wsin, sum;

		// Test if there is sufficient convexity for the update.
		dd = dg = temp = 0.0;
		for (i = 1; i <= n; ++i) {
			xs[i - 1] = x[i - 1] - xs[i - 1];
			dd += (xs[i - 1] * xs[i - 1]);
			temp += (gs[i - 1] * xs[i - 1]);
			gs[i - 1] = g[i - 1] - gs[i - 1];
			dg += (gs[i - 1] * xs[i - 1]);
		}
		if (dg < 0.1 * Math.abs(temp)) {
			return;
		}

		// Transform the Z matrix.
		k = n;

		while (true) {
			kp = k;
			--k;
			if (k > nact) {
				if (ztg[kp - 1] == 0.0) {
					continue;
				}
				temp = Math.abs(ztg[kp - 1]) * RealMath.hypot(1.0, ztg[k - 1] / ztg[kp - 1]);
				wcos = ztg[k - 1] / temp;
				wsin = ztg[kp - 1] / temp;
				ztg[k - 1] = temp;
				iz = k;
				for (i = 1; i <= n; ++i) {
					temp = wcos * z[iz + 1 - 1] - wsin * z[iz - 1];
					z[iz - 1] = wcos * z[iz - 1] + wsin * z[iz + 1 - 1];
					z[iz + 1 - 1] = temp;
					iz += n;
				}
				continue;
			}
			break;
		}

		// Update the value of ZZNORM.
		if (zznorm[0] < 0.0) {
			zznorm[0] = dd / dg;
		} else {
			temp = Math.sqrt(zznorm[0] * dd / dg);
			zznorm[0] = Math.min(zznorm[0], temp);
			zznorm[0] = Math.max(zznorm[0], 0.1 * temp);
		}

		// Complete the updating of Z.
		np = nact + 1;
		temp = Math.sqrt(dg);
		iz = np;
		for (i = 1; i <= n; ++i) {
			z[iz - 1] = xs[i - 1] / temp;
			iz += n;
		}
		if (np < n) {
			km = np + 1;
			for (k = km; k <= n; ++k) {
				temp = 0.0;
				iz = k;
				for (i = 1; i <= n; ++i) {
					temp += (gs[i - 1] * z[iz - 1]);
					iz += n;
				}
				temp /= dg;
				sum = 0.0;
				iz = k;
				for (i = 1; i <= n; ++i) {
					z[iz - 1] -= (temp * xs[i - 1]);
					sum += (z[iz - 1] * z[iz - 1]);
					iz += n;
				}
				if (sum < zznorm[0]) {
					temp = Math.sqrt(zznorm[0] / sum);
					iz = k;
					for (i = 1; i <= n; ++i) {
						z[iz - 1] *= temp;
						iz += n;
					}
				}
			}
		}
	}

	private static void getfes(final int n, final int m, final double[][] a, final double[] b, final double[] xl,
			final double[] xu, final double[] x, final int[] iact, final int[] nact, final double[] par,
			final int[] info, final double[] g, final double[] z, final double[] u, final double[] xbig,
			final double relacc, final double[] tol, final int meql, final int[] msat, final int mtot,
			final double[] bres, final double[] d, final double[] ztg, final double[] gm, final double[] gmnew,
			final double[] parnew, final double[] cgrad) {

		int i, itest = 0, msatk = 0;
		double sumrsk = 0.0;
		final int[] indxbd = new int[1];
		final double[] stepcb = new double[1], sumres = new double[1];

		// Make the correction to X for the active constraints.
		info[0] = 0;
		int flag = 10;
		while (true) {

			if (flag == 10) {
				satact(n, m, a, b, xl, xu, x, iact, nact, info, z, u, xbig, relacc, tol[0], meql);
				if (info[0] > 0) {
					msat[0] = nact[0];
				}
				if (msat[0] == mtot) {
					return;
				}
				flag = 20;
			}

			if (flag == 20) {

				// Try to correct the infeasibility.
				msatk = msat[0];
				sumrsk = 0.0;
			}

			conres(n, m, a, b, xl, xu, x, iact, nact, par, g, z, u, xbig, bres, d, ztg, relacc, tol[0], stepcb, sumres,
					meql, msat, mtot, indxbd, gm, gmnew, parnew, cgrad);

			// Include the new constraint in the active set.
			if (stepcb[0] > 0.0) {
				for (i = 1; i <= n; ++i) {
					x[i - 1] += (stepcb[0] * d[i - 1]);
					xbig[i - 1] = Math.max(xbig[i - 1], Math.abs(x[i - 1]));
				}
				addcon(n, m, a, iact, nact, z, u, relacc, indxbd[0], gmnew, cgrad);
			}

			// Test whether to continue the search for feasibility.
			if (msat[0] < mtot) {
				if (stepcb[0] != 0.0) {
					if (msatk < msat[0]) {
						flag = 20;
						continue;
					}
					if (sumrsk == 0.0 || sumres[0] < sumrsk) {
						sumrsk = sumres[0];
						itest = 0;
					}
					++itest;
					if (itest <= 2) {
						flag = 30;
						continue;
					}
				}

				// Reduce TOL if it may be too large to allow feasibility.
				if (tol[0] > relacc) {
					adjtol(n, m, a, b, xl, xu, x, iact, nact[0], xbig, relacc, tol, meql);
					flag = 10;
					continue;
				}
			}
			break;
		}
	}

	private static void getd(final int n, final int m, final double[][] a, final int[] iact, final int[] nact,
			final double[] par, final double[] g, final double[] z, final double[] u, final double[] d,
			final double[] ztg, final double relacc, final double[] ddotg, final int meql, final int mdeg,
			final double[] gm, final double[] gmnew, final double[] parnew, final double[] cgrad) {

		int i, iz, j, jm, k;
		double temp;
		final double[] ddotgm = new double[1];

		// Initialize GM and cycle backwards through the active set.
		System.arraycopy(g, 0, gm, 0, n);
		k = nact[0];

		while (k > 0) {

			// Set TEMP to the next multiplier, but reduce the active set if
			// TEMP has an unacceptable sign.
			temp = 0.0;
			iz = k;
			for (i = 1; i <= n; ++i) {
				temp += (z[iz - 1] * gm[i - 1]);
				iz += n;
			}
			temp *= u[k - 1];
			if (k > meql && temp > 0.0) {
				delcon(n, m, a, iact, nact, z, u, relacc, k);

				// Initialize GM and cycle backwards through the active set.
				System.arraycopy(g, 0, gm, 0, n);
				k = nact[0];
				continue;
			}

			// Update GM using the multiplier that has just been calculated.
			j = iact[k - 1];
			if (j <= m) {
				BlasMath.daxpym(n, -temp, a[j - 1], 1, gm, 1);
			} else {
				jm = j - m;
				if (jm <= n) {
					gm[jm - 1] += temp;
				} else {
					gm[jm - n - 1] -= temp;
				}
			}
			par[k - 1] = temp;
			--k;
		}

		// Calculate the search direction and DDOTG.
		ddotg[0] = 0.0;
		if (nact[0] < n) {
			sdegen(n, m, a, iact, nact, par, z, u, d, ztg, gm, relacc, ddotgm, meql, mdeg, gmnew, parnew, cgrad);
			if (ddotgm[0] < 0.0) {
				ddotg[0] += BlasMath.ddotm(n, d, 1, g, 1);
			}
		}
	}

	private static void eqcons(final int n, final int m, final int meq, final double[][] a, final double[] b,
			final double[] xu, final int[] iact, final int[] meql, final int[] info, final double[] z, final double[] u,
			final double relacc, final double[] am, final double[] cgrad) {

		int i, iz, keq, np, j, jm, k;
		double sum, sumabs, rhs, vmult;

		// Try to add the next equality constraint to the active set.
		for (keq = 1; keq <= meq; ++keq) {

			if (meql[0] < n) {
				np = meql[0] + 1;
				iact[np - 1] = keq;
				addcon(n, m, a, iact, meql, z, u, relacc, np, am, cgrad);
				if (meql[0] == np) {
					continue;
				}
			}

			// If linear dependence occurs then find the multipliers of the
			// dependence relation and apply them to the right hand sides.
			sum = b[keq - 1];
			sumabs = Math.abs(b[keq - 1]);
			if (meql[0] > 0) {
				System.arraycopy(a[keq - 1], 0, am, 0, n);
				k = meql[0];

				while (true) {
					vmult = 0.0;
					iz = k;
					for (i = 1; i <= n; ++i) {
						vmult += (z[iz - 1] * am[i - 1]);
						iz += n;
					}
					vmult *= u[k - 1];
					j = iact[k - 1];
					if (j <= m) {
						BlasMath.daxpym(n, -vmult, a[j - 1], 1, am, 1);
						rhs = b[j - 1];
					} else {
						jm = j - m - n;
						am[jm - 1] -= vmult;
						rhs = xu[jm - 1];
					}
					sum -= (rhs * vmult);
					sumabs += Math.abs(rhs * vmult);
					--k;
					if (k < 1) {
						break;
					}
				}
			}

			// Error return if the constraints are inconsistent.
			if (Math.abs(sum) > relacc * sumabs) {
				info[0] = 5;
				return;
			}
		}
	}

	private static void adjtol(final int n, final int m, final double[][] a, final double[] b, final double[] xl,
			final double[] xu, final double[] x, final int[] iact, final int nact, final double[] xbig,
			final double relacc, final double[] tol, final int meql) {

		int i, j, jm, k, kl;
		double viol, res, resabs;

		// Set VIOL to the greatest relative constraint residual
		// of the first NACT constraints.
		viol = 0.0;
		if (nact > meql) {
			kl = meql + 1;
			for (k = kl; k <= nact; ++k) {
				j = iact[k - 1];
				if (j <= m) {
					res = b[j - 1];
					resabs = Math.abs(res);
					for (i = 1; i <= n; ++i) {
						res -= (a[j - 1][i - 1] * x[i - 1]);
						resabs += Math.abs(a[j - 1][i - 1] * xbig[i - 1]);
					}
				} else {
					jm = j - m;
					if (jm <= n) {
						res = x[jm - 1] - xl[jm - 1];
						resabs = xbig[jm - 1] + Math.abs(xl[jm - 1]);
					} else {
						jm -= n;
						res = xu[jm - 1] - x[jm - 1];
						resabs = xbig[jm - 1] + Math.abs(xu[jm - 1]);
					}
				}
				if (res > 0.0) {
					viol = Math.max(viol, res / resabs);
				}
			}
		}

		// Adjust TOL.
		tol[0] = 0.1 * Math.min(tol[0], viol);
		if (tol[0] <= relacc + relacc) {
			tol[0] = relacc;
			for (i = 1; i <= n; ++i) {
				xbig[i - 1] = Math.abs(x[i - 1]);
			}
		}
	}

	private static void sdirn(final int n, final int nact, final double[] z, final double[] d, final double[] ztg,
			final double[] gm, final double relacc, final double[] ddotgm) {

		double sum, sumabs, temp;
		int i, iz, j, np;

		ddotgm[0] = 0.0;
		if (nact >= n) {
			return;
		}

		// Premultiply GM by the transpose of Z.
		np = nact + 1;
		for (j = np; j <= n; ++j) {
			sum = sumabs = 0.0;
			iz = j;
			for (i = 1; i <= n; ++i) {
				temp = z[iz - 1] * gm[i - 1];
				sum += temp;
				sumabs += Math.abs(temp);
				iz += n;
			}
			if (Math.abs(sum) <= relacc * sumabs) {
				sum = 0.0;
			}
			ztg[j - 1] = sum;
		}

		// Form D by premultiplying ZTG by -Z.
		iz = 0;
		for (i = 1; i <= n; ++i) {
			sum = sumabs = 0.0;
			for (j = np; j <= n; ++j) {
				temp = z[iz + j - 1] * ztg[j - 1];
				sum -= temp;
				sumabs += Math.abs(temp);
			}
			if (Math.abs(sum) <= relacc * sumabs) {
				sum = 0.0;
			}
			d[i - 1] = sum;
			iz += n;
		}

		// Test that the search direction is downhill.
		sumabs = 0.0;
		for (i = 1; i <= n; ++i) {
			temp = d[i - 1] * gm[i - 1];
			ddotgm[0] += temp;
			sumabs += Math.abs(temp);
		}
		if (ddotgm[0] + relacc * sumabs >= 0.0) {
			ddotgm[0] = 0.0;
		}
	}

	private static void initzu(final int n, final int m, final double[] xl, final double[] xu, final double[] x,
			final int[] iact, final int[] meql, final int[] info, final double[] z, final double[] u,
			final double[] xbig, final double[] relacc) {

		double ztpar, tempa, tempb;
		int i, iz, j, jact, nn;

		// Set RELACC.
		ztpar = 100.0;
		relacc[0] = 1.0;

		while (true) {
			relacc[0] *= 0.5;
			tempa = ztpar + 0.5 * relacc[0];
			tempb = ztpar + relacc[0];
			if (ztpar >= tempa || tempa >= tempb) {
				break;
			}
		}

		// Seek bound inconsistencies and bound equality constraints.
		meql[0] = 0;
		for (i = 1; i <= n; ++i) {
			if (xl[i - 1] > xu[i - 1]) {
				return;
			}
			if (xl[i - 1] == xu[i - 1]) {
				++meql[0];
			}
		}

		// Initialize U, Z and XBIG.
		jact = 0;
		nn = n * n;
		Arrays.fill(z, 0, nn, 0.0);
		iz = 0;
		for (i = 1; i <= n; ++i) {
			if (xl[i - 1] == xu[i - 1]) {
				x[i - 1] = xu[i - 1];
				++jact;
				u[jact - 1] = 1.0;
				iact[jact - 1] = i + m + n;
				j = jact;
			} else {
				j = i + meql[0] - jact;
			}
			z[iz + j - 1] = 1.0;
			iz += n;
			xbig[i - 1] = Math.abs(x[i - 1]);
		}
		info[0] = 1;
	}
}
