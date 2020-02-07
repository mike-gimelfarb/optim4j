package opt.multivariate.constrained.lp;

import java.util.Arrays;

import opt.Optimizer;
import utils.Constants;
import utils.RealMath;

/**
 * A translation of the revised simplex algorithm for minimizing a linear
 * function subject to linear constraints by Alan Miller originally written by
 * Alfred Morris at the Naval Surface Warfare Center.
 * 
 * [1] Code found at: https://jblevins.org/mirror/amiller/#nswc
 * 
 * @author Michael
 */
public final class RealSimplexAlgorithm extends Optimizer<double[], Double, RealLinearProgram> {

	private final int myMaxIters;
	private int myIters;

	/**
	 *
	 * @param maxIterations
	 */
	public RealSimplexAlgorithm(final int maxIterations) {
		myMaxIters = maxIterations;
	}

	@Override
	public final double[] optimize(final RealLinearProgram lp, final double[] guess) {

		// prepare variables
		final double[][] a = lp.mySimplex.getA();
		final double[] b = lp.mySimplex.getB();
		final double[] c = new double[lp.mySimplex.myD];
		final int numle = lp.mySimplex.myNumLe;
		final int numge = lp.mySimplex.myNumGe;
		final double[] rerr = new double[1];
		final int[] iter = new int[1];

		// the subroutine does a max problem, so take negative of cost
		// for the min problem, e.g. min c*x <--> max -c*x
		for (int i = 0; i < lp.mySimplex.myD; ++i) {
			c[i] = -lp.myCostVec[i];
		}

		// call main subroutine
		final double[] result = smplx(a, b, c, iter, myMaxIters, numle, numge, rerr);
		myIters += iter[0];
		return result;
	}

	/**
	 *
	 * @return
	 */
	public final int countIterations() {
		return myIters;
	}

	/**
	 *
	 */
	public final void resetCounters() {
		myIters = 0;
	}

	private static double[] smplx(final double[][] a, final double[] b0, final double[] c, final int[] iter,
			final int mxiter, final int numle, final int numge, final double[] rerr) {

		// prepare variables
		final int ka = a.length;
		final int m = b0.length;
		final int n0 = a[0].length;
		final int[] ind = { 0 };
		final int[] ibasis = new int[m];
		final double[] x = new double[n0 + numle + numge];
		final double[] z = new double[1];
		final double[][] bi = new double[m][m];
		final double[] c1 = new double[n0 + numle + numge];
		System.arraycopy(c, 0, c1, 0, n0);

		// call main subroutine
		smplx(a, b0, c1, ka, m, n0, ind, ibasis, x, z, iter, mxiter, numle, numge, bi, rerr);
		if (ind[0] == 0 || ind[0] == 6) {
			return Arrays.copyOf(x, n0);
		} else {
			return null;
		}
	}

	private static void smplx(final double[][] a, final double[] b0, final double[] c, final int ka, final int m,
			final int n0, final int[] ind, final int[] ibasis, final double[] x, final double[] z, final int[] iter,
			final int mxiter, final int numle, final int numge, final double[][] bi, final double[] rerr) {
		double eps0 = dpmpar(1);
		double rerrmn = 10.0 * eps0;
		double rerrmx = 1.0e-4;
		if (eps0 < 1.0e-13) {
			rerrmx = 1.0e-5;
		}
		smplx1(a, b0, c, ka, m, n0, ind, ibasis, x, z, iter, mxiter, eps0, rerrmn, rerrmx, rerr, numle, numge, bi);
	}

	private static void smplx1(final double[][] a, final double[] b0, final double[] c, final int ka, final int m,
			final int n0, final int[] ind, final int[] ibasis, final double[] r, final double[] z, final int[] iter,
			final int mxiter, final double eps0, final double rerrmn, final double rerrmx, final double[] rerr,
			final int numle, final int numge, final double[][] bi) {

		int i, ibeg, icount, iend, ii, il, imin = 0, iout = 0, ip = 0, j, jj, jmin, jp = 0, k, kk, ki, kj, l, ll, lrow,
				m0, mcheck, ms, n, npos = 0, nrow, ns, nstep = 0, num, bflag = 0;
		double amax, binorm, bmax, bmin, bnorm, cmin, cons, eps, epsi, ratio, rerr1, rmin, rtol, s, sgn, t, tol, total,
				w, xmax, zero = 0.0, dsum, dsump, dsumn, dt;
		final int[] basis = new int[m + n0], indx = new int[m + n0], ierr = new int[1];
		final double[] xb = new double[m], y = new double[m];

		xmax = dpmpar(3);
		iter[0] = icount = 0;
		mcheck = Math.min(5, 1 + m / 15);
		z[0] = zero;

		// CHECK FOR INPUT ERRORS
		ms = numle + numge;
		if (m < 2 || n0 < 2 || ms > m || ka < m) {
			ind[0] = 5;
			return;
		}
		for (i = 1; i <= m; ++i) {
			if (b0[i - 1] < zero) {
				ind[0] = 5;
				return;
			}
			xb[i - 1] = zero;
		}
		rtol = xmax;
		for (i = 1; i <= n0; ++i) {
			if (c[i - 1] != zero) {
				rtol = Math.min(Math.abs(c[i - 1]), rtol);
			}
		}
		rtol *= rerrmx;

		// FORMATION OF THE IBASIS AND BASIS ARRAYS.
		// (IF IND = 1 THEN THE IBASIS ARRAY IS DEFINED BY THE USER.)
		ns = n0 + numle;
		n = ns + numge;
		if (ind[0] == 0) {
			num = n0 + m;
			for (i = 1; i <= m; ++i) {
				ibasis[i - 1] = n0 + i;
			}
		} else {
			num = n;
			for (i = 1; i <= m; ++i) {
				if (ibasis[i - 1] > n) {
					++num;
				}
			}
		}

		int gotoflag = 32;
		while (true) {

			if (gotoflag == 32) {
				bflag = 0;
				Arrays.fill(basis, 0, n, 0);
				for (i = 1; i <= m; ++i) {
					ki = ibasis[i - 1];
					basis[ki - 1] = 1;
				}
				if (ind[0] != 1) {

					// CALCULATION OF XB AND BI WHEN IND = 0
					rerr[0] = rerrmn;
					for (j = 1; j <= m; ++j) {
						xb[j - 1] = b0[j - 1];
						for (kk = 1; kk <= m; ++kk) {
							bi[kk - 1][j - 1] = zero;
						}
						bi[j - 1][j - 1] = 1.0;
					}
					if (numge == 0) {
						gotoflag = 630;
					} else {
						jmin = numle + 1;
						for (j = jmin; j <= ms; ++j) {
							xb[j - 1] = -xb[j - 1];
							bi[j - 1][j - 1] = -1.0;
						}
						gotoflag = 601;
					}
				} else {
					gotoflag = 100;
				}
			}

			if (gotoflag == 100) {

				// REORDER THE BASIS
				ibeg = 1;
				iend = m;
				for (i = 1; i <= m; ++i) {
					if (ibasis[i - 1] > n0) {
						indx[ibeg - 1] = ibasis[i - 1];
						++ibeg;
					} else {
						indx[iend - 1] = ibasis[i - 1];
						--iend;
					}
				}
				if (iend == m) {
					if (ind[0] == 0) {
						ind[0] = 3;
						gotoflag = 220;
					} else {
						ind[0] = 0;
						num = n0 + m;
						for (i = 1; i <= m; ++i) {
							ibasis[i - 1] = n0 + i;
						}
						gotoflag = 32;
						continue;
					}
				} else {
					System.arraycopy(indx, 0, ibasis, 0, m);

					// REINVERSION OF THE BASIS MATRIX
					for (j = 1; j <= m; ++j) {
						kj = ibasis[j - 1];
						if (kj <= n0) {
							for (kk = 1; kk <= m; ++kk) {
								bi[kk - 1][j - 1] = a[kk - 1][kj - 1];
							}
						} else if (kj <= ns) {
							l = kj - n0;
							for (kk = 1; kk <= m; ++kk) {
								bi[kk - 1][j - 1] = zero;
							}
							bi[l - 1][j - 1] = 1.0;
						} else if (kj <= n) {
							l = kj - n0;
							for (kk = 1; kk <= m; ++kk) {
								bi[kk - 1][j - 1] = zero;
							}
							bi[l - 1][j - 1] = -1.0;
						} else {
							l = kj - n0;
							for (kk = 1; kk <= m; ++kk) {
								bi[kk - 1][j - 1] = zero;
							}
							bi[l - 1][j - 1] = 1.0;
						}
					}
					icount = 0;
					final double[] wa = new double[m];
					for (kk = 1; kk <= m; ++kk) {
						wa[kk - 1] = bi[kk - 1][1 - 1];
					}
					crout1(wa, m, m, iend, indx, y, ierr);
					for (kk = 1; kk <= m; ++kk) {
						bi[kk - 1][1 - 1] = wa[kk - 1];
					}
					if (ierr[0] != 0) {
						if (iter[0] == 0) {
							ind[0] = 5;
							return;
						} else if (bflag == 0) {
							ind[0] = 3;
							gotoflag = 220;
						} else {
							bflag = 0;
							for (ip = 1; ip <= m; ++ip) {
								if (jp == ibasis[ip - 1]) {
									break;
								}
							}
							ibasis[ip - 1] = iout;
							basis[jp - 1] = 0;
							basis[iout - 1] = 1;
							if (iout > n) {
								++num;
							}
							gotoflag = 100;
							continue;
						}
					} else {

						// CHECK THE ACCURACY OF BI AND RESET RERR
						bnorm = zero;
						for (j = 1; j <= m; ++j) {
							kj = ibasis[j - 1];
							if (kj <= n0) {
								total = zero;
								for (kk = 1; kk <= m; ++kk) {
									total += Math.abs(a[kk - 1][kj - 1]);
								}
							} else {
								total = 1.0;
							}
							bnorm = Math.max(bnorm, total);
						}
						binorm = zero;
						for (j = 1; j <= m; ++j) {
							total = zero;
							for (kk = 1; kk <= m; ++kk) {
								total += Math.abs(bi[kk - 1][j - 1]);
							}
							binorm = Math.max(binorm, total);
						}
						rerr[0] = Math.max(rerrmn, eps0 * bnorm * binorm);
						if (rerr[0] > 1.0e-2) {
							if (iter[0] == 0) {
								ind[0] = 5;
								return;
							} else if (bflag == 0) {
								ind[0] = 3;
								gotoflag = 220;
							} else {
								bflag = 0;
								for (ip = 1; ip <= m; ++ip) {
									if (jp == ibasis[ip - 1]) {
										break;
									}
								}
								ibasis[ip - 1] = iout;
								basis[jp - 1] = 0;
								basis[iout - 1] = 1;
								if (iout > n) {
									++num;
								}
								gotoflag = 100;
								continue;
							}
						} else {
							bflag = 0;

							// RECALCULATION OF XB
							for (i = 1; i <= m; ++i) {
								dsump = dsumn = zero;
								for (l = 1; l <= m; ++l) {
									dt = bi[i - 1][l - 1] * b0[l - 1];
									if (dt > zero) {
										dsump += dt;
									} else {
										dsumn += dt;
									}
								}
								xb[i - 1] = dsump + dsumn;
								s = dsump;
								t = dsumn;
								tol = rerrmx * Math.max(s, -t);
								if (Math.abs(xb[i - 1]) < tol) {
									xb[i - 1] = zero;
								}
							}
							gotoflag = 601;
						}
					}
				}
			}

			if (gotoflag == 200) {

				// FIND THE NEXT VECTOR A(--, JP) TO BE INSERTED INTO THE BASIS
				jp = 0;
				rmin = zero;
				if (nstep == 3) {
					rmin = -rtol;
				}
				for (j = 1; j <= n0; ++j) {
					if (basis[j - 1] != 0) {
						continue;
					}
					if (r[j - 1] >= rmin) {
						continue;
					}
					jp = j;
					rmin = r[j - 1];
				}
				if (n0 != n) {
					jmin = n0 + 1;
					rmin *= 1.1;
					for (j = jmin; j <= n; ++j) {
						if (basis[j - 1] != 0) {
							continue;
						}
						if (r[j - 1] >= rmin) {
							continue;
						}
						jp = j;
						rmin = r[j - 1];
					}
				}
				if (jp != 0) {
					gotoflag = 300;
				} else if (nstep < 2) {
					gotoflag = 800;
				} else if (nstep == 2) {

					// COMPLETION OF THE NSTEP = 2 CASE
					for (i = 1; i <= m; ++i) {
						if (ibasis[i - 1] <= n) {
							continue;
						}
						if (xb[i - 1] > zero) {
							gotoflag = 800;
							break;
						}
					}
					if (gotoflag != 800) {
						gotoflag = 680;
					}
				} else {

					// COMPLETION OF THE NSTEP = 3 CASE
					if (rerr[0] > 1.0e-2) {
						if (icount >= 5) {
							gotoflag = 100;
							continue;
						} else {
							ind[0] = 6;
							gotoflag = 800;
						}
					} else {
						ind[0] = 0;
						gotoflag = 800;
					}
				}
			}

			if (gotoflag == 220) {

				// INSERT THE VALUES OF THE ORGINAL, SLACK, AND SURPLUS
				// VARIABLES INTO R, THEN TERMINATE.
				Arrays.fill(r, 0, n, zero);
				for (i = 1; i <= m; ++i) {
					ki = ibasis[i - 1];
					if (ki <= n) {
						r[ki - 1] = xb[i - 1];
					}
				}
				return;
			}

			if (gotoflag == 300) {

				// IF MXITER ITERATIONS HAVE NOT BEEN PERFORMED THEN BEGIN THE
				// NEXT ITERATION. COMPUTE THE JP-TH COLUMN OF BI*A AND STORE IT IN
				// Y.
				if (iter[0] >= mxiter) {
					ind[0] = 2;
					gotoflag = 220;
					continue;
				}
				++iter[0];
				++icount;
				if (jp > ns) {
					l = jp - n0;
					for (kk = 1; kk <= m; ++kk) {
						y[kk - 1] = -bi[kk - 1][l - 1];
					}
				} else if (jp > n0) {
					l = jp - n0;
					for (kk = 1; kk <= m; ++kk) {
						y[kk - 1] = bi[kk - 1][l - 1];
					}
				} else {

					nrow = 0;
					amax = zero;
					for (i = 1; i <= m; ++i) {
						if (a[i - 1][jp - 1] == zero) {
							continue;
						}
						++nrow;
						indx[nrow - 1] = i;
						amax = Math.max(Math.abs(a[i - 1][jp - 1]), amax);
					}
					if (nrow == 0) {
						ind[0] = 4;
						gotoflag = 220;
						continue;
					} else {
						rerr1 = rerrmx * amax;
						for (i = 1; i <= m; ++i) {
							dsum = zero;
							for (ll = 1; ll <= nrow; ++ll) {
								l = indx[ll - 1];
								dsum += bi[i - 1][l - 1] * a[l - 1][jp - 1];
							}
							y[i - 1] = dsum;
							if (Math.abs(y[i - 1]) >= 5e-3) {
								continue;
							}
							bmax = zero;
							for (l = 1; l <= m; ++l) {
								bmax = Math.max(Math.abs(bi[i - 1][l - 1]), bmax);
							}
							tol = rerr1 * bmax;
							if (Math.abs(y[i - 1]) < tol) {
								y[i - 1] = zero;
							}
						}
					}
				}
				for (i = 1; i <= m; ++i) {
					if (y[i - 1] != zero) {
						gotoflag = 360;
						break;
					}
				}
				if (gotoflag != 360) {
					r[jp - 1] = zero;
					--iter[0];
					--icount;
					gotoflag = 200;
					continue;
				}
			}

			if (gotoflag == 360) {

				if (nstep == 2) {

					// FINDING THE VARIABLE XB(IP) TO BE MADE NONBASIC FOR THE NSTEP
					// = 2 CASE
					npos = 0;
					epsi = xmax;
					for (i = 1; i <= m; ++i) {
						if (y[i - 1] <= zero) {
							continue;
						}
						ratio = xb[i - 1] / y[i - 1];
						if (ratio < epsi) {
							epsi = ratio;
							npos = 1;
							indx[1 - 1] = i;
						} else if (ratio > epsi) {
							continue;
						}
						++npos;
						indx[npos - 1] = i;
					}
					if (npos != 0) {
						gotoflag = 460;
					} else if (icount >= 5) {
						gotoflag = 100;
						continue;
					} else {
						ind[0] = 4;
						gotoflag = 220;
						continue;
					}
				} else if (nstep > 2) {

					// FINDING THE VARIABLE XB(IP) TO BE MADE NONBASIC FOR THE NSTEP
					// = 3 CASE
					npos = 0;
					epsi = xmax;
					for (i = 1; i <= m; ++i) {
						if (y[i - 1] < zero) {
							if (ibasis[i - 1] <= n) {
								continue;
							}
							ip = i;
							gotoflag = 500;
							break;
						} else if (y[i - 1] > zero) {
							ratio = xb[i - 1] / y[i - 1];
							if (ratio < epsi) {
								epsi = ratio;
								npos = 1;
								indx[1 - 1] = i;
							} else if (ratio > epsi) {
								continue;
							}
							++npos;
							indx[npos - 1] = i;
						}
					}
					if (gotoflag != 500) {
						if (npos != 0) {
							gotoflag = 460;
						} else if (icount >= 5) {
							gotoflag = 100;
							continue;
						} else {
							ind[0] = 4;
							gotoflag = 220;
							continue;
						}
					}
				} else {

					// FINDING THE VARIABLE XB(IP) TO BE MADE NONBASIC FOR THE NSTEP
					// = 1 CASE
					npos = ip = 0;
					eps = zero;
					epsi = xmax;
					for (i = 1; i <= m; ++i) {
						if (xb[i - 1] < zero || y[i - 1] <= zero) {
							continue;
						}
						ratio = xb[i - 1] / y[i - 1];
						if (ratio < epsi) {
							epsi = ratio;
							npos = 1;
							indx[1 - 1] = i;
							continue;
						} else if (ratio > epsi) {
							continue;
						}
						++npos;
						indx[npos - 1] = i;
					}
					if (npos == 0) {
						for (i = 1; i <= m; ++i) {
							if (xb[i - 1] >= zero || y[i - 1] >= zero) {
								continue;
							}
							ratio = xb[i - 1] / y[i - 1];
							if (ratio < eps) {
								continue;
							}
							eps = ratio;
							ip = i;
						}
						gotoflag = 500;
					} else if (epsi == zero) {
						gotoflag = 460;
					} else {
						for (i = 1; i <= m; ++i) {
							if (xb[i - 1] >= zero || y[i - 1] >= zero) {
								continue;
							}
							ratio = xb[i - 1] / y[i - 1];
							if (ratio > epsi) {
								continue;
							}
							if (ratio < eps) {
								continue;
							}
							eps = ratio;
							ip = i;
						}
						if (ip != 0) {
							gotoflag = 500;
						} else {
							gotoflag = 460;
						}
					}
				}
			}

			if (gotoflag == 460) {

				// TIE BREAKING PROCEDURE
				ip = indx[1 - 1];
				if (npos != 1) {
					ip = 0;
					bmin = cmin = xmax;
					for (ii = 1; ii <= npos; ++ii) {
						i = indx[ii - 1];
						l = ibasis[i - 1];
						if (l > n0) {
							if (l <= n) {
								lrow = l - n0;
								s = b0[lrow - 1];
								if (lrow <= numle) {
									if (s > bmin) {
										continue;
									}
									ip = i;
									bmin = s;
								} else {
									s = -s;
									bmin = Math.min(zero, bmin);
									if (s > bmin) {
										continue;
									}
									ip = i;
									bmin = s;
								}
							} else {
								ip = i;
								gotoflag = 500;
								break;
							}
						}
						if (c[l - 1] <= zero) {
							cmin = Math.min(zero, cmin);
						}
						if (c[l - 1] > cmin) {
							continue;
						}
						imin = i;
						cmin = c[l - 1];
					}
					if (gotoflag != 500) {
						if (cmin <= zero || ip == 0) {
							ip = imin;
						}
					}
				}
				gotoflag = 500;
			}

			if (gotoflag == 500) {

				// TRANSFORMATION OF XB
				if (xb[ip - 1] != zero) {
					cons = xb[ip - 1] / y[ip - 1];
					for (i = 1; i <= m; ++i) {
						s = xb[i - 1];
						xb[i - 1] -= cons * y[i - 1];
						if (xb[i - 1] >= zero) {
							continue;
						}
						if (s >= zero || xb[i - 1] >= rerrmx * s) {
							xb[i - 1] = zero;
						}
					}
					xb[ip - 1] = cons;
				}

				// TRANSFORMATION OF BI
				for (j = 1; j <= m; ++j) {
					if (bi[ip - 1][j - 1] == zero) {
						continue;
					}
					cons = bi[ip - 1][j - 1] / y[ip - 1];
					for (kk = 1; kk <= m; ++kk) {
						bi[kk - 1][j - 1] -= cons * y[kk - 1];
					}
					bi[ip - 1][j - 1] = cons;
				}

				// UPDATING IBASIS AND BASIS
				iout = ibasis[ip - 1];
				ibasis[ip - 1] = jp;
				basis[iout - 1] = 0;
				basis[jp - 1] = 1;
				if (iout > n) {
					--num;
				}

				// CHECK THE ACCURACY OF BI AND RESET RERR
				if (rerr[0] > 1.0e-2) {

					// THE ACCURACY CRITERIA ARE NOT SATISFIED
					if (icount < 5) {
						gotoflag = 600;
					} else {
						bflag = 1;
						gotoflag = 100;
						continue;
					}
				} else {
					k = 0;
					for (j = 1; j <= m; ++j) {
						kj = ibasis[j - 1];
						if (kj > n0) {
							continue;
						}
						total = zero;
						for (kk = 1; kk <= m; ++kk) {
							total += bi[j - 1][kk - 1] * a[kk - 1][kj - 1];
						}
						rerr[0] = Math.max(rerr[0], Math.abs(1.0 - total));
						++k;
						if (k >= mcheck) {
							break;
						}
					}
					if (rerr[0] > 1.0e-2) {

						// THE ACCURACY CRITERIA ARE NOT SATISFIED
						if (icount < 5) {
							gotoflag = 600;
						} else {
							bflag = 1;
							gotoflag = 100;
							continue;
						}
					} else {
						gotoflag = 600;
					}
				}
			}

			if (gotoflag == 600) {

				// SET UP THE R ARRAY FOR THE NSTEP = 1 CASE
				if (nstep == 2) {
					gotoflag = 630;
				} else if (nstep > 2) {

					// UPDATE THE R ARRAY FOR THE NSTEP = 3 CASE
					cons = r[jp - 1];
					for (j = 1; j <= n0; ++j) {
						if (basis[j - 1] != 0) {
							r[j - 1] = zero;
						} else {
							total = zero;
							for (kk = 1; kk <= m; ++kk) {
								total += bi[ip - 1][kk - 1] * a[kk - 1][j - 1];
							}
							r[j - 1] -= cons * total;
							if (r[j - 1] >= zero) {
								continue;
							}
							tol = rerrmx * Math.abs(c[j - 1]);
							if (Math.abs(r[j - 1]) < tol) {
								r[j - 1] = zero;
							}
						}
					}
					if (n0 != ns) {
						jmin = n0 + 1;
						for (j = jmin; j <= ns; ++j) {
							if (basis[j - 1] != 0) {
								r[j - 1] = zero;
							} else {
								jj = j - n0;
								r[j - 1] -= cons * bi[ip - 1][jj - 1];
							}
						}
					}
					if (ns != n) {
						jmin = ns + 1;
						for (j = jmin; j <= n; ++j) {
							if (basis[j - 1] != 0) {
								r[j - 1] = zero;
							} else {
								jj = j - n0;
								r[j - 1] += cons * bi[ip - 1][jj - 1];
							}
						}
					}
					gotoflag = 200;
					continue;
				} else {
					gotoflag = 601;
				}
			}

			if (gotoflag == 601) {
				for (j = 1; j <= m; ++j) {
					if (xb[j - 1] < zero) {
						gotoflag = 610;
						break;
					}
				}
				if (gotoflag != 610) {
					gotoflag = 630;
				}
			}

			if (gotoflag == 610) {

				nstep = 1;
				m0 = 0;
				for (l = 1; l <= m; ++l) {
					if (xb[l - 1] >= zero) {
						continue;
					}
					++m0;
					indx[m0 - 1] = l;
				}
				for (j = 1; j <= m; ++j) {
					dsump = dsumn = zero;
					for (ll = 1; ll <= m0; ++ll) {
						l = indx[ll - 1];
						if (bi[l - 1][j - 1] < zero) {
							dsumn += bi[l - 1][j - 1];
						} else if (bi[l - 1][j - 1] > zero) {
							dsump += bi[l - 1][j - 1];
						}
					}
					y[j - 1] = dsump + dsumn;
					s = dsump;
					t = dsumn;
					tol = rerrmx * Math.max(s, -t);
					if (Math.abs(y[j - 1]) < tol) {
						y[j - 1] = zero;
					}
				}
				for (j = 1; j <= n0; ++j) {
					if (basis[j - 1] == 0) {
						r[j - 1] = zero;
						for (kk = 1; kk <= m; ++kk) {
							r[j - 1] += y[kk - 1] * a[kk - 1][j - 1];
						}
					} else {
						r[j - 1] = zero;
					}
				}
				if (n0 != ns) {
					jmin = n0 + 1;
					for (j = jmin; j <= ns; ++j) {
						r[j - 1] = zero;
						if (basis[j - 1] != 0) {
							continue;
						}
						jj = j - n0;
						r[j - 1] = y[jj - 1];
					}
				}
				if (ns != n) {
					jmin = ns + 1;
					for (j = jmin; j <= n; ++j) {
						r[j - 1] = zero;
						if (basis[j - 1] != 0) {
							continue;
						}
						jj = j - n0;
						r[j - 1] = -y[jj - 1];
					}
				}
				gotoflag = 200;
				continue;
			}

			if (gotoflag == 630) {

				// SET UP THE R ARRAY FOR THE NSTEP = 2 CASE
				if (n != num) {
					nstep = 2;
					m0 = 0;
					for (l = 1; l <= m; ++l) {
						if (ibasis[l - 1] <= n) {
							continue;
						}
						++m0;
						indx[m0 - 1] = l;
					}
					for (j = 1; j <= m; ++j) {
						dsump = dsumn = zero;
						for (ll = 1; ll <= m0; ++ll) {
							l = indx[ll - 1];
							if (bi[l - 1][j - 1] < zero) {
								dsumn += bi[l - 1][j - 1];
							} else if (bi[l - 1][j - 1] > zero) {
								dsump += bi[l - 1][j - 1];
							}
						}
						y[j - 1] = -(dsump + dsumn);
						s = dsump;
						t = dsumn;
						tol = rerrmx * Math.max(s, -t);
						if (Math.abs(y[j - 1]) < tol) {
							y[j - 1] = zero;
						}
					}
					for (j = 1; j <= n0; ++j) {
						if (basis[j - 1] == 0) {
							r[j - 1] = zero;
							for (kk = 1; kk <= m; ++kk) {
								r[j - 1] += y[kk - 1] * a[kk - 1][j - 1];
							}
						} else {
							r[j - 1] = zero;
						}
					}
					if (n0 != ns) {
						jmin = n0 + 1;
						for (j = jmin; j <= ns; ++j) {
							r[j - 1] = zero;
							if (basis[j - 1] != 0) {
								continue;
							}
							jj = j - n0;
							r[j - 1] = y[jj - 1];
						}
					}
					if (ns != n) {
						jmin = ns + 1;
						for (j = jmin; j <= n; ++j) {
							r[j - 1] = zero;
							if (basis[j - 1] != 0) {
								continue;
							}
							jj = j - n0;
							r[j - 1] = -y[jj - 1];
						}
					}
					gotoflag = 200;
					continue;
				} else {
					gotoflag = 680;
				}
			}

			if (gotoflag == 680) {

				// SET UP A NEW R ARRAY FOR THE NSTEP = 3 CASE
				nstep = 3;
				for (j = 1; j <= m; ++j) {
					dsum = zero;
					for (l = 1; l <= m; ++l) {
						il = ibasis[l - 1];
						if (il <= n0) {
							dsum += c[il - 1] * bi[l - 1][j - 1];
						}
					}
					y[j - 1] = dsum;
				}
				for (j = 1; j <= n0; ++j) {
					r[j - 1] = zero;
					if (basis[j - 1] != 0) {
						continue;
					}
					dsum = -c[j - 1];
					for (kk = 1; kk <= m; ++kk) {
						dsum += y[kk - 1] * a[kk - 1][j - 1];
					}
					r[j - 1] = dsum;
					if (r[j - 1] >= zero) {
						continue;
					}
					tol = rerrmx * Math.abs(c[j - 1]);
					if (Math.abs(r[j - 1]) < tol) {
						r[j - 1] = zero;
					}
				}
				if (n0 != ns) {
					jmin = n0 + 1;
					for (j = jmin; j <= ns; ++j) {
						r[j - 1] = zero;
						if (basis[j - 1] != 0) {
							continue;
						}
						jj = j - n0;
						r[j - 1] = y[jj - 1];
					}
				}
				if (ns != n) {
					jmin = ns + 1;
					for (j = jmin; j <= n; ++j) {
						r[j - 1] = zero;
						if (basis[j - 1] != 0) {
							continue;
						}
						jj = j - n0;
						r[j - 1] = -y[jj - 1];
					}
				}
				gotoflag = 200;
				continue;
			}

			// -----------------------------------------------------------------------
			// REFINE XB AND STORE THE RESULT IN Y
			// -----------------------------------------------------------------------
			if (gotoflag == 800) {

				Arrays.fill(y, 0, m, zero);
				m0 = 0;
				for (j = 1; j <= m; ++j) {
					kj = ibasis[j - 1];
					if (kj <= n0) {
						++m0;
						indx[m0 - 1] = j;
					} else if (kj <= ns) {
						l = kj - n0;
						y[l - 1] = xb[j - 1];
					} else if (kj <= n) {
						l = kj - n0;
						y[l - 1] = -xb[j - 1];
					} else {
						l = kj - n0;
						y[l - 1] = xb[j - 1];
					}
				}
				if (m0 == 0) {
					for (kk = 1; kk <= m; ++kk) {
						r[kk - 1] = b0[kk - 1] - y[kk - 1];
					}
				} else {
					for (i = 1; i <= m; ++i) {
						dsum = y[i - 1];
						for (jj = 1; jj <= m0; ++jj) {
							j = indx[jj - 1];
							kj = ibasis[j - 1];
							dsum += a[i - 1][kj - 1] * xb[j - 1];
						}
						r[i - 1] = b0[i - 1] - dsum;
					}
				}
				rerr1 = Math.min(rerrmx, rerr[0]);
				for (i = 1; i <= m; ++i) {
					y[i - 1] = zero;
					if (xb[i - 1] < zero) {
						sgn = -1.0;
						dsump = zero;
						dsumn = xb[i - 1];
					} else if (xb[i - 1] > zero) {
						sgn = 1.0;
						dsump = xb[i - 1];
						dsumn = zero;
					} else {
						continue;
					}
					for (l = 1; l <= m; ++l) {
						dt = bi[i - 1][l - 1] * r[l - 1];
						if (dt > zero) {
							dsump += dt;
						} else {
							dsumn += dt;
						}
					}
					w = dsump + dsumn;
					if (w == zero) {
						continue;
					}
					if (sgn != RealMath.sign(1.0, w)) {
						continue;
					}
					s = dsump;
					t = dsumn;
					tol = rerr1 * Math.max(s, -t);
					if (Math.abs(w) > tol) {
						y[i - 1] = w;
					}
				}
				if (nstep == 2) {

					// CHECK THE REFINEMENT (NSTEP = 2)
					for (i = 1; i <= m; ++i) {
						if (ibasis[i - 1] <= n) {
							xb[i - 1] = y[i - 1];
						} else if (y[i - 1] > rerrmx) {
							if (icount >= 5) {
								gotoflag = 100;
								break;
							} else {
								ind[0] = 1;
								gotoflag = 220;
								break;
							}
						} else {
							y[i - 1] = zero;
							xb[i - 1] = y[i - 1];
						}
					}
					if (gotoflag != 100 && gotoflag != 220) {
						gotoflag = 680;
					}
				} else if (nstep > 2) {

					// COMPUTE Z (NSTEP = 3)
					dsum = zero;
					for (i = 1; i <= m; ++i) {
						ki = ibasis[i - 1];
						if (ki > n0) {
							xb[i - 1] = y[i - 1];
						} else {
							dsum += c[ki - 1] * y[i - 1];
							xb[i - 1] = y[i - 1];
						}
					}
					z[0] = dsum;
					gotoflag = 220;
				} else {

					// CHECK THE REFINEMENT (NSTEP = 1)
					for (i = 1; i <= m; ++i) {
						if (y[i - 1] >= zero) {
							xb[i - 1] = y[i - 1];
						} else if (y[i - 1] < -rerrmx) {
							if (icount >= 5) {
								gotoflag = 100;
								break;
							} else {
								ind[0] = 1;
								gotoflag = 220;
								break;
							}
						} else {
							y[i - 1] = zero;
							xb[i - 1] = y[i - 1];
						}
					}
					if (gotoflag != 100 && gotoflag != 220) {
						gotoflag = 630;
					}
				}
			}
		}
	}

	private static void crout1(final double[] a, final int ka, final int n, final int iend, final int[] indx,
			final double[] temp, final int[] ierr) {
		double zero = 0.0;
		int i, ibeg, ij, ik, il, j, jcol, jj, k, kcol, kcount, kj, kj0, kk, kl, km1, kp1, l, lj, lj0, lk, lmin, maxdim,
				mcol, ncol, nk, nm1, nmj, nmk, nn;
		double dsum, c, pmin, s;

		maxdim = ka * n;
		mcol = iend * ka;
		if (iend != 0) {

			// PROCESS THE FIRST IEND COLUMNS OF A
			kcol = 0;
			for (k = 1; k <= iend; ++k) {
				kk = kcol + k;
				nk = kcol + n;
				int flag = 300;
				for (lk = kk; lk <= nk; ++lk) {
					if (a[lk - 1] < zero) {
						flag = 20;
						break;
					} else if (a[lk - 1] > zero) {
						flag = 30;
						break;
					}
				}
				if (flag == 300) {
					ierr[0] = 1;
					return;
				}
				if (flag == 20) {
					l = lk - kcol;
					lj0 = mcol + l;
					for (lj = lj0; lj <= maxdim; lj += ka) {
						a[lj - 1] = -a[lj - 1];
					}
					flag = 30;
				}
				if (flag == 30) {
					l = lk - kcol;
					indx[k - 1] = l;
					if (k != l) {
						lj = lk;
						for (kj = kk; kj <= maxdim; kj += ka) {
							c = a[kj - 1];
							a[kj - 1] = a[lj - 1];
							a[lj - 1] = c;
							lj += ka;
						}
					}
					kcol += ka;
				}
			}
		}

		// PROCESS THE REMAINING COLUMNS OF A
		nm1 = n - 1;
		ierr[0] = 0;
		pmin = zero;
		ibeg = iend + 1;
		if (ibeg != n) {
			k = ibeg;
			km1 = iend;
			kp1 = k + 1;
			kcol = mcol;
			kk = kcol + k;
			for (kcount = ibeg; kcount <= nm1; ++kcount) {

				// SEARCH FOR THE K-TH PIVOT ELEMENT (K=IBEG, ..., N-1)
				l = k;
				s = Math.abs(a[kk - 1]);
				for (i = kp1; i <= n; ++i) {
					ik = kcol + i;
					c = Math.abs(a[ik - 1]);
					if (s >= c) {
						continue;
					}
					l = i;
					s = c;
				}
				if (k <= ibeg || s < pmin) {
					pmin = s;
					if (s == zero) {
						ierr[0] = 1;
						return;
					}
				}

				// INTERCHANGING ROWS K AND L
				indx[k - 1] = l;
				if (k != l) {
					kj0 = mcol + k;
					lj = mcol + l;
					for (kj = kj0; kj <= maxdim; kj += ka) {
						c = a[kj - 1];
						a[kj - 1] = a[lj - 1];
						a[lj - 1] = c;
						lj += ka;
					}
				}

				// COMPUTE THE K-TH ROW OF U (K=IBEG, ..., N-1)
				c = a[kk - 1];
				if (k > ibeg) {
					kl = mcol + k;
					for (l = ibeg; l <= km1; ++l) {
						temp[l - 1] = a[kl - 1];
						kl += ka;
					}
					kj0 = kk + ka;
					for (kj = kj0; kj <= maxdim; kj += ka) {
						jcol = kj - k;
						dsum = -a[kj - 1];
						for (l = ibeg; l <= km1; ++l) {
							lj = jcol + l;
							dsum += temp[l - 1] * a[lj - 1];
						}
						a[kj - 1] = -dsum / c;
					}
				} else {
					kj0 = kk + ka;
					for (kj = kj0; kj <= maxdim; kj += ka) {
						a[kj - 1] /= c;
					}
				}

				// COMPUTE THE K-TH COLUMN OF L (K=IBEG+1, ..., N)
				km1 = k;
				k = kp1;
				kp1 = k + 1;
				kcol += ka;
				kk = kcol + k;
				for (l = ibeg; l <= km1; ++l) {
					lk = kcol + l;
					temp[l - 1] = a[lk - 1];
				}
				for (i = k; i <= n; ++i) {
					il = mcol + i;
					dsum = zero;
					for (l = ibeg; l <= km1; ++l) {
						dsum += a[il - 1] * temp[l - 1];
						il += ka;
					}
					a[il - 1] -= dsum;
				}
			}
		}

		// CHECK THE N-TH PIVOT ELEMENT
		ncol = maxdim - ka;
		nn = ncol + n;
		c = Math.abs(a[nn - 1]);
		if (c > pmin) {
		} else if (c == zero) {
			ierr[0] = 1;
			return;
		}

		// REPLACE L WITH THE INVERSE OF L
		if (ibeg != n) {
			jj = mcol + ibeg;
			i = ka + 1;
			for (j = ibeg; j <= nm1; ++j) {
				a[jj - 1] = 1.0 / a[jj - 1];
				temp[j - 1] = a[jj - 1];
				kj = jj;
				for (km1 = j; km1 <= nm1; ++km1) {
					k = km1 + 1;
					++kj;
					dsum = zero;
					kl = kj;
					for (l = j; l <= km1; ++l) {
						dsum += a[kl - 1] * temp[l - 1];
						kl += ka;
					}
					a[kj - 1] = -dsum / a[kl - 1];
					temp[k - 1] = a[kj - 1];
				}
				jj += i;
			}
		}
		a[nn - 1] = 1.0 / a[nn - 1];
		if (n == 1) {
			return;
		}

		// SOLVE UX = Y WHERE Y IS THE INVERSE OF L
		for (nmk = 1; nmk <= nm1; ++nmk) {
			k = n - nmk;
			lmin = Math.max(ibeg, k + 1);
			kl = (lmin - 1) * ka + k;
			for (l = lmin; l <= n; ++l) {
				temp[l - 1] = a[kl - 1];
				a[kl - 1] = zero;
				kl += ka;
			}
			kj0 = mcol + k;
			for (kj = kj0; kj <= maxdim; kj += ka) {
				dsum = -a[kj - 1];
				lj = (kj - k) + lmin;
				for (l = lmin; l <= n; ++l) {
					dsum += temp[l - 1] * a[lj - 1];
					++lj;
				}
				a[kj - 1] = -dsum;
			}
		}

		// COLUMN INTERCHANGES
		jcol = ncol - ka;
		for (nmj = 1; nmj <= nm1; ++nmj) {
			j = n - nmj;
			k = indx[j - 1];
			if (j != k) {
				ij = jcol;
				ik = (k - 1) * ka;
				for (i = 1; i <= n; ++i) {
					++ij;
					++ik;
					c = a[ij - 1];
					a[ij - 1] = a[ik - 1];
					a[ik - 1] = c;
				}
			}
			jcol -= ka;
		}
	}

	private static double dpmpar(final int i) {
		switch (i) {
		case 1:
			return Constants.EPSILON;
		case 2:
			return Double.MIN_VALUE;
		case 3:
			return Double.MAX_VALUE;
		default:
			return Double.NaN;
		}
	}
}
