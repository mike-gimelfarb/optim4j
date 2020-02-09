/*
---------------------------------------------------------------------------
Copyright:

    Subroutines PMIN, PBUN, PNEW, PVAR, Copyright ACM, 2001, were published 
    as Algorithm 811 in ACM Transactions on Mathematical Software, Vol.27,  
    No.2, 2001. Subroutines PLIS, PLIP, PNET, PNED, PNEC, PSED, PSEC, PSEN,  
    PGAD, PGAC, PMAX, PSUM, PEQN, PEQL, Copyright ACM, 2009, were published 
    as Algorithm 896 in ACM Transactions on Mathematical Software, Vol.36, 
    No.3, 2009. Here are the author's modifications of the above ACM 
    algorithms. They are posted here by permission of ACM for your personal 
    use, not for redistribution. The remaining subroutines, Copyright 
    Ladislav Luksan, 2007, were supported by the Czech Academy of Sciences. 
    Many of sparse matrix modules were prepared by Miroslav Tuma.

License: 

    This library (with exception of the ACM algorithms) is a free software;
    you can redistribute it and/or modify it under the terms of the GNU 
    Lesser General Public License as published by the Free Software 
    Foundation; either version 2.1 of the License, or (at your option) 
    any later version (see http://www.gnu.org/copyleft/gpl.html).

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    Permission is hereby granted to use or copy this program under the
    terms of the GNU LGPL, provided that the Copyright, this License,
    and the Availability of the original version is retained on all copies.
    User documentation of any code that uses this code or any modified
    version of this code must cite the Copyright, this License, the
    Availability note, and "Used by permission." Permission to modify
    the code and to distribute modified code is granted, provided the
    Copyright, this License, and the Availability note are retained,
    and a notice that the code was modified is included.

Availability:

    http://www.cs.cas.cz/~luksan/subroutines.html

Acknowledgements:

    This work was supported by the Grant Agency of the Czech Academy of 
    Sciences, under grant IAA1030405.
*/
package opt.multivariate.constrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.unconstrained.order1.GradientOptimizer;
import utils.BlasMath;
import utils.IntMath;
import utils.RealMath;

/**
 * A translation of a SQP variable metric method for minimization of a general
 * non-linear differentiable function subject to general constraints.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Original code found at: http://www.cs.cas.cz/~luksan/subroutines.html
 */
public final class SqpAlgorithm extends GradientOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	/**
	 *
	 */
	public enum UpdateMethod {

		BFGS, HOSHINO;
	}

	@FunctionalInterface
	private static interface Con {

		void apply(int nf, int kc, double[] x, double[] f);
	}

	@FunctionalInterface
	private static interface Dcon {

		void apply(int nf, int kc, double[] x, double[] gc);
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// COMMON /STAT
	private int nres, ndec, nrem, nit, nfv, nfg, nfh;
	private final int[] nadd = new int[1];

	// SAVED VARIABLES
	private int mtyp, mode, mes1, mes2;
	private double rl, fl, ru, fu, ri, fi;

	// PARAMETERS
	private final UpdateMethod method;
	private final boolean correc;
	private final int maxevals;
	private final double penalty, maxstep, tolc, tolg;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolX
	 * @param tolConstr
	 * @param tolGrad
	 * @param penaltyCoeff
	 * @param maxStepSize
	 * @param maxEvaluations
	 * @param updateMethod
	 * @param doCorrection
	 */
	public SqpAlgorithm(final double tolX, final double tolConstr, final double tolGrad, final double penaltyCoeff,
			final double maxStepSize, final int maxEvaluations, final UpdateMethod updateMethod,
			final boolean doCorrection) {
		super(tolX);
		tolc = tolConstr;
		tolg = tolGrad;
		penalty = penaltyCoeff;
		maxstep = maxStepSize;
		maxevals = maxEvaluations;
		method = updateMethod;
		correc = doCorrection;
	}

	/**
	 *
	 * @param tolX
	 * @param tolConstr
	 * @param tolGrad
	 * @param penaltyCoeff
	 * @param maxStepSize
	 * @param maxEvaluations
	 */
	public SqpAlgorithm(final double tolX, final double tolConstr, final double tolGrad, final double penaltyCoeff,
			final double maxStepSize, final int maxEvaluations) {
		this(tolX, tolConstr, tolGrad, penaltyCoeff, maxStepSize, maxEvaluations, UpdateMethod.BFGS, true);
	}

	/**
	 *
	 * @param tolerance
	 * @param penaltyCoeff
	 * @param maxStepSize
	 * @param maxEvaluations
	 */
	public SqpAlgorithm(final double tolerance, final double penaltyCoeff, final double maxStepSize,
			final int maxEvaluations) {
		this(tolerance, tolerance, tolerance, penaltyCoeff, maxStepSize, maxEvaluations);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess) {
		return optimize(f, df, guess, null, null);
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param obj
	 * @param dobj
	 * @param guess
	 * @param ix
	 * @param lower
	 * @param upper
	 * @return
	 */
	public final double[] optimize(final Function<? super double[], Double> obj,
			final Function<? super double[], double[]> dobj, final double[] guess, final double[] lower,
			final double[] upper) {
		final double[] cf = new double[1];
		final double[] cl = new double[1];
		final double[] cu = new double[1];
		final int[] ic = new int[0];
		final int[] ix = new int[guess.length];
		if (lower != null) {
			Arrays.fill(ix, 3);
		}
		final double[] result = psqpn1(obj, dobj, null, null, 1, 0, guess, ix, lower, upper, cf, ic, cl, cu);
		myEvals = nfv;
		myGEvals = nfg;
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private double[] psqpn1(final Function<? super double[], Double> f, final Function<? super double[], double[]> df,
			final Con con, final Dcon dcon, final int nb, final int nc, final double[] guess, final int[] ix,
			final double[] xl, final double[] xu, final double[] cf, final int[] ic, final double[] cl,
			final double[] cu) {

		// prepare variables
		final int nf = guess.length;
		final double[] x = Arrays.copyOf(guess, nf);
		final int[] ipar = new int[] { Integer.MAX_VALUE, maxevals, 0, 0, method == UpdateMethod.BFGS ? 1 : 2,
				correc ? 2 : 1 };
		final double[] rpar = new double[] { maxstep, myTol, tolc, tolg, penalty };
		final double[] farr = new double[1];
		final double[] gmax = new double[1];
		final double[] cmax = new double[1];
		final int[] iterm = new int[1];

		// call main subroutine
		psqpn(f, df, con, dcon, nf, nb, nc, x, ix, xl, xu, cf, ic, cl, cu, ipar, rpar, farr, gmax, cmax, 0, iterm);

		if (iterm[0] == 1 || iterm[0] == 2 || iterm[0] == 3 || iterm[0] == 4 || iterm[0] == -6) {
			return x;
		} else {
			return null;
		}
	}

	private void psqpn(final Function<? super double[], Double> obj, final Function<? super double[], double[]> dobj,
			final Con con, final Dcon dcon, final int nf, final int nb, final int nc, final double[] x, final int[] ix,
			final double[] xl, final double[] xu, final double[] cf, final int[] ic, final double[] cl,
			final double[] cu, final int[] ipar, final double[] rpar, final double[] f, final double[] gmax,
			final double[] cmax, final int iprnt, final int[] iterm) {

		final double[] cg = new double[nf * nc];
		final double[] cfo = new double[nc + 1];
		final double[] cfd = new double[nc];
		final double[] gc = new double[nf];
		final double[] cr = new double[nf * (nf + 1) / 2];
		final double[] cz = new double[nf];
		final double[] cp = new double[nc];
		final double[] gf = new double[nf];
		final double[] g = new double[nf];
		final double[] h = new double[nf * (nf + 1) / 2];
		final double[] s = new double[nf];
		final double[] xo = new double[nf];
		final double[] go = new double[nf];
		final int[] ia = new int[nf];
		final double[] rpar1 = { rpar[1 - 1] };
		final double[] rpar2 = { rpar[2 - 1] };
		final double[] rpar3 = { rpar[3 - 1] };
		final double[] rpar4 = { rpar[4 - 1] };
		final double[] rpar5 = { rpar[5 - 1] };
		final int[] ipar1 = { ipar[1 - 1] };
		final int[] ipar2 = { ipar[2 - 1] };
		final int[] ipar5 = { ipar[5 - 1] };
		final int[] ipar6 = { ipar[6 - 1] };
		psqp(obj, dobj, con, dcon, nf, nb, nc, x, ix, xl, xu, cf, ic, cl, cu, cg, cfo, cfd, gc, ia, cr, cz, cp, gf, g,
				h, s, xo, go, rpar1, rpar2, rpar3, rpar4, rpar5, cmax, gmax, f, ipar1, ipar2, ipar5, ipar6, iprnt,
				iterm);
		rpar[1 - 1] = rpar1[0];
		rpar[2 - 1] = rpar2[0];
		rpar[3 - 1] = rpar3[0];
		rpar[4 - 1] = rpar4[0];
		rpar[5 - 1] = rpar5[0];
		ipar[1 - 1] = ipar1[0];
		ipar[2 - 1] = ipar2[0];
		ipar[5 - 1] = ipar5[0];
		ipar[6 - 1] = ipar6[0];
	}

	private void psqp(final Function<? super double[], Double> obj, final Function<? super double[], double[]> dobj,
			final Con con, final Dcon dcon, final int nf, final int nb, final int nc, final double[] x, final int[] ix,
			final double[] xl, final double[] xu, final double[] cf, final int[] ic, final double[] cl,
			final double[] cu, final double[] cg, final double[] cfo, final double[] cfd, final double[] gc,
			final int[] ica, final double[] cr, final double[] cz, final double[] cp, final double[] gf,
			final double[] g, final double[] h, final double[] s, final double[] xo, final double[] go,
			final double[] xmax, final double[] tolx, final double[] tolc, final double[] tolg, final double[] rpf,
			final double[] cmax, final double[] gmax, final double[] f, final int[] mit, final int[] mfv,
			final int[] met, final int[] mec, final int iprnt, final int[] iterm) {

		final double[] ff = new double[1], fc = new double[1], umax = new double[1], r = new double[1],
				rp = new double[1], fp = new double[1], pp = new double[1], p = new double[1], po = new double[1],
				dmax = new double[1];
		final int[] iterl = new int[1], ld = new int[1], idecf = new int[1], iterq = new int[1], n = new int[1],
				kd = new int[1], nred = new int[1], maxst = new int[1], iters = new int[1], isys = new int[1],
				iterh = new int[1];
		double told, tols, alf1, alf2, cmaxo, eps7, eps9, eta0, eta2, eta9, fmax, fmin, fo, gnorm = 0.0, rmax, rmin, ro,
				snorm = 0.0, tolb, tolf;
		int met1, mes, i, iext, irest, iterd, kbf, kbc, kc, kit, mred, mtesf, mtesx, k, ntesx = 0, iest, inits, kters,
				mfp, ipom, lds;

		// INITIATION
		kbf = kbc = 0;
		if (nb > 0) {
			kbf = 2;
		}
		if (nc > 0) {
			kbc = 2;
		}
		nres = ndec = nrem = nadd[0] = nit = nfv = nfg = nfh = 0;
		isys[0] = iest = iext = 0;
		mtesx = mtesf = 2;
		inits = 1;
		iterm[0] = iters[0] = iterd = iterq[0] = 0;
		mred = 20;
		irest = 1;
		iters[0] = 2;
		kters = 5;
		idecf[0] = 1;
		eta0 = eta2 = 1e-15;
		eta9 = 1e60;
		eps7 = 1e-15;
		eps9 = 1e-8;
		alf1 = 1e-10;
		alf2 = 1e10;
		fmax = 1e60;
		fmin = -fmax;
		tolb = -fmax;
		dmax[0] = eta9;
		tolf = 1e-16;
		if (xmax[0] <= 0.0) {
			xmax[0] = 1e16;
		}
		if (tolx[0] <= 0.0) {
			tolx[0] = 1e-16;
		}
		if (tolg[0] <= 0.0) {
			tolg[0] = 1e-6;
		}
		if (tolc[0] <= 0.0) {
			tolc[0] = 1e-6;
		}
		told = 1e-8;
		tols = 1e-4;
		if (rpf[0] <= 0.0) {
			rpf[0] = 1e-4;
		}
		if (met[0] <= 0) {
			met[0] = 1;
		}
		met1 = 2;
		if (mec[0] <= 0) {
			mec[0] = 2;
		}
		mes = 1;
		if (mit[0] <= 0) {
			mit[0] = 1000;
		}
		if (mfv[0] <= 0) {
			mfv[0] = 2000;
		}
		kd[0] = 1;
		ld[0] = -1;
		kit = 0;
		mxvset(nc, 0.0, cp, 1);

		// INITIAL OPERATIONS WITH SIMPLE BOUNDS
		if (kbf > 0) {
			for (i = 1; i <= nf; ++i) {
				if ((ix[i - 1] == 3 || ix[i - 1] == 4) && xu[i - 1] <= xl[i - 1]) {
					xu[i - 1] = xl[i - 1];
					ix[i - 1] = 5;
				} else if (ix[i - 1] == 5 || ix[i - 1] == 6) {
					xl[i - 1] = x[i - 1];
					xu[i - 1] = x[i - 1];
					ix[i - 1] = 5;
				}
				if (ix[i - 1] == 1 || ix[i - 1] == 3) {
					x[i - 1] = Math.max(x[i - 1], xl[i - 1]);
				}
				if (ix[i - 1] == 2 || ix[i - 1] == 3) {
					x[i - 1] = Math.min(x[i - 1], xu[i - 1]);
				}
			}
		}

		// INITIAL OPERATIONS WITH GENERAL CONSTRAINTS
		if (kbc > 0) {
			k = 0;
			for (kc = 1; kc <= nc; ++kc) {
				if ((ic[kc - 1] == 3 || ic[kc - 1] == 4) && cu[kc - 1] <= cl[kc - 1]) {
					cu[kc - 1] = cl[kc - 1];
					ic[kc - 1] = 5;
				} else if (ic[kc - 1] == 5 || ic[kc - 1] == 6) {
					cu[kc - 1] = cl[kc - 1];
					ic[kc - 1] = 5;
				}
				k += nf;
			}
		}
		if (kbf > 0) {
			for (i = 1; i <= nf; ++i) {
				if (ix[i - 1] >= 5) {
					ix[i - 1] = -ix[i - 1];
				}
				if (ix[i - 1] <= 0) {
				} else if ((ix[i - 1] == 1 || ix[i - 1] == 3) && x[i - 1] <= xl[i - 1]) {
					x[i - 1] = xl[i - 1];
				} else if ((ix[i - 1] == 2 || ix[i - 1] == 3) && x[i - 1] >= xu[i - 1]) {
					x[i - 1] = xu[i - 1];
				}
				plnews(x, ix, xl, xu, eps9, i, iterl);
				if (ix[i - 1] > 10) {
					ix[i - 1] = 10 - ix[i - 1];
				}
			}
		}
		fo = fmin;
		gmax[0] = dmax[0] = eta9;

		while (true) {

			lds = ld[0];
			pf1f01(obj, dobj, nf, x, gf, gf, ff, f, kd[0], ld, iext);
			ld[0] = lds;
			pc1f01(con, dcon, nf, nc, x, fc, cf, cl, cu, ic, gc, cg, cmax, kd[0], ld);
			cf[nc + 1 - 1] = f[0];

			// START OF THE ITERATION WITH TESTS FOR TERMINATION.
			if (iterm[0] < 0) {
				return;
			}
			if (iters[0] != 0) {
				if (f[0] <= tolb) {
					iterm[0] = 3;
					return;
				}
				if (dmax[0] <= tolx[0]) {
					iterm[0] = 1;
					++ntesx;
					if (ntesx >= mtesx) {
						return;
					}
				} else {
					ntesx = 0;
				}
			}
			if (nit >= mit[0]) {
				iterm[0] = 11;
				return;
			}
			if (nfv >= mfv[0]) {
				iterm[0] = 12;
				return;
			}
			iterm[0] = 0;
			++nit;

			while (true) {

				// RESTART
				n[0] = nf;
				if (irest > 0) {
					mxdsmi(n[0], h);
					ld[0] = Math.min(ld[0], 1);
					idecf[0] = 1;
					if (kit < nit) {
						++nres;
						kit = nit;
					} else {
						iterm[0] = -10;
						if (iters[0] < 0) {
							iterm[0] = iters[0] - 5;
						}
						return;
					}
				}

				// DIRECTION DETERMINATION USING A QUADRATIC PROGRAMMING PROCEDURE
				System.arraycopy(cf, 0, cfo, 0, nc + 1);
				mfp = 2;
				ipom = 0;

				while (true) {
					plqdb1(nf, nc, x, ix, xl, xu, cf, cfd, ic, ica, cl, cu, cg, cr, cz, g, gf, h, s, mfp, kbf, kbc,
							idecf, eta2, eta9, eps7, eps9, umax, gmax, n, iterq);
					if (iterq[0] < 0) {
						if (ipom < 10) {
							++ipom;
							plredl(nc, cf, ic, cl, cu, kbc);
						} else {
							iterd = iterq[0] - 10;
							break;
						}
					} else {
						// ipom = 0;
						iterd = 1;
						gmax[0] = mxvmax(nf, g);
						gnorm = Math.sqrt(mxvdot(nf, g, 1, g, 1));
						snorm = Math.sqrt(mxvdot(nf, s, 1, s, 1));
						break;
					}
				}

				if (iterd < 0) {
					iterm[0] = iterd;
				}
				if (iterm[0] != 0) {
					return;
				}
				System.arraycopy(cfo, 0, cf, 0, nc + 1);

				// TEST FOR SUFFICIENT DESCENT
				p[0] = mxvdot(nf, g, 1, s, 1);
				irest = 1;
				if (snorm <= 0.0) {
				} else if (p[0] + told * gnorm * snorm <= 0.0) {
					irest = 0;
				}
				if (irest == 0) {
					nred[0] = 0;
					rmin = alf1 * gnorm / snorm;
					rmax = Math.min(alf2 * gnorm / snorm, xmax[0] / snorm);
				} else {
					continue;
				}
				if (gmax[0] <= tolg[0] && cmax[0] <= tolc[0]) {
					iterm[0] = 4;
					return;
				}
				ppset2(nf, n[0], nc, ica, cz, cp);
				mxvina(nc, ic);
				pp0af8(nf, n[0], nc, cf, ic, ica, cl, cu, cz, rpf[0], fc, f);

				// PREPARATION OF LINE SEARCH
				ro = 0.0;
				fo = f[0];
				po[0] = p[0];
				cmaxo = cmax[0];
				System.arraycopy(x, 0, xo, 0, nf);
				System.arraycopy(g, 0, go, 0, nf);
				System.arraycopy(gf, 0, cr, 0, nf);
				System.arraycopy(cf, 0, cfo, 0, nc + 1);

				// LINE SEARCH WITHOUT DIRECTIONAL DERIVATIVES
				while (true) {
					ps0l02(r, ro, rp, f[0], fo, fp, po[0], pp, fmin, fmax, rmin, rmax, tols, kd, ld, nit, kit, nred,
							mred, maxst, iest, inits, iters, kters, mes, isys);
					if (isys[0] == 0) {
						break;
					} else {
						mxvdir(nf, r[0], s, 1, xo, x, 1);
						lds = ld[0];
						pf1f01(obj, dobj, nf, x, gf, g, ff, f, kd[0], ld, iext);
						ld[0] = lds;
						pc1f01(con, dcon, nf, nc, x, fc, cf, cl, cu, ic, gc, cg, cmax, kd[0], ld);
						cf[nc + 1 - 1] = f[0];
						pp0af8(nf, n[0], nc, cf, ic, ica, cl, cu, cz, rpf[0], fc, f);
					}
				}
				kd[0] = 1;

				// DECISION AFTER UNSUCCESSFUL LINE SEARCH
				if (iters[0] <= 0) {
					r[0] = 0.0;
					f[0] = fo;
					p[0] = po[0];
					System.arraycopy(xo, 0, x, 0, nf);
					System.arraycopy(cr, 0, gf, 0, nf);
					System.arraycopy(cfo, 0, cf, 0, nc + 1);
					irest = 1;
					ld[0] = kd[0];
				} else {
					break;
				}
			}

			// COMPUTATION OF THE VALUE AND THE GRADIENT OF THE OBJECTIVE
			// FUNCTION TOGETHER WITH THE VALUES AND THE GRADIENTS OF THE
			// APPROXIMATED FUNCTIONS
			if (kd[0] > ld[0]) {
				lds = ld[0];
				pf1f01(obj, dobj, nf, x, gf, gf, ff, f, kd[0], ld, iext);
				ld[0] = lds;
				pc1f01(con, dcon, nf, nc, x, fc, cf, cl, cu, ic, gc, cg, cmax, kd[0], ld);
			}

			// PREPARATION OF VARIABLE METRIC UPDATE
			System.arraycopy(gf, 0, g, 0, nf);
			pytrnd(nf, n, x, xo, ica, cg, cz, g, go, r[0], f, fo, p, po, cmax, cmaxo, dmax, kd[0], ld, iters[0]);

			// VARIABLE METRIC UPDATE
			pudbg1(n[0], h, g, s, xo, go, r[0], po[0], nit, kit, iterh, met[0], met1, mec[0]);

			// END OF THE ITERATION
		}
	}

	private void pc1f01(final Con con, final Dcon dcon, final int nf, final int nc, final double[] x, final double[] fc,
			final double[] cf, final double[] cl, final double[] cu, final int[] ic, final double[] gc,
			final double[] cg, final double[] cmax, final int kd, final int[] ld) {
		int kc;
		double pom, temp;

		if (kd <= ld[0]) {
			return;
		}
		if (ld[0] < 0) {
			cmax[0] = 0.0;
		}
		for (kc = 1; kc <= nc; ++kc) {
			if (kd < 0) {
				continue;
			}
			if (ld[0] >= 0) {
				fc[0] = cf[kc - 1];
			} else {
				con.apply(nf, kc, x, fc);
				cf[kc - 1] = fc[0];
				if (ic[kc - 1] > 0) {
					pom = 0.0;
					temp = cf[kc - 1];
					if (ic[kc - 1] == 1 || ic[kc - 1] >= 3) {
						pom = Math.min(pom, temp - cl[kc - 1]);
					}
					if (ic[kc - 1] == 2 || ic[kc - 1] >= 3) {
						pom = Math.min(pom, cu[kc - 1] - temp);
					}
					if (pom < 0.0) {
						cmax[0] = Math.max(cmax[0], -pom);
					}
				}
			}

			if (kd < 1) {
				continue;
			}
			if (ld[0] >= 1) {
				System.arraycopy(cg, (kc - 1) * nf + 1 - 1, gc, 0, nf);
			} else {
				dcon.apply(nf, kc, x, gc);
				System.arraycopy(gc, 0, cg, (kc - 1) * nf + 1 - 1, nf);
			}
		}
		ld[0] = kd;
	}

	private void pf1f01(final Function<? super double[], Double> obj, final Function<? super double[], double[]> dobj,
			final int nf, final double[] x, final double[] gf, final double[] g, final double[] ff, final double[] f,
			final int kd, final int[] ld, final int iext) {
		if (kd <= ld[0]) {
			return;
		}

		if (ld[0] < 0) {
			++nfv;
			ff[0] = obj.apply(x);
			if (iext <= 0) {
				f[0] = ff[0];
			} else {
				f[0] = -ff[0];
			}
		}

		if (kd < 1 || ld[0] >= 1) {
		} else {
			++nfg;
			System.arraycopy(dobj.apply(x), 0, gf, 0, nf);
			if (iext > 0) {
				mxvneg(nf, gf, g);
			}
		}
		ld[0] = kd;
	}

	private void plqdb1(final int nf, final int nc, final double[] x, final int[] ix, final double[] xl,
			final double[] xu, final double[] cf, final double[] cfd, final int[] ic, final int[] ica,
			final double[] cl, final double[] cu, final double[] cg, final double[] cr, final double[] cz,
			final double[] g, final double[] go, final double[] h, final double[] s, final int mfp, final int kbf,
			final int kbc, final int[] idecf, final double eta2, final double eta9, final double eps7,
			final double eps9, final double[] umax, final double[] gmax, final int[] n, final int[] iterq) {

		final double[] temp = new double[1], step = new double[1], par = new double[1];
		final int[] inf = new int[1], inew = new int[1], knew = new int[1], ier = new int[1], krem = new int[1];
		double con, step1, step2, snorm = 0.0, dmax;
		int nca = 0, ncr = 0, i, j, k, iold, jold, jnew, kc, nred;

		con = eta9;
		if (idecf[0] < 0) {
			idecf[0] = 1;
		}
		if (idecf[0] == 0) {

			// GILL-MURRAY DECOMPOSITION
			temp[0] = eta2;
			mxdpgf(nf, h, inf, temp, step);
			++ndec;
			idecf[0] = 1;
		}
		if (idecf[0] >= 2 && idecf[0] <= 8) {
			iterq[0] = -10;
			return;
		}

		// INITIATION
		nred = jold = jnew = iterq[0] = 0;
		dmax = 0.0;
		if (mfp != 3) {
			n[0] = nf;
			nca = ncr = 0;
			if (kbf > 0) {
				mxvina(nf, ix);
			}
			if (kbc > 0) {
				mxvina(nc, ic);
			}
		}

		boolean do1 = true;
		while (true) {

			if (do1) {

				// DIRECTION DETERMINATION
				mxvneg(nf, go, s);
				for (j = 1; j <= nca; ++j) {
					kc = ica[j - 1];
					if (kc > 0) {
						mxvdir(nf, cz[j - 1], cg, (kc - 1) * nf + 1, s, s, 1);
					} else {
						k = -kc;
						s[k - 1] += cz[j - 1];
					}
				}
				System.arraycopy(s, 0, g, 0, nf);
				if (idecf[0] == 1) {
					mxdpgb(nf, h, s, 0);
				} else {
					mxdsmm(nf, h, g, 1, s);
				}
				if (iterq[0] == 3) {
					return;
				}

				// CHECK OF FEASIBILITY
				inew[0] = 0;
				par[0] = 0.0;
				plminn(nf, nc, cf, cfd, ic, cl, cu, cg, s, eps9, par, kbc, inew, knew);
				plmins(nf, ix, x, xl, xu, s, kbf, inew, knew, eps9, par);
				if (inew[0] == 0) {

					// SOLUTION ACHIEVED
					mxvneg(nf, g, g);
					iterq[0] = 2;
					return;
				} else {
					snorm = 0.0;
				}
			}
			ier[0] = 0;

			// STEPSIZE DETERMINATION
			pladr1(nf, n, ica, cg, cr, h, s, g, eps7, gmax, umax, idecf[0], inew[0], nadd, ier, 1);
			mxdprb(nca, cr, g, -1);
			if (knew[0] < 0) {
				mxvneg(nca, g, g);
			}

			// PRIMAL STEPSIZE
			if (ier[0] != 0) {
				step1 = con;
			} else {
				step1 = -par[0] / umax[0];
			}

			// DUAL STEPSIZE
			iold = 0;
			step2 = con;
			for (j = 1; j <= nca; ++j) {
				kc = ica[j - 1];
				if (kc >= 0) {
					k = ic[kc - 1];
				} else {
					i = -kc;
					k = ix[i - 1];
				}
				if (k <= -5) {
				} else if ((k == -1 || k == -3) && g[j - 1] <= 0.0) {
				} else if ((k == -2 || k == -4) && g[j - 1] >= 0.0) {
				} else {
					temp[0] = cz[j - 1] / g[j - 1];
					if (step2 > temp[0]) {
						iold = j;
						step2 = temp[0];
					}
				}
			}

			// FINAL STEPSIZE
			step[0] = Math.min(step1, step2);
			if (step[0] >= con) {

				// FEASIBLE SOLUTION DOES NOT EXIST
				iterq[0] = -1;
				return;
			}

			// NEW LAGRANGE MULTIPLIERS
			dmax = step[0];
			mxvdir(nca, -step[0], g, 1, cz, cz, 1);
			snorm += (RealMath.sign(1.0, knew[0]) * step[0]);
			par[0] -= ((step[0] / step1) * par[0]);
			if (step[0] == step1) {
				if (n[0] <= 0) {

					// IMPOSSIBLE SITUATION
					iterq[0] = -5;
					return;
				}

				// CONSTRAINT ADDITION
				if (ier[0] == 0) {
					--n[0];
					++nca;
					ncr += nca;
					cz[nca - 1] = snorm;
				}
				if (inew[0] > 0) {
					kc = inew[0];
					mxvinv(ic, kc, knew[0]);
				} else if (IntMath.abs(knew[0]) == 1) {
					i = -inew[0];
					mxvinv(ix, i, knew[0]);
				} else {
					i = -inew[0];
					if (knew[0] > 0) {
						ix[i - 1] = -3;
					}
					if (knew[0] < 0) {
						ix[i - 1] = -4;
					}
				}
				++nred;
				++nadd[0];
				jnew = inew[0];
				jold = 0;
				do1 = true;
			} else {

				// CONSTRAINT DELETION
				for (j = iold; j <= nca - 1; ++j) {
					cz[j - 1] = cz[j + 1 - 1];
				}
				plrmf0(nf, nc, ix, ic, ica, cr, ic, g, n, iold, krem, ier);
				ncr -= nca;
				--nca;
				jold = iold;
				jnew = 0;
				if (kbc > 0) {
					mxvina(nc, ic);
				}
				if (kbf > 0) {
					mxvina(nf, ix);
				}
				for (j = 1; j <= nca; ++j) {
					kc = ica[j - 1];
					if (kc > 0) {
						ic[kc - 1] = -ic[kc - 1];
					} else {
						kc = -kc;
						ix[kc - 1] = -ix[kc - 1];
					}
				}
				do1 = false;
			}
		}
	}

	private void ps0l02(final double[] r, final double ro, final double[] rp, final double f, final double fo,
			final double[] fp, final double po, final double[] pp, final double fmin, final double fmax,
			final double rmin, final double rmax, final double tols, final int[] kd, final int[] ld, final int nit,
			final int kit, final int[] nred, final int mred, final int[] maxst, final int iest, final int inits,
			final int[] iters, final int kters, final int mes, final int[] isys) {
		final int[] merr = new int[1];
		double rtemp, tol;
		int init1;
		boolean l1, l2, l3, l4, l6, l7;

		tol = 1.e-4;
		if (isys[0] == 1) {

			if (iters[0] != 0) {
				isys[0] = 0;
				return;
			}
			if (f <= fmin) {
				iters[0] = 7;
				isys[0] = 0;
				return;
			} else {
				l1 = r[0] <= rmin && nit != kit;
				l2 = r[0] >= rmax;
				l3 = f - fo <= tols * r[0] * po || f - fmin <= (fo - fmin) / 10.0;
				l4 = f - fo >= (1.0 - tols) * r[0] * po || mes2 == 2 && mode == 2;
				l6 = ru - rl <= tol * ru && mode == 2;
				l7 = mes2 <= 2 || mode != 0;
				maxst[0] = 0;
				if (l2) {
					maxst[0] = 1;
				}
			}

			// TEST ON TERMINATION
			if (l1 && !l3) {
				iters[0] = 0;
				isys[0] = 0;
				return;
			} else if (l2 && !(f >= fu)) {
				iters[0] = 7;
				isys[0] = 0;
				return;
			} else if (l6) {
				iters[0] = 1;
				isys[0] = 0;
				return;
			} else if (l3 && l7 && kters == 5) {
				iters[0] = 5;
				isys[0] = 0;
				return;
			} else if (l3 && l4 && l7 && (kters == 2 || kters == 3 || kters == 4)) {
				iters[0] = 2;
				isys[0] = 0;
				return;
			} else if (kters < 0 || kters == 6 && l7) {
				iters[0] = 6;
				isys[0] = 0;
				return;
			} else if (IntMath.abs(nred[0]) >= mred) {
				iters[0] = -1;
				isys[0] = 0;
				return;
			} else {
				rp[0] = r[0];
				fp[0] = f;
				mode = Math.max(mode, 1);
				mtyp = IntMath.abs(mes);
				if (f >= fmax) {
					mtyp = 1;
				}
			}

			if (mode == 1) {

				// INTERVAL CHANGE AFTER EXTRAPOLATION
				rl = ri;
				fl = fi;
				ri = ru;
				fi = fu;
				ru = r[0];
				fu = f;
				if (f >= fi) {
					nred[0] = 0;
					mode = 2;
				} else if (mes1 == 1) {
					mtyp = 1;
				}
			} else if (r[0] <= ri) {

				// INTERVAL CHANGE AFTER INTERPOLATION
				if (f <= fi) {
					ru = ri;
					fu = fi;
					ri = r[0];
					fi = f;
				} else {
					rl = r[0];
					fl = f;
				}
			} else if (f <= fi) {
				rl = ri;
				fl = fi;
				ri = r[0];
				fi = f;
			} else {
				ru = r[0];
				fu = f;
			}
		} else {

			mes1 = mes2 = 2;
			iters[0] = 0;
			if (po >= 0.0) {
				r[0] = 0.0;
				iters[0] = -2;
				isys[0] = 0;
				return;
			}
			if (rmax <= 0.0) {
				iters[0] = 0;
				isys[0] = 0;
				return;
			}

			// INITIAL STEPSIZE SELECTION
			if (inits > 0) {
				rtemp = fmin - f;
			} else if (iest == 0) {
				rtemp = f - fp[0];
			} else {
				rtemp = Math.max(f - fp[0], 10.0 * (fmin - f));
			}
			init1 = IntMath.abs(inits);
			rp[0] = 0.0;
			fp[0] = fo;
			pp[0] = po;
			if (init1 == 0) {
			} else if ((init1 == 1) || (inits >= 1) && (iest == 0)) {
				r[0] = 1.0;
			} else if (init1 == 2) {
				r[0] = Math.min(1.0, 4.0 * rtemp / po);
			} else if (init1 == 3) {
				r[0] = Math.min(1.0, 2.0 * rtemp / po);
			} else if (init1 == 4) {
				r[0] = 2.0 * rtemp / po;
			}
			// rtemp = r[0];
			r[0] = Math.max(r[0], rmin);
			r[0] = Math.min(r[0], rmax);
			mode = 0;
			rl = 0.0;
			fl = fo;
			ru = 0.0;
			fu = fo;
			ri = 0.0;
			fi = fo;
		}

		// NEW STEPSIZE SELECTION (EXTRAPOLATION OR INTERPOLATION)
		pnint3(ro, rl, ru, ri, fo, fl, fu, fi, po, r, mode, mtyp, merr);
		if (merr[0] > 0) {
			iters[0] = -merr[0];
			isys[0] = 0;
			return;
		} else if (mode == 1) {
			--nred[0];
			r[0] = Math.min(r[0], rmax);
		} else if (mode == 2) {
			++nred[0];
		}

		// COMPUTATION OF THE NEW FUNCTION VALUE
		kd[0] = 0;
		ld[0] = -1;
		isys[0] = 1;
	}

	private void plrmf0(final int nf, final int nc, final int[] ix, final int[] ia, final int[] iaa, final double[] ar,
			final int[] ic, final double[] s, final int[] n, final int iold, final int[] krem, final int[] ier) {
		int l;
		plrmr0(nf, iaa, ar, s, n[0], iold, krem, ier);
		++n[0];
		++nrem;
		l = iaa[nf - n[0] + 1 - 1];
		if (l > nc) {
			l -= nc;
			ia[l - 1] = -ia[l - 1];
		} else if (l > 0) {
			ic[l - 1] = -ic[l - 1];
		} else {
			l = -l;
			ix[l - 1] = -ix[l - 1];
		}
	}

	private static void plrmr0(final int nf, final int[] ica, final double[] cr, final double[] g, final int n,
			final int iold, final int[] krem, final int[] ier) {
		final double[] ck = new double[1], cl = new double[1], p1 = new double[1], p2 = new double[2];
		int i, j, k, kc, l, nca;

		nca = nf - n;
		if (iold < nca) {
			k = iold * (iold - 1) / 2;
			kc = ica[iold - 1];
			System.arraycopy(cr, k + 1 - 1, g, 0, iold);
			mxvset(nca - iold, 0.0, g, iold + 1);
			k += iold;
			for (i = iold + 1; i <= nca; ++i) {
				k += i;
				p1[0] = cr[k - 1 - 1];
				p2[0] = cr[k - 1];
				mxvort(p1, p2, ck, cl, ier);
				cr[k - 1 - 1] = p1[0];
				cr[k - 1] = p2[0];
				p1[0] = g[i - 1 - 1];
				p2[0] = g[i - 1];
				mxvrot(p1, p2, ck[0], cl[0], ier[0]);
				g[i - 1 - 1] = p1[0];
				g[i - 1] = p2[0];
				l = k;
				for (j = i; j <= nca - 1; ++j) {
					l += j;
					p1[0] = cr[l - 1 - 1];
					p2[0] = cr[l - 1];
					mxvrot(p1, p2, ck[0], cl[0], ier[0]);
					cr[l - 1 - 1] = p1[0];
					cr[l - 1] = p2[0];
				}
			}
			k = iold * (iold - 1) / 2;
			for (i = iold; i <= nca - 1; ++i) {
				l = k + i;
				ica[i - 1] = ica[i + 1 - 1];
				System.arraycopy(cr, l + 1 - 1, cr, k + 1 - 1, i);
				k = l;
			}
			ica[nca - 1] = kc;
			System.arraycopy(g, 0, cr, k + 1 - 1, nca);
		}
		krem[0] = 1;
	}

	private static void plmins(final int nf, final int[] ix, final double[] xo, final double[] xl, final double[] xu,
			final double[] s, final int kbf, final int[] inew, final int[] knew, final double eps9,
			final double[] par) {
		double pom, temp;
		int i;
		if (kbf > 0) {
			for (i = 1; i <= nf; ++i) {
				if (ix[i - 1] > 0) {
					temp = 1.0;
					if (ix[i - 1] == 1 || ix[i - 1] >= 3) {
						pom = xo[i - 1] + s[i - 1] * temp - xl[i - 1];
						final double max = Math.max(Math.abs(xl[i - 1]), temp);
						if (pom < Math.min(par[0], -eps9 * max)) {
							inew[0] = -i;
							knew[0] = 1;
							par[0] = pom;
						}
					}
					if (ix[i - 1] == 2 || ix[i - 1] >= 3) {
						pom = xu[i - 1] - s[i - 1] * temp - xo[i - 1];
						final double max = Math.max(Math.abs(xu[i - 1]), temp);
						if (pom < Math.min(par[0], -eps9 * max)) {
							inew[0] = -i;
							knew[0] = -1;
							par[0] = pom;
						}
					}
				}
			}
		}
	}

	private static void pladr1(final int nf, final int[] n, final int[] ica, final double[] cg, final double[] cr,
			final double[] h, final double[] s, final double[] g, final double eps7, final double[] gmax,
			final double[] umax, final int idecf, final int inew, final int[] nadd, final int[] ier, final int job) {
		int nca, ncr, jcg, j, k, l;

		ier[0] = 0;
		if (job == 0 && n[0] <= 0) {
			ier[0] = 2;
		}
		if (inew == 0) {
			ier[0] = 3;
		}
		if (idecf != 1 && idecf != 9) {
			ier[0] = -2;
		}
		if (ier[0] != 0) {
			return;
		}
		nca = nf - n[0];
		ncr = nca * (nca + 1) / 2;
		if (inew > 0) {
			jcg = (inew - 1) * nf + 1;
			if (idecf == 1) {
				System.arraycopy(cg, jcg - 1, s, 0, nf);
				mxdpgb(nf, h, s, 0);
			} else {
				mxdsmm(nf, h, cg, jcg, s);
			}
			gmax[0] = mxvdot(nf, cg, jcg, s, 1);
		} else {
			k = -inew;
			if (idecf == 1) {
				mxvset(nf, 0.0, s, 1);
				s[k - 1] = 1.0;
				mxdpgb(nf, h, s, 0);
			} else {
				mxdsmv(nf, h, s, k);
			}
			gmax[0] = s[k - 1];
		}
		for (j = 1; j <= nca; ++j) {
			l = ica[j - 1];
			if (l > 0) {
				g[j - 1] = mxvdot(nf, cg, (l - 1) * nf + 1, s, 1);
			} else {
				l = -l;
				g[j - 1] = s[l - 1];
			}
		}
		if (n[0] == 0) {
			mxdprb(nca, cr, g, 1);
			umax[0] = 0.0;
			ier[0] = 2;
			return;
		} else if (nca == 0) {
			umax[0] = gmax[0];
		} else {
			mxdprb(nca, cr, g, 1);
			umax[0] = gmax[0] - mxvdot(nca, g, 1, g, 1);
			System.arraycopy(g, 0, cr, ncr + 1 - 1, nca);
		}
		if (umax[0] <= eps7 * gmax[0]) {
			ier[0] = 1;
		} else {
			++nca;
			ncr += nca;
			ica[nca - 1] = inew;
			cr[ncr - 1] = Math.sqrt(umax[0]);
			if (job == 0) {
				--n[0];
				++nadd[0];
			}
		}
	}

	private static void plminn(final int nf, final int nc, final double[] cf, final double[] cfd, final int[] ic,
			final double[] cl, final double[] cu, final double[] cg, final double[] s, final double eps9,
			final double[] par, final int kbc, final int[] inew, final int[] knew) {
		double temp, pom;
		int jcg, kc;

		if (kbc > 0) {
			jcg = 1;
			for (kc = 1; kc <= nc; ++kc) {
				if (ic[kc - 1] > 0) {
					temp = mxvdot(nf, cg, jcg, s, 1);
					cfd[kc - 1] = temp;
					temp += cf[kc - 1];
					if (ic[kc - 1] == 1 || ic[kc - 1] >= 3) {
						pom = temp - cl[kc - 1];
						final double max = Math.max(Math.abs(cl[kc - 1]), 1.0);
						if (pom < Math.min(par[0], -eps9 * max)) {
							inew[0] = kc;
							knew[0] = 1;
							par[0] = pom;
						}
					}
					if (ic[kc - 1] == 2 || ic[kc - 1] >= 3) {
						pom = cu[kc - 1] - temp;
						final double max = Math.max(Math.abs(cu[kc - 1]), 1.0);
						if (pom < Math.min(par[0], -eps9 * max)) {
							inew[0] = kc;
							knew[0] = -1;
							par[0] = pom;
						}
					}
				}
				jcg += nf;
			}
		}
	}

	private static void pudbg1(final int n, final double[] h, final double[] g, final double[] s, final double[] xo,
			final double[] go, final double r, final double po, final int nit, final int kit, final int[] iterh,
			final int met, final int met1, final int mec) {
		double a, b, c, gam = 0.0, par = 0.0, den, dis;
		boolean l1, l3;

		l1 = met1 >= 3 || met1 == 2 && nit == kit;
		l3 = !l1;

		// DETERMINATION OF THE PARAMETERS B, C
		b = mxvdot(n, xo, 1, go, 1);
		a = 0.0;
		if (l1) {
			System.arraycopy(go, 0, s, 0, n);
			mxdpgb(n, h, s, 1);
			a = mxdpgp(n, h, s, s);
			if (a <= 0.0) {
				iterh[0] = 1;
				return;
			}
		}
		mxvdif(n, go, g, s);
		mxvscl(n, r, s, 1, s, 1);
		c = -r * po;
		if (c <= 0.0) {
			iterh[0] = 3;
			return;
		}
		if (mec > 1) {
			if (b <= 1.0e-4 * c) {

				// POWELL'S CORRECTION
				dis = (1.0 - 0.1) * c / (c - b);
				mxvdif(n, go, s, go);
				mxvdir(n, dis, go, 1, s, go, 1);
				b = c + dis * (b - c);
				if (l1) {
					a = c + 2.0 * (1.0 - dis) * (b - c) + dis * dis * (a - c);
				}
			}
		} else if (b <= 1.0e-4 * c) {
			iterh[0] = 2;
			return;
		}

		if (l1) {

			// DETERMINATION OF THE PARAMETER GAM (SELF SCALING)
			if (met == 1) {
				par = c / b;
			} else if (a <= 0.0) {
				par = c / b;
			} else {
				par = Math.sqrt(c / a);
			}
			gam = par;
			if (met1 > 1) {
				if (nit != kit) {
					l3 = gam < 0.5 || gam > 4.0;
				}
			}
		}
		if (l3) {
			gam = 1.0;
			par = gam;
		}
		if (met == 1) {

			// BFGS UPDATE
			mxdpgu(n, h, par / b, go, xo);
			mxdpgu(n, h, -1.0 / c, s, xo);
		} else {

			// HOSHINO UPDATE
			den = par * b + c;
			dis = 0.5 * b;
			mxvdir(n, par, go, 1, s, s, 1);
			mxdpgu(n, h, par / dis, go, xo);
			mxdpgu(n, h, -1.0 / den, s, xo);
		}
		iterh[0] = 0;
		if (gam == 1.0) {
			return;
		}
		mxdpgs(n, h, 1.0 / gam);
	}

	private static void pytrnd(final int nf, final int[] n, final double[] x, final double[] xo, final int[] ica,
			final double[] cg, final double[] cz, final double[] g, final double[] go, final double r, final double[] f,
			final double fo, final double[] p, final double[] po, final double[] cmax, final double cmaxo,
			final double[] dmax, final int kd, final int[] ld, final int iters) {
		int i, j, l;
		for (j = 1; j <= nf - n[0]; ++j) {
			l = ica[j - 1];
			if (l > 0) {
				mxvdir(nf, -cz[j - 1], cg, (l - 1) * nf + 1 - 1, g, g, 1);
			} else {
				l = -l;
				g[l - 1] -= cz[j - 1];
			}
		}
		if (iters > 0) {
			mxvdif(nf, x, xo, xo);
			mxvdif(nf, g, go, go);
			po[0] *= r;
			p[0] *= r;
		} else {
			f[0] = fo;
			p[0] = po[0];
			cmax[0] = cmaxo;
			mxvsav(nf, x, xo);
			mxvsav(nf, g, go);
			ld[0] = kd;
		}
		dmax[0] = 0.0;
		for (i = 1; i <= nf; ++i) {
			dmax[0] = Math.max(dmax[0], Math.abs(xo[i - 1]) / Math.max(Math.abs(x[i - 1]), 1.0));
		}
		n[0] = nf;
	}

	private static void pnint3(final double ro, final double rl, final double ru, final double ri, final double fo,
			final double fl, final double fu, final double fi, final double po, final double[] r, final int mode,
			final int mtyp, final int[] merr) {
		final double zero = 0.0, half = 0.5, one = 1.0, two = 2.0, three = 3.0;
		final double c1l = 1.1, c1u = 1.e3, c2l = 1.e-2, c2u = 0.9, c3l = 1.e-1;
		double ai = 0.0, al, au = 0.0, den, dis;
		int ntyp;
		boolean l1, l2;

		merr[0] = 0;
		if (mode <= 0) {
			return;
		}
		if (po >= zero) {
			merr[0] = 2;
			return;
		} else if (ru <= rl) {
			merr[0] = 3;
			return;
		}
		l1 = rl <= ro;
		l2 = ri <= rl;
		for (ntyp = mtyp; ntyp >= 1; --ntyp) {

			if (ntyp == 1) {

				// BISECTION
				if (mode == 1) {
					r[0] = two * ru;
					return;
				} else if (ri - rl <= ru - ri) {
					r[0] = half * (ri + ru);
					return;
				} else {
					r[0] = half * (rl + ri);
					return;
				}
			} else if (ntyp == mtyp && l1) {
				if (!l2) {
					ai = (fi - fo) / (ri * po);
				}
				au = (fu - fo) / (ru * po);
			}
			if (l1 && (ntyp == 2 || l2)) {

				// TWO POINT QUADRATIC EXTRAPOLATION OR INTERPOLATION
				if (au >= one) {
					continue;
				}
				r[0] = half * ru / (one - au);
			} else if (!l1 || !l2 && ntyp == 3) {

				// THREE POINT QUADRATIC EXTRAPOLATION OR INTERPOLATION
				al = (fi - fl) / (ri - rl);
				au = (fu - fi) / (ru - ri);
				den = au - al;
				if (den <= zero) {
					continue;
				}
				r[0] = ri - half * (au * (ri - rl) + al * (ru - ri)) / den;
			} else if (l1 && !l2 && ntyp == 4) {

				// THREE POINT CUBIC EXTRAPOLATION OR INTERPOLATION
				dis = (ai - one) * (ru / ri);
				den = (au - one) * (ri / ru) - dis;
				dis = au + ai - den - two * (one + dis);
				dis = den * den - three * dis;
				if (dis < zero) {
					continue;
				}
				den += Math.sqrt(dis);
				if (den == zero) {
					continue;
				}
				r[0] = (ru - ri) / den;
			} else {
				continue;
			}

			if (mode == 1 && r[0] > ru) {

				// EXTRAPOLATION ACCEPTED
				r[0] = Math.max(r[0], c1l * ru);
				r[0] = Math.min(r[0], c1u * ru);
				return;
			} else if (mode == 2 && r[0] > rl && r[0] < ru) {

				// INTERPOLATION ACCEPTED
				if (ri == zero && ntyp != 4) {
					r[0] = Math.max(r[0], rl + c2l * (ru - rl));
				} else {
					r[0] = Math.max(r[0], rl + c3l * (ru - rl));
				}
				r[0] = Math.min(r[0], rl + c2u * (ru - rl));
				if (r[0] == ri) {
					continue;
				}
				return;
			}
		}
	}

	private static void ppset2(final int nf, final int n, final int nc, final int[] ica, final double[] cz,
			final double[] cp) {
		double temp;
		int j, l;
		BlasMath.dscalm(nc, 0.5, cp, 1);
		for (j = 1; j <= nf - n; ++j) {
			l = ica[j - 1];
			if (l > 0) {
				temp = Math.abs(cz[j - 1]);
				cp[l - 1] = Math.max(temp, cp[l - 1] + 0.5 * temp);
			}
		}
	}

	private static void plredl(final int nc, final double[] cf, final int[] ic, final double[] cl, final double[] cu,
			final int kbc) {
		int kc, k;
		double temp;
		if (kbc > 0) {
			for (kc = 1; kc <= nc; kc++) {
				k = ic[kc - 1];
				if (IntMath.abs(k) == 1 || IntMath.abs(k) == 3 || IntMath.abs(k) == 4) {
					temp = cf[kc - 1] - cl[kc - 1];
					if (temp < 0.0) {
						cf[kc - 1] = cl[kc - 1] + 0.1 * temp;
					}
				}
				if (IntMath.abs(k) == 2 || IntMath.abs(k) == 3 || IntMath.abs(k) == 4) {
					temp = cf[kc - 1] - cu[kc - 1];
					if (temp > 0.0) {
						cf[kc - 1] = cu[kc - 1] + 0.1 * temp;
					}
				}
				if (IntMath.abs(k) == 5 || IntMath.abs(k) == 6) {
					temp = cf[kc - 1] - cl[kc - 1];
					cf[kc - 1] = cl[kc - 1] + 0.1 * temp;
				}
			}
		}
	}

	private static void plnews(final double[] x, final int[] ix, final double[] xl, final double[] xu,
			final double eps9, final int i, final int[] inew) {
		double temp = 1.0;
		if (ix[i - 1] <= 0) {
		} else if (ix[i - 1] == 1) {
			if (x[i - 1] <= xl[i - 1] + eps9 * Math.max(Math.abs(xl[i - 1]), temp)) {
				ix[i - 1] = 11;
				inew[0] = -i;
			}
		} else if (ix[i - 1] == 2) {
			if (x[i - 1] >= xu[i - 1] - eps9 * Math.max(Math.abs(xu[i - 1]), temp)) {
				ix[i - 1] = 12;
				inew[0] = -i;
			}
		} else if (ix[i - 1] == 3 || ix[i - 1] == 4) {
			if (x[i - 1] <= xl[i - 1] + eps9 * Math.max(Math.abs(xl[i - 1]), temp)) {
				ix[i - 1] = 13;
				inew[0] = -i;
			}
			if (x[i - 1] >= xu[i - 1] - eps9 * Math.max(Math.abs(xu[i - 1]), temp)) {
				ix[i - 1] = 14;
				inew[0] = -i;
			}
		}
	}

	private static void pp0af8(final int nf, final int n, final int nc, final double[] cf, final int[] ic,
			final int[] ica, final double[] cl, final double[] cu, final double[] cz, final double rpf,
			final double[] fc, final double[] f) {
		double pom, temp;
		int j, kc;
		fc[0] = 0.0;

		for (kc = 1; kc <= nc; ++kc) {
			if (ic[kc - 1] > 0) {
				pom = 0.0;
				temp = cf[kc - 1];
				if (ic[kc - 1] == 1 || ic[kc - 1] >= 3) {
					pom = Math.min(pom, temp - cl[kc - 1]);
				}
				if (ic[kc - 1] == 2 || ic[kc - 1] >= 3) {
					pom = Math.min(pom, cu[kc - 1] - temp);
				}
				fc[0] += rpf * Math.abs(pom);
			}
		}
		for (j = 1; j <= nf - n; ++j) {
			kc = ica[j - 1];
			if (kc > 0) {
				pom = 0.0;
				temp = cf[kc - 1];
				if (ic[kc - 1] == 1 || ic[kc - 1] == 3 || ic[kc - 1] == 5) {
					pom = Math.min(pom, temp - cl[kc - 1]);
				}
				if (ic[kc - 1] == 2 || ic[kc - 1] == 4 || ic[kc - 1] == 6) {
					pom = Math.max(pom, temp - cu[kc - 1]);
				}
				fc[0] -= cz[j - 1] * pom;
			}
		}
		f[0] = cf[nc + 1 - 1] + fc[0];
	}

	private static void mxdprb(final int n, final double[] a, final double[] x, final int job) {
		int i, ii, ij, j;

		if (job >= 0) {

			// PHASE 1 : X:=TRANS(R)**(-1)*X
			ij = 0;
			for (i = 1; i <= n; ++i) {
				for (j = 1; j <= i - 1; ++j) {
					++ij;
					x[i - 1] -= (a[ij - 1] * x[j - 1]);
				}
				++ij;
				x[i - 1] /= a[ij - 1];
			}
		}
		if (job <= 0) {

			// PHASE 2 : X:=R**(-1)*X
			ii = n * (n + 1) / 2;
			for (i = n; i >= 1; --i) {
				ij = ii;
				for (j = i + 1; j <= n; ++j) {
					ij += (j - 1);
					x[i - 1] -= (a[ij - 1] * x[j - 1]);
				}
				x[i - 1] /= a[ii - 1];
				ii -= i;
			}
		}
	}

	private static void mxdpgf(final int n, final double[] a, final int[] inf, final double[] alf, final double[] tau) {
		double bet, del, gam, rho, sig, tol;
		int i, ij, ik, j, k, kj, kk, l;
		l = inf[0] = 0;
		tol = alf[0];

		// ESTIMATION OF THE MATRIX NORM
		alf[0] = bet = gam = tau[0] = 0.0;
		kk = 0;
		for (k = 1; k <= n; ++k) {
			kk += k;
			bet = Math.max(bet, Math.abs(a[kk - 1]));
			kj = kk;
			for (j = k + 1; j <= n; ++j) {
				kj += (j - 1);
				gam = Math.max(gam, Math.abs(a[kj - 1]));
			}
		}
		bet = Math.max(tol, Math.max(bet, gam / n));
		del = tol * Math.max(bet, 1.0);
		kk = 0;
		for (k = 1; k <= n; ++k) {
			kk += k;

			// DETERMINATION OF A DIAGONAL CORRECTION
			sig = a[kk - 1];
			if (alf[0] > sig) {
				alf[0] = sig;
				l = k;
			}
			gam = 0.0;
			kj = kk;
			for (j = k + 1; j <= n; ++j) {
				kj += (j - 1);
				gam = Math.max(gam, Math.abs(a[kj - 1]));
			}
			gam = gam * gam;
			rho = Math.max(Math.abs(sig), Math.max(gam / bet, del));
			if (tau[0] < rho - sig) {
				tau[0] = rho - sig;
				inf[0] = -1;
			}

			// GAUSSIAN ELIMINATION
			a[kk - 1] = rho;
			kj = kk;
			for (j = k + 1; j <= n; ++j) {
				kj += (j - 1);
				gam = a[kj - 1];
				a[kj - 1] = gam / rho;
				ik = kk;
				ij = kj;
				for (i = k + 1; i <= j; ++i) {
					ik += (i - 1);
					++ij;
					a[ij - 1] -= (a[ik - 1] * gam);
				}
			}
		}
		if (l > 0 && Math.abs(alf[0]) > del) {
			inf[0] = l;
		}
	}

	private static double mxdpgp(final int n, final double[] a, final double[] x, final double[] y) {
		double temp = 0.0;
		int i, j = 0;
		for (i = 1; i <= n; ++i) {
			j += i;
			temp += (x[i - 1] * y[i - 1] / a[j - 1]);
		}
		return temp;
	}

	private static void mxdpgb(final int n, final double[] a, final double[] x, final int job) {
		int i, ii, ij, j;

		if (job >= 0) {

			// PHASE 1 : X:=L**(-1)*X
			ij = 0;
			for (i = 1; i <= n; ++i) {
				for (j = 1; j <= i - 1; ++j) {
					++ij;
					x[i - 1] -= (a[ij - 1] * x[j - 1]);
				}
				++ij;
			}
		}
		if (job == 0) {

			// PHASE 2 : X:=D**(-1)*X
			ii = 0;
			for (i = 1; i <= n; ++i) {
				ii += i;
				x[i - 1] /= a[ii - 1];
			}
		}

		if (job <= 0) {

			// PHASE 3 : X:=TRANS(L)**(-1)*X
			ii = n * (n - 1) / 2;
			for (i = n - 1; i >= 1; --i) {
				ij = ii;
				for (j = i + 1; j <= n; ++j) {
					ij += (j - 1);
					x[i - 1] -= (a[ij - 1] * x[j - 1]);
				}
				ii -= i;
			}
		}
	}

	private static void mxdpgs(final int n, final double[] a, final double alf) {
		int i, j = 0;
		for (i = 1; i <= n; ++i) {
			j += i;
			a[j - 1] *= alf;
		}
	}

	private static void mxdpgu(final int n, final double[] a, final double alf, final double[] x, final double[] y) {
		final double zero = 0.0, one = 1.0, four = 4.0, con = 1.0e-8;
		double alfr, b, d, p, r, t, to;
		int i, ii, ij, j;

		if (alf >= zero) {

			// FORWARD CORRECTION IN CASE WHEN THE SCALING FACTOR IS NONNEGATIVE
			alfr = Math.sqrt(alf);
			mxvscl(n, alfr, x, 1, y, 1);
			to = one;
			ii = 0;
			for (i = 1; i <= n; ++i) {
				ii += i;
				d = a[ii - 1];
				p = y[i - 1];
				t = to + p * p / d;
				r = to / t;
				a[ii - 1] = d / r;
				b = p / (d * t);
				if (a[ii - 1] <= four * d) {

					// AN EASY FORMULA FOR LIMITED DIAGONAL ELEMENT
					ij = ii;
					for (j = i + 1; j <= n; ++j) {
						ij += (j - 1);
						d = a[ij - 1];
						y[j - 1] -= (p * d);
						a[ij - 1] = d + b * y[j - 1];
					}
				} else {

					// A MORE COMPLICATE BUT NUMERICALLY STABLE FORMULA FOR
					// UNLIMITED DIAGONAL ELEMENT
					ij = ii;
					for (j = i + 1; j <= n; ++j) {
						ij += (j - 1);
						d = a[ij - 1];
						a[ij - 1] = r * d + b * y[j - 1];
						y[j - 1] -= (p * d);
					}
				}
				to = t;
			}
		} else {

			// BACKWARD CORRECTION IN CASE WHEN THE SCALING FACTOR IS NEGATIVE
			alfr = Math.sqrt(-alf);
			mxvscl(n, alfr, x, 1, y, 1);
			to = one;
			ij = 0;
			for (i = 1; i <= n; ++i) {
				d = y[i - 1];
				for (j = 1; j <= i - 1; ++j) {
					++ij;
					d -= (a[ij - 1] * y[j - 1]);
				}
				y[i - 1] = d;
				++ij;
				to -= (d * d / a[ij - 1]);
			}
			if (to <= zero) {
				to = con;
			}
			ii = n * (n + 1) / 2;
			for (i = n; i >= 1; --i) {
				d = a[ii - 1];
				p = y[i - 1];
				t = to + p * p / d;
				a[ii - 1] = d * to / t;
				b = -p / (d * to);
				to = t;
				ij = ii;
				for (j = i + 1; j <= n; ++j) {
					ij += (j - 1);
					d = a[ij - 1];
					a[ij - 1] = d + b * y[j - 1];
					y[j - 1] += (p * d);
				}
				ii -= i;
			}
		}
	}

	protected static final void mxdsmi(final int n, final double[] a) {
		int i, m;
		m = n * (n + 1) / 2;
		Arrays.fill(a, 0, m, 0.0);
		m = 0;
		for (i = 1; i <= n; ++i) {
			m += i;
			a[m - 1] = 1.0;
		}
	}

	protected static final void mxdsmv(final int n, final double[] a, final double[] x, final int k) {
		int i, l;
		l = k * (k - 1) / 2;
		for (i = 1; i <= n; ++i) {
			if (i <= k) {
				++l;
			} else {
				l += (i - 1);
			}
			x[i - 1] = a[l - 1];
		}
	}

	protected static final void mxvneg(final int n, final double[] x, final double[] y) {
		for (int i = 1; i <= n; ++i) {
			y[i - 1] = -x[i - 1];
		}
	}

	protected static void mxdsmm(final int n, final double[] a, final double[] x, final int ix, final double[] y) {
		double temp;
		int i, j, k = 0, l;
		for (i = 1; i <= n; ++i) {
			temp = 0.0;
			l = k;
			for (j = 1; j <= i; ++j) {
				++l;
				temp += (a[l - 1] * x[j - 1 + ix - 1]);
			}
			for (j = i + 1; j <= n; ++j) {
				l += (j - 1);
				temp += (a[l - 1] * x[j - 1 + ix - 1]);
			}
			y[i - 1] = temp;
			k += i;
		}
	}

	protected static final void mxvina(final int n, final int[] ix) {
		for (int i = 1; i <= n; ++i) {
			ix[i - 1] = IntMath.abs(ix[i - 1]);
			if (ix[i - 1] > 10) {
				ix[i - 1] -= 10;
			}
		}
	}

	protected static final void mxvinv(final int[] ix, final int i, final int job) {
		if ((ix[i - 1] == 3 || ix[i - 1] == 5) && job < 0) {
			++ix[i - 1];
		}
		if ((ix[i - 1] == 4 || ix[i - 1] == 6) && job > 0) {
			--ix[i - 1];
		}
		ix[i - 1] = -ix[i - 1];
	}

	protected static final void mxvset(final int n, final double a, final double[] x, final int ix) {
		Arrays.fill(x, ix - 1, n - 1 + ix, a);
	}

	protected static final void mxvsav(final int n, final double[] x, final double[] y) {
		double temp;
		for (int i = 1; i <= n; ++i) {
			temp = y[i - 1];
			y[i - 1] = x[i - 1] - y[i - 1];
			x[i - 1] = temp;
		}
	}

	protected static final void mxvscl(final int n, final double a, final double[] x, final int ix, final double[] y,
			final int iy) {
		BlasMath.dscal1(n, a, x, ix, y, iy);
	}

	protected static final void mxvdir(final int n, final double a, final double[] x, final int ix, final double[] y,
			final double[] z, final int iz) {
		BlasMath.daxpy1(n, a, x, ix, y, 1, z, iz);
	}

	protected static final double mxvdot(final int n, final double[] x, final int ix, final double[] y, final int iy) {
		return BlasMath.ddotm(n, x, ix, y, iy);
	}

	protected static final void mxvdif(final int n, final double[] x, final double[] y, final double[] z) {
		for (int i = 1; i <= n; ++i) {
			z[i - 1] = x[i - 1] - y[i - 1];
		}
	}

	protected static final double mxvmax(final int n, final double[] x) {
		double mxvmax = 0.0;
		for (int i = 1; i <= n; ++i) {
			mxvmax = Math.max(mxvmax, Math.abs(x[i - 1]));
		}
		return mxvmax;
	}

	private static void mxvort(final double[] xk, final double[] xl, final double[] ck, final double[] cl,
			final int[] ier) {
		double den, pom;
		if (xl[0] == 0.0) {
			ier[0] = 2;
		} else if (xk[0] == 0.0) {
			xk[0] = xl[0];
			xl[0] = 0.0;
			ier[0] = 1;
		} else {
			if (Math.abs(xk[0]) >= Math.abs(xl[0])) {
				pom = xl[0] / xk[0];
				den = RealMath.hypot(1.0, pom);
				ck[0] = 1.0 / den;
				cl[0] = pom / den;
				xk[0] *= den;
			} else {
				pom = xk[0] / xl[0];
				den = RealMath.hypot(1.0, pom);
				cl[0] = 1.0 / den;
				ck[0] = pom / den;
				xk[0] = xl[0] * den;
			}
			xl[0] = 0.0;
			ier[0] = 0;
		}
	}

	private static void mxvrot(final double[] xk, final double[] xl, final double ck, final double cl, final int ier) {
		double yk, yl;
		if (ier == 0) {
			yk = xk[0];
			yl = xl[0];
			xk[0] = ck * yk + cl * yl;
			xl[0] = cl * yk - ck * yl;
		} else if (ier == 1) {
			yk = xk[0];
			xk[0] = xl[0];
			xl[0] = yk;
		}
	}
}
