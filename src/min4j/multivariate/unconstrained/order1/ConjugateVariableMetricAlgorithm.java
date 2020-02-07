package min4j.multivariate.unconstrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import min4j.utils.BlasMath;
import min4j.utils.IntMath;
import min4j.utils.RealMath;

/**
 *
 * @author Michael
 */
public final class ConjugateVariableMetricAlgorithm extends GradientOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	private interface Obj {

		double obj(int nf, double[] x);
	}

	private interface DObj {

		void dobj(int nf, double[] x, double[] gf);
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int[] nit = new int[1];
	private int nfv, nfg;

	private int mtyp, mode, mes1, mes2, mes3;
	private double rl, fl, pl, ru, fu, pu;

	private final double tolx, tolf, toldf, tolg, maxStep;
	private final int maxEvals, maxUpdt;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolX
	 * @param tolF
	 * @param tolDf
	 * @param tolG
	 * @param maxStepSize
	 * @param maxEvaluations
	 * @param maxMetricUpdates
	 */
	public ConjugateVariableMetricAlgorithm(final double tolX, final double tolF, final double tolDf, final double tolG,
			final double maxStepSize, final int maxEvaluations, final int maxMetricUpdates) {
		super(tolX);
		tolx = tolX;
		tolf = tolF;
		toldf = tolDf;
		tolg = tolG;
		maxStep = maxStepSize;
		maxEvals = maxEvaluations;
		maxUpdt = maxMetricUpdates;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxStepSize
	 * @param maxEvaluations
	 * @param maxMetricUpdates
	 */
	public ConjugateVariableMetricAlgorithm(final double tolerance, final double maxStepSize, final int maxEvaluations,
			final int maxMetricUpdates) {
		this(tolerance, tolerance, tolerance, tolerance, maxStepSize, maxEvaluations, maxMetricUpdates);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], Double> func,
			final Function<? super double[], double[]> dfunc, final double[] guess) {

		// prepare functions
		final Obj obj = (pnf, px) -> func.apply(px);
		final DObj dobj = (pnf, px, pgf) -> {
			System.arraycopy(dfunc.apply(px), 0, pgf, 0, pnf);
		};

		// prepare other variables
		final int[] nf = { guess.length };
		final double[] x = Arrays.copyOf(guess, nf[0]);
		final int[] ipar = { MAX_ITERS, maxEvals, 0, 0, 0, 0, maxUpdt };
		final double[] rpar = { maxStep, tolx, toldf, tolf, tolg, 0.0, 0.0, 0.0, 0.0 };
		final double[] f = { func.apply(x) };
		final double[] gmax = { 0.0 };
		final int iprnt = 0;
		final int[] iterm = new int[1];

		// call main subroutine
		plicu(obj, dobj, nf, x, ipar, rpar, f, gmax, iprnt, iterm);
		myEvals += nfv;
		myGEvals += nfg;
		if (iterm[0] == 1 || iterm[0] == 2 || iterm[0] == 3 || iterm[0] == 4 || iterm[0] == 6) {
			return x;
		} else {
			return null;
		}
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void plicu(final Obj obj, final DObj dobj, final int[] nf, final double[] x, final int[] ipar,
			final double[] rpar, final double[] f, final double[] gmax, final int iprnt, final int[] iterm) {
		int mf, nb;
		mf = ipar[7 - 1];
		if (mf <= 0) {
			mf = 5;
		}
		final double[] ra = new double[nf[0]];
		nb = 0;

		// POINTERS FOR AUXILIARY ARRAYS
		final double[] ra1 = new double[nf[0]], ra2 = new double[nf[0]], ra3 = new double[nf[0]],
				ra4 = new double[nf[0]], ra5 = new double[nf[0] * mf], ra6 = new double[nf[0] * mf],
				ra7 = new double[mf * mf], ra8 = new double[mf * mf];
		final double[] rpar1 = { rpar[0] }, rpar2 = { rpar[1] }, rpar3 = { rpar[2] }, rpar4 = { rpar[3] },
				rpar5 = { rpar[4] }, rpar6 = { rpar[5] };
		final int[] ipar1 = { ipar[0] }, ipar2 = { ipar[1] }, ipar3 = { ipar[3] };

		// CALL MAIN SUBROUTINE
		plic(obj, dobj, nf, nb, x, ipar, ra, ra, ra1, ra2, ra3, ra4, ra5, ra6, ra7, ra8, rpar1, rpar2, rpar3, rpar4,
				rpar5, rpar6, gmax, f, ipar1, ipar2, ipar3, mf, iprnt, iterm);

		// COPY MODIFIED PARAMETERS BACK
		rpar[0] = rpar1[0];
		rpar[1] = rpar2[0];
		rpar[2] = rpar3[0];
		rpar[3] = rpar4[0];
		rpar[4] = rpar5[0];
		rpar[5] = rpar6[0];
		ipar[0] = ipar1[0];
		ipar[1] = ipar2[0];
		ipar[3] = ipar3[0];
	}

	private void plic(final Obj obj, final DObj dobj, final int[] nf, final int nb, final double[] x, final int[] ix,
			final double[] xl, final double[] xu, final double[] gf, final double[] s, final double[] xo,
			final double[] go, final double[] xm, final double[] gm, final double[] xr, final double[] gr,
			final double[] xmax, final double[] tolx, final double[] tolf, final double[] tolb, final double[] tolg,
			final double[] fmin, final double[] gmax, final double[] f, final int[] mit, final int[] mfv,
			final int[] iest, final int mf, final int iprnt, final int[] iterm) {
		final int[] inew = new int[1], n = new int[1], iold = new int[1], ntesx = new int[1], ntesf = new int[1],
				irest = new int[1], kd = new int[1], ld = new int[1], nred = new int[1], maxst = new int[1],
				iters = new int[1], isys = new int[1];
		int iterd, mtesx, mtesf, mred, kit, kbf, mes, ites, inits, kters, ires1, ires2, i, mfg, k, l, mx = 0, met;
		final double[] umax = new double[1], fo = new double[1], ro = new double[1], fp = new double[1],
				po = new double[1], rmax = new double[1], r = new double[1], rp = new double[1], pp = new double[1],
				par1 = new double[1], par2 = new double[1], dmax = new double[1], p = new double[1];
		double gnorm, snorm, rmin = 0.0, fmax, eta9, eps8, eps9, alf1, alf2, b = 0.0, bp, q1, q2, q3, told, tols, tolp;

		// INITIATION
		kbf = 0;
		if (nb > 0) {
			kbf = 2;
		}
		nit[0] = nfv = nfg = isys[0] = 0;
		ites = 1;
		mtesx = mtesf = inits = 2;
		iterm[0] = iterd = 0;
		iters[0] = 2;
		kters = 3;
		irest[0] = 0;
		ires1 = 999;
		ires2 = 0;
		mred = 10;
		met = 1;
		mes = 4;
		mes1 = mes2 = mes3 = 2;
		// eta0 = 1.0e-15;
		eta9 = 1.0e120;
		eps8 = 1.0;
		eps9 = 1.0e-8;
		alf1 = 1.0e-10;
		alf2 = 1.0e10;
		rmax[0] = dmax[0] = eta9;
		fmax = 1.0e20;
		if (iest[0] <= 0) {
			fmin[0] = -1.0e60;
		}
		if (iest[0] > 0) {
			iest[0] = 1;
		}
		if (xmax[0] <= 0.0) {
			xmax[0] = 1.0e16;
		}
		if (tolx[0] <= 0.0) {
			tolx[0] = 1.0e-16;
		}
		if (tolf[0] <= 0.0) {
			tolf[0] = 1.0e-14;
		}
		if (tolg[0] <= 0.0) {
			tolg[0] = 1.0e-6;
		}
		if (tolb[0] <= 0.0) {
			tolb[0] = fmin[0] + 1.0e-16;
		}
		told = tols = 1.0e-4;
		tolp = 0.8;
		if (mit[0] <= 0) {
			mit[0] = 9000;
		}
		if (mfv[0] <= 0) {
			mfv[0] = 9000;
		}
		mfg = mfv[0];
		kd[0] = 1;
		ld[0] = -1;
		kit = -(ires1 * nf[0] + ires2);
		fo[0] = fmin[0];

		// INITIAL OPERATIONS WITH SIMPLE BOUNDS
		if (kbf > 0) {
			for (i = 1; i <= nf[0]; ++i) {
				if ((ix[i - 1] == 3 || ix[i - 1] == 4) && xu[i - 1] <= xl[i - 1]) {
					xu[i - 1] = xl[i - 1];
					ix[i - 1] = 5;
				} else if (ix[i - 1] == 5 || ix[i - 1] == 6) {
					xl[i - 1] = xu[i - 1] = x[i - 1];
					ix[i - 1] = 5;
				}
			}
			pcbs04(nf[0], x, ix, xl, xu, eps9, kbf);
			pyadc0(nf[0], n, x, ix, xl, xu, inew);
		}
		if (iterm[0] != 0) {
			return;
		}
		f[0] = obj.obj(nf[0], x);
		++nfv;
		dobj.dobj(nf[0], x, gf);
		++nfg;

		boolean do11120 = true;
		while (true) {

			if (do11120) {
				pytrcg(nf[0], nf, ix, gf, umax, gmax, kbf, iold);
				pyfut1(nf[0], f[0], fo, umax[0], gmax[0], dmax[0], tolx[0], tolf[0], tolb[0], tolg[0], kd[0], nit, kit,
						mit[0], nfv, mfv[0], nfg, mfg, ntesx, mtesx, ntesf, mtesf, ites, ires1, ires2, irest, iters[0],
						iterm);
				if (iterm[0] != 0) {
					return;
				}
				if (kbf > 0 && rmax[0] > 0.0) {
					pyrmc0(nf[0], n[0], ix, gf, eps8, umax[0], gmax[0], rmax[0], iold, irest);
				}
			}

			// DIRECTION DETERMINATION
			gnorm = Math.sqrt(mxudot(nf[0], gf, 1, gf, 1, ix, kbf));
			if (irest[0] != 0) {

				// STEEPEST DESCENT DIRECTION
				mx = 0;
				mxuneg(nf[0], gf, s, ix, kbf);
				snorm = gnorm;
				if (kit < nit[0]) {
					kit = nit[0];
				} else {
					iterm[0] = -10;
					if (iters[0] < 0) {
						iterm[0] = iters[0] - 5;
					}
				}
			} else {

				// LIMITED MEMORY VARIABLE METRIC DIRECTION
				mxdsrv(nf[0], mx, xm, gm, gf, s, gr, xr, b / mxudot(nf[0], go, 1, go, 1, ix, kbf));
				mxuneg(nf[0], s, s, ix, kbf);
				snorm = Math.sqrt(mxudot(nf[0], s, 1, s, 1, ix, kbf));
			}

			// TEST ON DESCENT DIRECTION AND PREPARATION OF LINE SEARCH
			if (kd[0] > 0) {
				p[0] = mxudot(nf[0], gf, 1, s, 1, ix, kbf);
			}
			if (iterd < 0) {
				iterm[0] = iterd;
			} else {

				// TEST ON DESCENT DIRECTION
				if (snorm <= 0.0) {
					irest[0] = Math.max(irest[0], 1);
				} else if (p[0] + told * gnorm * snorm <= 0.0) {
					irest[0] = 0;
				} else {

					// UNIFORM DESCENT CRITERION
					irest[0] = Math.max(irest[0], 1);
				}
				if (irest[0] == 0) {

					// PREPARATION OF LINE SEARCH
					nred[0] = 0;
					rmin = alf1 * gnorm / snorm;
					rmax[0] = Math.min(alf2 * gnorm / snorm, xmax[0] / snorm);
				}
			}
			if (iterm[0] != 0) {
				return;
			}
			if (irest[0] != 0) {
				do11120 = false;
				continue;
			}
			pytrcs(nf[0], x, ix, xo, xl, xu, gf, go, s, ro, fp, fo, f[0], po, p[0], rmax, eta9, kbf);
			if (rmax[0] != 0.0) {

				while (true) {
					ps1l01(r, rp, f[0], fo[0], fp, p[0], po[0], pp, fmin[0], fmax, rmin, rmax[0], tols, tolp, par1,
							par2, kd, ld, nit[0], kit, nred, mred, maxst, iest[0], inits, iters, kters, mes, isys);
					if (isys[0] == 0) {
						break;
					} else {
						mxudir(nf[0], r[0], s, xo, x, ix, kbf);
						pcbs04(nf[0], x, ix, xl, xu, eps9, kbf);
						f[0] = obj.obj(nf[0], x);
						++nfv;
						dobj.dobj(nf[0], x, gf);
						++nfg;
						p[0] = mxudot(nf[0], gf, 1, s, 1, ix, kbf);
					}
				}

				if (iters[0] <= 0) {
					r[0] = 0.0;
					f[0] = fo[0];
					p[0] = po[0];
					System.arraycopy(xo, 0, x, 0, nf[0]);
					System.arraycopy(go, 0, gf, 0, nf[0]);
					irest[0] = Math.max(irest[0], 1);
					ld[0] = kd[0];
					do11120 = false;
					continue;
				}
				pytrcd(nf[0], x, ix, xo, gf, go, r[0], f, fo[0], p, po, dmax, kbf, kd[0], ld, iters[0]);
			}

			// 11175
			b = mxudot(nf[0], xo, 1, go, 1, ix, kbf);
			if (b <= 0.0) {
				irest[0] = Math.max(irest[0], 1);
			} else {

				// AKTUALIZACE REDUKOVANYCH MATIC XM, GM A POLE XR
				if (mx >= mf) {
					--mx;
					System.arraycopy(xm, nf[0] + 1 - 1, xm, 0, nf[0] * mx);
					System.arraycopy(gm, nf[0] + 1 - 1, gm, 0, nf[0] * mx);
				}
				bp = b;
				k = 0;
				l = (mx - 1) * nf[0] + 1;
				if (mx > 0) {
					q1 = mxudot(nf[0], xm, l, go, 1, ix, kbf);
					q2 = mxudot(nf[0], gm, l, xo, 1, ix, kbf);
					q3 = q1;
					if (q1 * q2 > 0.0) {
						bp = b - q1 * q2;
						if (met == 1) {
							if (Math.abs(q2 - q1) * b < 1.0) {
								k = 1;
								if (bp > 1.0e-2 * b) {
									q3 = RealMath.sign(Math.sqrt(q1 * q2), q2);
								}
							}
						} else if (met == 2) {
							if ((q2 - q1) * (q2 - q1) * b * (q2 - q1) * (q2 - q1) * b < bp) {
								k = 1;
								if (bp > 1.0e-2 * b) {
									q3 = RealMath.sign(Math.sqrt(q1 * q2), q2)
											+ (q2 - q1) * (q2 - q1) * (q2 - q1) * b * b;
								}
							}
						}
					}
					if (k == 1) {
						mxvdir(nf[0], -q2, xm, l, xo, xm, nf[0] + l);
						mxvdir(nf[0], -q3, gm, l, go, gm, nf[0] + l);
						bp = mxudot(nf[0], xm, nf[0] + l, gm, nf[0] + l, ix, kbf);
						if (bp <= 0.0) {
							k = 0;
						}
					}
				}
				if (k == 1) {
					mxvscl(nf[0], 1.0 / Math.sqrt(bp), xm, nf[0] + l, xm, nf[0] + l);
					mxvscl(nf[0], 1.0 / Math.sqrt(bp), gm, nf[0] + l, gm, nf[0] + l);
				} else {
					mxvscl(nf[0], 1.0 / Math.sqrt(b), xo, 1, xm, nf[0] + l);
					mxvscl(nf[0], 1.0 / Math.sqrt(b), go, 1, gm, nf[0] + l);
				}
				++mx;
				xr[mx - 1] = 1.0;
			}
			if (kbf > 0) {
				mxvine(nf[0], ix);
				pyadc0(nf[0], n, x, ix, xl, xu, inew);
			}
			do11120 = true;
		}
	}

	private void ps1l01(final double[] r, final double[] rp, final double f, final double fo, final double[] fp,
			final double p, final double po, final double[] pp, final double fmin, final double fmax, final double rmin,
			final double rmax, final double tols, final double tolp, final double[] par1, final double[] par2,
			final int[] kd, final int[] ld, final int nit, final int kit, final int[] nred, final int mred,
			final int[] maxst, final int iest, final int inits, final int[] iters, final int kters, final int mes,
			final int[] isys) {
		final int[] merr = new int[1];
		int init1;
		boolean l1, l2, l3, l5, l7, m1, m2, m3;
		double rtemp, con = 1.0e-2, con1 = 1.0e-13;

		if (isys[0] == 1) {
			if (mode == 0) {
				par1[0] = p / po;
				par2[0] = f - fo;
			}
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
				l3 = f - fo <= tols * r[0] * po;
				l5 = p >= tolp * po || mes2 == 2 && mode == 2;
				l7 = mes2 <= 2 || mode != 0;
				// m1 = m2 = false;
				m3 = l3;
				if (mes3 >= 1) {
					m1 = Math.abs(p) <= con * Math.abs(po) && fo - f >= (con1 / con) * Math.abs(fo);
					l3 = l3 || m1;
				}
				if (mes3 >= 2) {
					m2 = Math.abs(p) <= 0.5 * Math.abs(po) && Math.abs(fo - f) <= 2.0 * con1 * Math.abs(fo);
					l3 = l3 || m2;
				}
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
			} else if (l2 && l3 && !l5) {
				iters[0] = 7;
				isys[0] = 0;
				return;
			} else if (m3 && mes1 == 3) {
				iters[0] = 5;
				isys[0] = 0;
				return;
			} else if (l3 && l5 && l7) {
				iters[0] = 4;
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
				pp[0] = p;
				mode = Math.max(mode, 1);
				mtyp = IntMath.abs(mes);
				if (f >= fmax) {
					mtyp = 1;
				}
			}
			if (mode == 1) {

				// INTERVAL CHANGE AFTER EXTRAPOLATION
				rl = ru;
				fl = fu;
				pl = pu;
				ru = r[0];
				fu = f;
				pu = p;
				if (!l3) {
					nred[0] = 0;
					mode = 2;
				} else if (mes1 == 1) {
					mtyp = 1;
				}
			} else // INTERVAL CHANGE AFTER INTERPOLATION
			if (!l3) {
				ru = r[0];
				fu = f;
				pu = p;
			} else {
				rl = r[0];
				fl = f;
				pl = p;
			}
		} else {
			mes1 = mes2 = mes3 = 2;
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
				rtemp = Math.max(f - fp[0], fmin - f);
			}
			init1 = IntMath.abs(inits);
			rp[0] = 0.0;
			fp[0] = fo;
			pp[0] = po;
			if (init1 == 0) {
			} else if (init1 == 1 || inits >= 1 && iest == 0) {
				r[0] = 1.0;
			} else if (init1 == 2) {
				r[0] = Math.min(1.0, 4.0 * rtemp / po);
			} else if (init1 == 3) {
				r[0] = Math.min(1.0, 2.0 * rtemp / po);
			} else if (init1 == 4) {
				r[0] = 2.0 * rtemp / po;
			}
			r[0] = Math.max(r[0], rmin);
			r[0] = Math.min(r[0], rmax);
			mode = 0;
			ru = 0.0;
			fu = fo;
			pu = po;
		}

		// NEW STEPSIZE SELECTION (EXTRAPOLATION OR INTERPOLATION)
		pnint1(rl, ru, fl, fu, pl, pu, r, mode, mtyp, merr);
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

		// COMPUTATION OF THE NEW FUNCTION VALUE AND THE NEW DIRECTIONAL
		// DERIVATIVE
		kd[0] = 1;
		ld[0] = -1;
		isys[0] = 1;
	}

	private static void pcbs04(final int nf, final double[] x, final int[] ix, final double[] xl, final double[] xu,
			final double eps9, final int kbf) {
		double temp;
		int i, ixi;
		if (kbf > 0) {
			for (i = 1; i <= nf; ++i) {
				temp = 1.0;
				ixi = IntMath.abs(ix[i - 1]);
				if ((ixi == 1 || ixi == 3 || ixi == 4)
						&& x[i - 1] <= xl[i - 1] + eps9 * Math.max(Math.abs(xl[i - 1]), temp)) {
					x[i - 1] = xl[i - 1];
				}
				if ((ixi == 2 || ixi == 3 || ixi == 4)
						&& x[i - 1] >= xu[i - 1] - eps9 * Math.max(Math.abs(xu[i - 1]), temp)) {
					x[i - 1] = xu[i - 1];
				}
			}
		}
	}

	private static void pyadc0(final int nf, final int[] n, final double[] x, final int[] ix, final double[] xl,
			final double[] xu, final int[] inew) {
		int i, ii, ixi;
		n[0] = nf;
		inew[0] = 0;
		for (i = 1; i <= nf; ++i) {
			ii = ix[i - 1];
			ixi = IntMath.abs(ii);
			if (ixi >= 5) {
				ix[i - 1] = -ixi;
			} else if ((ixi == 1 || ixi == 3 || ixi == 4) && x[i - 1] <= xl[i - 1]) {
				x[i - 1] = xl[i - 1];
				if (ixi == 4) {
					ix[i - 1] = -3;
				} else {
					ix[i - 1] = -ixi;
				}
				--n[0];
				if (ii > 0) {
					++inew[0];
				}
			} else if ((ixi == 2 || ixi == 3 || ixi == 4) && x[i - 1] >= xu[i - 1]) {
				x[i - 1] = xu[i - 1];
				if (ixi == 3) {
					ix[i - 1] = -4;
				} else {
					ix[i - 1] = -ixi;
				}
				--n[0];
				if (ii > 0) {
					++inew[0];
				}
			}
		}
	}

	private static void pytrcg(final int nf, final int[] n, final int[] ix, final double[] g, final double[] umax,
			final double[] gmax, final int kbf, final int[] iold) {
		double temp;
		int i;
		if (kbf > 0) {
			gmax[0] = umax[0] = 0.0;
			iold[0] = 0;
			for (i = 1; i <= nf; ++i) {
				temp = g[i - 1];
				if (ix[i - 1] >= 0) {
					gmax[0] = Math.max(gmax[0], Math.abs(temp));
				} else if (ix[i - 1] <= -5) {
				} else if ((ix[i - 1] == -1 || ix[i - 1] == -3) && umax[0] + temp >= 0.0) {
				} else if ((ix[i - 1] == -2 || ix[i - 1] == -4) && umax[0] - temp >= 0.0) {
				} else {
					iold[0] = i;
					umax[0] = Math.abs(temp);
				}
			}
		} else {
			umax[0] = 0.0;
			gmax[0] = mxvmax(nf, g);
		}
		n[0] = nf;
	}

	private static void pyfut1(final int n, final double f, final double[] f0, final double umax, final double gmax,
			final double dmax, final double tolx, final double tolf, final double tolb, final double tolg, final int kd,
			final int[] nit, final int kit, final int mit, final int nfv, final int mfv, final int nfg, final int mfg,
			final int[] ntesx, final int mtesx, final int[] ntesf, final int mtesf, final int ites, final int ires1,
			final int ires2, final int[] irest, final int iters, final int[] iterm) {
		double temp;
		if (iterm[0] < 0) {
			return;
		}
		if (ites > 0 && iters != 0) {
			if (nit[0] <= 0) {
				f0[0] = f + Math.min(Math.sqrt(Math.abs(f)), Math.abs(f) / 10.0);
			}
			if (f <= tolb) {
				iterm[0] = 3;
				return;
			}
			if (kd > 0) {
				if (gmax <= tolg && umax <= tolg) {
					iterm[0] = 4;
					return;
				}
			}
			if (nit[0] <= 0) {
				ntesx[0] = ntesf[0] = 0;
			}
			if (dmax <= tolx) {
				iterm[0] = 1;
				++ntesx[0];
				if (ntesx[0] >= mtesx) {
					return;
				}
			} else {
				ntesx[0] = 0;
			}
			temp = Math.abs(f0[0] - f) / Math.max(Math.abs(f), 1.0);
			if (temp <= tolf) {
				iterm[0] = 2;
				++ntesf[0];
				if (ntesf[0] >= mtesf) {
					return;
				}
			} else {
				ntesf[0] = 0;
			}
		}
		if (nit[0] >= mit) {
			iterm[0] = 11;
			return;
		}
		if (nfv >= mfv) {
			iterm[0] = 12;
			return;
		}
		if (nfg >= mfg) {
			iterm[0] = 13;
			return;
		}
		iterm[0] = 0;
		if (n > 0 && nit[0] - kit >= ires1 * n + ires2) {
			irest[0] = Math.max(irest[0], 1);
		}
		++nit[0];
	}

	private static void pyrmc0(final int nf, final int n, final int[] ix, final double[] g, final double eps8,
			final double umax, final double gmax, final double rmax, final int[] iold, final int[] irest) {
		int i, ixi;
		if (n == 0 || rmax > 0.0) {
			if (umax > eps8 * gmax) {
				iold[0] = 0;
				for (i = 1; i <= nf; ++i) {
					ixi = ix[i - 1];
					if (ixi >= 0) {
					} else if (ixi <= -5) {
					} else if ((ixi == -1 || ixi == -3) && -g[i - 1] <= 0.0) {
					} else if ((ixi == -2 || ixi == -4) && g[i - 1] <= 0.0) {
					} else {
						++iold[0];
						ix[i - 1] = Math.min(IntMath.abs(ix[i - 1]), 3);
						if (rmax == 0.0) {
							break;
						}
					}
				}
				if (iold[0] > 1) {
					irest[0] = Math.max(irest[0], 1);
				}
			}
		}
	}

	private static void pytrcs(final int nf, final double[] x, final int[] ix, final double[] xo, final double[] xl,
			final double[] xu, final double[] g, final double[] go, final double[] s, final double[] ro,
			final double[] fp, final double[] fo, final double f, final double[] po, final double p,
			final double[] rmax, final double eta9, final int kbf) {
		int i;
		fp[0] = fo[0];
		ro[0] = 0.0;
		fo[0] = f;
		po[0] = p;
		System.arraycopy(x, 0, xo, 0, nf);
		System.arraycopy(g, 0, go, 0, nf);
		if (kbf > 0) {
			for (i = 1; i <= nf; ++i) {
				if (ix[i - 1] < 0) {
					s[i - 1] = 0.0;
				} else {
					if (ix[i - 1] == 1 || ix[i - 1] >= 3) {
						if (s[i - 1] < -1.0 / eta9) {
							rmax[0] = Math.min(rmax[0], (xl[i - 1] - x[i - 1]) / s[i - 1]);
						}
					}
					if (ix[i - 1] == 2 || ix[i - 1] >= 3) {
						if (s[i - 1] > 1.0 / eta9) {
							rmax[0] = Math.min(rmax[0], (xu[i - 1] - x[i - 1]) / s[i - 1]);
						}
					}
				}
			}
		}
	}

	private static void pytrcd(final int nf, final double[] x, final int[] ix, final double[] xo, final double[] g,
			final double[] go, final double r, final double[] f, final double fo, final double[] p, final double[] po,
			final double[] dmax, final int kbf, final int kd, final int[] ld, final int iters) {
		int i;
		if (iters > 0) {
			mxvdif(nf, x, xo, xo);
			mxvdif(nf, g, go, go);
			po[0] *= r;
			p[0] *= r;
		} else {
			f[0] = fo;
			p[0] = po[0];
			mxvsav(nf, x, xo);
			mxvsav(nf, g, go);
			ld[0] = kd;
		}
		dmax[0] = 0.0;
		for (i = 1; i <= nf; ++i) {
			if (kbf > 0) {
				if (ix[i - 1] < 0) {
					xo[i - 1] = go[i - 1] = 0.0;
					continue;
				}
			}
			dmax[0] = Math.max(dmax[0], Math.abs(xo[i - 1]) / Math.max(Math.abs(x[i - 1]), 1.0));
		}
	}

	private static void pnint1(final double rl, final double ru, final double fl, final double fu, final double pl,
			final double pu, final double[] r, final int mode, final int mtyp, final int[] merr) {
		int ntyp;
		double a = 0.0, b = 0.0, c, d, dis, den = 0.0;
		double c1l = 1.1, c1u = 1.0e3, c2l = 1.0e-2, c2u = 0.9, c3l = 0.1;

		merr[0] = 0;
		if (mode <= 0) {
			return;
		}
		if (pl >= 0.0) {
			merr[0] = 2;
			return;
		} else if (ru <= rl) {
			merr[0] = 3;
			return;
		}
		for (ntyp = mtyp; ntyp >= 1; --ntyp) {
			if (ntyp == 1) {

				// BISECTION
				if (mode == 1) {
					r[0] = 4.0 * ru;
					return;
				} else {
					r[0] = 0.5 * (rl + ru);
					return;
				}
			} else if (ntyp == mtyp) {
				a = (fu - fl) / (pl * (ru - rl));
				b = pu / pl;
			}
			switch (ntyp) {
			case 2:
				// QUADRATIC EXTRAPOLATION OR INTERPOLATION WITH ONE DIRECTIONAL
				// DERIVATIVE
				den = 2.0 * (1.0 - a);
				break;
			case 3:
				// QUADRATIC EXTRAPOLATION OR INTERPOLATION WITH TWO DIRECTIONAL
				// DERIVATIVES
				den = 1.0 - b;
				break;
			case 4:
				// CUBIC EXTRAPOLATION OR INTERPOLATION
				c = b - 2.0 * a + 1.0;
				d = b - 3.0 * a + 2.0;
				dis = d * d - 3.0 * c;
				if (dis < 0.0) {
					continue;
				}
				den = d + Math.sqrt(dis);
				break;
			case 5:
				// CONIC EXTRAPOLATION OR INTERPOLATION
				dis = a * a - b;
				if (dis < 0.0) {
					continue;
				}
				den = a + Math.sqrt(dis);
				if (den <= 0.0) {
					continue;
				}
				den = 1.0 - b / den / den / den;
				break;
			default:
				break;
			}

			if (mode == 1 && den > 0.0 && den < 1.0) {

				// EXTRAPOLATION ACCEPTED
				r[0] = rl + (ru - rl) / den;
				r[0] = Math.max(r[0], c1l * ru);
				r[0] = Math.min(r[0], c1u * ru);
				return;
			} else if (mode == 2 && den > 1.0) {

				// INTERPOLATION ACCEPTED
				r[0] = rl + (ru - rl) / den;
				if (rl == 0.0) {
					r[0] = Math.max(r[0], rl + c2l * (ru - rl));
				} else {
					r[0] = Math.max(r[0], rl + c3l * (ru - rl));
				}
				r[0] = Math.min(r[0], rl + c2u * (ru - rl));
				return;
			}
		}
	}

	private static double mxvmax(final int n, final double[] x) {
		double mxvmax = 0.0;
		for (int i = 1; i <= n; ++i) {
			mxvmax = Math.max(mxvmax, Math.abs(x[i - 1]));
		}
		return mxvmax;
	}

	private static double mxudot(final int n, final double[] x, final int xi, final double[] y, final int yi,
			final int[] ix, final int job) {
		double temp = 0.0;
		int i;
		if (job == 0) {
			temp += BlasMath.ddotm(n, x, xi, y, yi);
		} else if (job > 0) {
			for (i = 1; i <= n; ++i) {
				if (ix[i - 1] >= 0) {
					temp += x[i - 1 + xi - 1] * y[i - 1 + yi - 1];
				}
			}
		} else {
			for (i = 1; i <= n; ++i) {
				if (ix[i - 1] != -5) {
					temp += x[i - 1 + xi - 1] * y[i - 1 + yi - 1];
				}
			}
		}
		return temp;
	}

	private static void mxudir(final int n, final double a, final double[] x, final double[] y, final double[] z,
			final int[] ix, final int job) {
		int i;
		if (job == 0) {
			BlasMath.daxpy1(n, a, x, 1, y, 1, z, 1);
		} else if (job > 0) {
			for (i = 1; i <= n; ++i) {
				if (ix[i - 1] >= 0) {
					z[i - 1] = y[i - 1] + a * x[i - 1];
				}
			}
		} else {
			for (i = 1; i <= n; ++i) {
				if (ix[i - 1] != -5) {
					z[i - 1] = y[i - 1] + a * x[i - 1];
				}
			}
		}
	}

	private static void mxuneg(final int n, final double[] x, final double[] y, final int[] ix, final int job) {
		int i;
		if (job == 0) {
			for (i = 1; i <= n; ++i) {
				y[i - 1] = -x[i - 1];
			}
		} else if (job > 0) {
			for (i = 1; i <= n; ++i) {
				if (ix[i - 1] >= 0) {
					y[i - 1] = -x[i - 1];
				} else {
					y[i - 1] = 0.0;
				}
			}
		} else {
			for (i = 1; i <= n; ++i) {
				if (ix[i - 1] != -5) {
					y[i - 1] = -x[i - 1];
				} else {
					y[i - 1] = 0.0;
				}
			}
		}
	}

	private static void mxdsrv(final int n, final int m, final double[] xm, final double[] gm, final double[] u,
			final double[] v, final double[] gr, final double[] xr, final double par) {
		double temp;
		int i, l;

		// STRANG RECURRENCES
		mxvscl(n, par, u, 1, v, 1);
		l = m * n;
		for (i = m; i >= 1; --i) {
			l -= n;
			temp = mxvdot(n, v, 1, xm, l + 1);
			gr[i - 1] = temp;
			mxvdir(n, -temp, gm, l + 1, v, v, 1);
		}
		for (i = 1; i <= m; ++i) {
			temp = xr[i - 1] * gr[i - 1] / par - mxvdot(n, v, 1, gm, l + 1);
			mxvdir(n, temp, xm, l + 1, v, v, 1);
			l += n;
		}
	}

	private static void mxvscl(final int n, final double a, final double[] x, final int ix, final double[] y,
			final int iy) {
		BlasMath.dscal1(n, a, x, ix, y, iy);
	}

	private static double mxvdot(final int n, final double[] x, final int ix, final double[] y, final int iy) {
		return BlasMath.ddotm(n, x, ix, y, iy);
	}

	private static void mxvdir(final int n, final double a, final double[] x, final int ix, final double[] y,
			final double[] z, final int iz) {
		BlasMath.daxpy1(n, a, x, ix, y, 1, z, iz);
	}

	private static void mxvdif(final int n, final double[] x, final double[] y, final double[] z) {
		for (int i = 1; i <= n; ++i) {
			z[i - 1] = x[i - 1] - y[i - 1];
		}
	}

	private static void mxvsav(final int n, final double[] x, final double[] y) {
		for (int i = 1; i <= n; ++i) {
			final double temp = y[i - 1];
			y[i - 1] = x[i - 1] - y[i - 1];
			x[i - 1] = temp;
		}
	}

	private static void mxvine(final int n, final int[] ix) {
		for (int i = 1; i <= n; ++i) {
			ix[i - 1] = IntMath.abs(ix[i - 1]);
		}
	}
}
