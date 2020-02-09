/*
SOLVOPT - SOLver for local OPTimization problems
Copyright (c) 1997, Alexei V. Kuntsevich (alex@bedvgm.kfunigraz.ac.at),
                    and Franz Kappel (franz.kappel@kfunigraz.ac.at)

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
package opt.multivariate.constrained.order1;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.unconstrained.order1.GradientOptimizer;
import utils.BlasMath;
import utils.RealMath;

/**
 * A translation of the SolvOpt program for minimization of a non-linear
 * function subject to general constraints.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Code from https://imsc.uni-graz.at/kuntsevich/solvopt/index.html
 */
public final class ShorAlgorithm extends GradientOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double myTolF, myDilation, myGradH;
	private final int myMaxEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param toleranceX
	 * @param toleranceF
	 * @param dilationCoeff
	 * @param minGradEstStepsize
	 * @param maxEvals
	 */
	public ShorAlgorithm(final double toleranceX, final double toleranceF, final double dilationCoeff,
			final double minGradEstStepsize, final int maxEvals) {
		super(toleranceX);
		myTolF = toleranceF;
		myDilation = dilationCoeff;
		myGradH = minGradEstStepsize;
		myMaxEvals = maxEvals;
	}

	/**
	 *
	 * @param toleranceX
	 * @param toleranceF
	 * @param maxEvals
	 */
	public ShorAlgorithm(final double toleranceX, final double toleranceF, final int maxEvals) {
		this(toleranceX, toleranceF, 2.5, 1.0e-11, maxEvals);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], Double> fun,
			final Function<? super double[], double[]> dfun, final double[] guess) {

		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		final double[] f = new double[1];
		final double[] options = new double[13];

		soptions(options);
		options[2 - 1] = myTol;
		options[3 - 1] = myTolF;
		options[4 - 1] = myMaxEvals;
		options[7 - 1] = myDilation;
		options[8 - 1] = myGradH;

		// call main subroutine
		solvopt(n, x, f, fun, true, dfun, options, false, null, false, null);
		myEvals += options[10 - 1];
		myGEvals += options[11 - 1];
		return options[9 - 1] > 0 ? x : null;
	}

	@Override
	public final double[] optimize(final Function<? super double[], Double> func, final double[] guess) {

		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);
		final double[] f = new double[1];
		final double[] options = new double[13];

		soptions(options);
		options[2 - 1] = myTol;
		options[3 - 1] = myTolF;
		options[4 - 1] = myMaxEvals;
		options[7 - 1] = myDilation;
		options[8 - 1] = myGradH;

		// call main subroutine
		solvopt(n, x, f, func, false, null, options, false, null, false, null);
		myEvals += options[10 - 1];
		return options[9 - 1] > 0 ? x : null;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static void solvopt(final int n, final double[] x, final double[] f,
			final Function<? super double[], Double> fun, final boolean flg,
			final Function<? super double[], double[]> grad, final double[] options, final boolean flfc,
			final Function<? super double[], Double> func, final boolean flgc,
			final Function<? super double[], double[]> gradc) {

		boolean constr, app, appconstr, fsbpnt = false, fsbpnt1 = false, termflag, stopf, stopping, dispwarn,
				reset = false, ksm, knan, obj;
		int kstore, ajp, ajpp, knorms, k, kcheck, numelem, dispdata, ld, mxtc, termx, limxterm, nzero, krerun, warnno,
				kflat, stepvanish, i, j, ni, ii, kd = 0, kj, kc, ip, iterlimit, kg, k1, k2, kless = 0;
		double kk, nx, ajb, ajs, des, dq, du20, du10, du03, n_float, cnteps = 0.0, low_bound, zerograd, ddx, y,
				lowxbound, lowfbound, detfr, detxr, grbnd, fp1 = 0.0, f1, f2, fopt, frec, fst, fp_rate, pencoef = 0.0,
				pencoefnew, gamma, w, wdef, h1, h, hp, dx, ng, ngc, nng, ngt, nrmz, ng1, d, dd, laststep, zero, one,
				two, three, four, five, six, seven, eight, nine, ten, hundr, infty, epsnorm, epsnorm2, powerm12;
		final double[] fp = new double[1], fc = new double[1], fm = new double[1];
		final double[] doptions = new double[13], nsteps = new double[3], gnorms = new double[10], g, g0, g1, gt, gc, z,
				x1, xopt, xrec, grec, xx, deltax, zeroes;
		final double[][] B;
		final int[] idx;

		// data
		zero = 0.0;
		one = 1.0;
		two = 2.0;
		three = 3.0;
		four = 4.0;
		five = 5.0;
		six = 6.0;
		seven = 7.0;
		eight = 8.0;
		nine = 9.0;
		ten = 10.0;
		hundr = 100.0;
		powerm12 = 1.0e-12;
		infty = 1.0e100;
		epsnorm = 1.0e-15;
		epsnorm2 = 1.0e-30;

		// Check the dimension
		if (n < 2) {
			options[9 - 1] = -one;
			return;
		}
		n_float = (double) n;

		// allocate working arrays
		B = new double[n][n];
		g = new double[n];
		g0 = new double[n];
		g1 = new double[n];
		gt = new double[n];
		gc = new double[n];
		z = new double[n];
		x1 = new double[n];
		xopt = new double[n];
		xrec = new double[n];
		grec = new double[n];
		xx = new double[n];
		deltax = new double[n];
		idx = new int[n];
		zeroes = new double[n];

		// store flags
		app = !flg;
		constr = flfc;
		appconstr = !flgc;

		// Default values for options
		soptions(doptions);
		for (i = 1; i <= 8; ++i) {
			if (options[i - 1] == zero) {
				options[i - 1] = doptions[i - 1];
			} else if (i == 2 || i == 3 || i == 6) {
				options[i - 1] = Math.max(options[i - 1], powerm12);
				options[i - 1] = Math.min(options[i - 1], one);
				if (i == 2) {
					options[i - 1] = Math.max(options[i - 1], options[8 - 1] * hundr);
				}
			} else if (i == 7) {
				options[7 - 1] = Math.max(options[i - 1], 1.5);
			}
		}

		// WORKING CONSTANTS AND COUNTERS
		options[10 - 1] = zero;
		options[11 - 1] = zero;
		options[12 - 1] = zero;
		options[13 - 1] = zero;
		iterlimit = (int) (options[4 - 1]);
		if (constr) {
			h1 = -one;
			cnteps = options[6 - 1];
		} else {
			h1 = RealMath.sign(one, options[1 - 1]);
		}
		k = 0;
		wdef = 1.0 / options[7 - 1] - one;

		// Gamma control
		ajb = one + 1.0e-1 / (n_float * n_float);
		ajp = 20;
		ajpp = ajp;
		ajs = 1.15;
		knorms = 0;
		Arrays.fill(gnorms, 0, 10, zero);

		// Display control
		if (options[5 - 1] <= zero) {
			dispdata = 0;
			dispwarn = options[5 - 1] != -one;
		} else {
			dispdata = (int) (options[5 - 1]);
			dispwarn = true;
		}
		ld = dispdata;

		// Stepsize control
		dq = 5.1;
		du20 = two;
		du10 = 1.5;
		du03 = 1.05;
		kstore = 3;
		Arrays.fill(nsteps, 0, kstore, zero);
		if (app) {
			des = 6.3;
		} else {
			des = 3.3;
		}
		mxtc = 3;
		termx = 0;
		limxterm = 50;

		// stepsize for gradient approximation
		ddx = Math.max(1.0e-11, options[8 - 1]);
		low_bound = -one + 1.0e-4;
		zerograd = n_float * 1.0e-16;
		nzero = 0;

		// ow bound for the values of variables to take into account
		lowxbound = Math.max(options[2 - 1], 1.0e-3);

		// Lower bound for function values to be considered as making difference
		lowfbound = options[3 - 1] * options[3 - 1];
		krerun = 0;
		detfr = options[3 - 1] * hundr;
		detxr = options[2 - 1] * ten;
		warnno = kflat = stepvanish = 0;
		stopf = false;

		// COMPUTE THE FUNCTION ( FIRST TIME )
		f[0] = fun.apply(x);
		options[10 - 1] += one;
		if (Math.abs(f[0]) >= infty) {
			options[9 - 1] = -three;
			return;
		}
		System.arraycopy(x, 0, xrec, 0, n);
		frec = f[0];

		// Constrained problem
		if (constr) {
			kless = 0;
			fp[0] = f[0];
			fc[0] = func.apply(x);
			options[12 - 1] += one;
			if (Math.abs(fc[0]) >= infty) {
				options[9 - 1] = -five;
				return;
			}
			pencoef = one;
			if (fc[0] <= cnteps) {
				fsbpnt = true;
				fc[0] = zero;
			} else {
				fsbpnt = true;
			}
			f[0] += (pencoef * fc[0]);
		}

		// COMPUTE THE GRADIENT ( FIRST TIME )
		if (app) {
			Arrays.fill(deltax, 0, n, h1 * ddx);
			obj = true;
			if (constr) {
				apprgrdn(n, g, x, fp, fun, deltax, obj);
			} else {
				apprgrdn(n, g, x, f, fun, deltax, obj);
			}
			options[10 - 1] += n_float;
		} else {
			System.arraycopy(grad.apply(x), 0, g, 0, n);
			options[11 - 1] += one;
		}
		ng = BlasMath.denorm(n, g);
		if (ng >= infty) {
			options[9 - 1] = -four;
			return;
		} else if (ng < zerograd) {
			options[9 - 1] = -four;
			return;
		}

		if (constr) {
			if (!fsbpnt) {
				if (appconstr) {
					for (j = 1; j <= n; ++j) {
						if (x[j - 1] >= zero) {
							deltax[j - 1] = ddx;
						} else {
							deltax[j - 1] = -ddx;
						}
					}
					obj = false;
					apprgrdn(n, gc, x, fc, func, deltax, obj);
				} else {
					System.arraycopy(gradc.apply(x), 0, gc, 0, n);
				}
				// ngc = NativeMath.denorm(n, gc);
				if (ng >= infty) {
					options[9 - 1] = -six;
					return;
				} else if (ng < zerograd) {
					options[9 - 1] = -six;
					return;
				}
				BlasMath.daxpym(n, pencoef, gc, 1, g, 1);
				System.arraycopy(g, 0, grec, 0, n);
				ng = BlasMath.denorm(n, g);
			}
		}
		System.arraycopy(g, 0, grec, 0, n);
		nng = ng;

		// INITIAL STEPSIZE
		d = zero;
		for (i = 1; i <= n; ++i) {
			if (d < Math.abs(x[i - 1])) {
				d = Math.abs(x[i - 1]);
			}
		}
		h = h1 * Math.sqrt(options[2 - 1]) * d;
		if (Math.abs(options[1 - 1]) != one) {
			h = h1 * Math.max(Math.abs(options[1 - 1]), Math.abs(h));
		} else {
			h = h1 * Math.max(one / Math.log(ng + 1.0), Math.abs(h));
		}

		// RESETTING LOOP
		while (true) {
			kcheck = kg = kj = 0;
			for (i = 1; i <= n; ++i) {
				System.arraycopy(zeroes, 0, B[i - 1], 0, n);
				B[i - 1][i - 1] = one;
			}
			System.arraycopy(g0, 0, g1, 0, n);
			fst = f[0];
			dx = 0.0;

			// MAIN ITERATIONS
			while (true) {
				++k;
				++kcheck;
				laststep = dx;

				// ADJUST GAMMA
				final double maxlog10 = Math.max(one, Math.log10(nng + one));
				gamma = one + Math.max(RealMath.pow(ajb, (ajp - kcheck) * n), two * options[3 - 1]);
				gamma = Math.min(gamma, Math.pow(ajs, maxlog10));

				ngt = ng1 = dd = zero;
				for (i = 1; i <= n; ++i) {
					d = zero;
					for (j = 1; j <= n; ++j) {
						d += (B[j - 1][i - 1] * g[j - 1]);
					}
					gt[i - 1] = d;
					dd += (d * g1[i - 1]);
					ngt += (d * d);
					ng1 += (g1[i - 1] * g1[i - 1]);
				}
				ngt = Math.sqrt(ngt);
				ng1 = Math.sqrt(ng1);
				dd = dd / ngt / ng1;
				w = wdef;

				// JUMPING OVER A RAVINE
				if (dd < low_bound) {
					if (kj == 2) {
						System.arraycopy(x, 0, xx, 0, n);
					}
					if (kj == 0) {
						kd = 4;
					}
					++kj;
					w = -0.9;
					h *= two;
					if (kj > 2 * kd) {
						++kd;
						warnno = 1;
					}
				} else {
					kj = 0;
				}

				// DILATION
				for (i = 1; i <= n; ++i) {
					z[i - 1] = gt[i - 1] - g1[i - 1];
				}
				nrmz = BlasMath.denorm(n, z);
				if (nrmz > epsnorm * ngt) {
					BlasMath.dscalm(n, 1.0 / nrmz, z, 1);

					// New direction in the transformed space: g1=gt+w*(z*gt')*z
					// new inverse matrix: B = B ( I + (1/alpha -1)zz' )
					d = BlasMath.ddotm(n, z, 1, gt, 1);
					d *= w;
					for (i = 1; i <= n; ++i) {
						g1[i - 1] = gt[i - 1] + d * z[i - 1];
						dd = BlasMath.ddotm(n, B[i - 1], 1, z, 1);
						dd *= w;
						BlasMath.daxpym(n, dd, z, 1, B[i - 1], 1);
					}
					ng1 = BlasMath.denorm(n, g1);
				} else {
					System.arraycopy(zeroes, 0, z, 0, n);
					System.arraycopy(gt, 0, g1, 0, n);
					nrmz = zero;
				}
				BlasMath.dscal1(n, 1.0 / ng1, g1, 1, gt, 1);
				for (i = 1; i <= n; ++i) {
					d = BlasMath.ddotm(n, B[i - 1], 1, gt, 1);
					g0[i - 1] = d;
				}

				// RESETTING
				if (kcheck > 1) {
					numelem = 0;
					for (i = 1; i <= n; ++i) {
						if (Math.abs(g[i - 1]) > zerograd) {
							++numelem;
							idx[numelem - 1] = i;
						}
					}
					if (numelem > 0) {
						grbnd = epsnorm * numelem * numelem;
						ii = 0;
						for (i = 1; i <= numelem; ++i) {
							j = idx[i - 1];
							if (Math.abs(g1[j - 1]) <= Math.abs(g[j - 1]) * grbnd) {
								++ii;
							}
						}
						if (ii == n || nrmz == zero) {
							if (Math.abs(fst - f[0]) < Math.abs(f[0]) * 1.0e-2) {
								ajp -= (10 * n);
							} else {
								ajp = ajpp;
							}
							h = h1 * dx / three;
							--k;
							break;
						}
					}
				}

				// STORE THE CURRENT VALUES AND SET THE COUNTERS FOR 1-D SEARCH
				System.arraycopy(x, 0, xopt, 0, n);
				fopt = f[0];
				k1 = k2 = 0;
				ksm = false;
				kc = 0;
				knan = false;
				hp = h;
				if (constr) {
					reset = false;
				}

				// 1-D SEARCH
				while (true) {
					System.arraycopy(x, 0, x1, 0, n);
					f1 = f[0];
					if (constr) {
						fsbpnt1 = fsbpnt;
						fp1 = fp[0];
					}

					// NEW POINT
					BlasMath.daxpym(n, hp, g0, 1, x, 1);
					ii = 0;
					for (i = 1; i <= n; ++i) {
						final double absx = Math.abs(x[i - 1]);
						if (Math.abs(x[i - 1] - x1[i - 1]) < absx * epsnorm) {
							++ii;
						}
					}

					// FUNCTION VALUE
					f[0] = fun.apply(x);
					options[10 - 1] += one;
					if (h1 * f[0] >= infty) {
						options[9 - 1] = -seven;
						return;
					}
					if (constr) {
						fp[0] = f[0];
						fc[0] = func.apply(x);
						options[12 - 1] += one;
						if (Math.abs(fc[0]) >= infty) {
							options[9 - 1] = -five;
							return;
						}
						if (fc[0] <= cnteps) {
							fsbpnt = true;
							fc[0] = zero;
						} else {
							fsbpnt = false;
							fp_rate = fp[0] - fp1;
							if (fp_rate < -epsnorm) {
								if (!fsbpnt1) {
									d = zero;
									for (i = 1; i <= n; ++i) {
										d += (x[i - 1] - x1[i - 1]) * (x[i - 1] - x1[i - 1]);
									}
									d = Math.sqrt(d);
									pencoefnew = -1.5e1 * fp_rate / d;
									if (pencoefnew > 1.2 * pencoef) {
										pencoef = pencoefnew;
										reset = true;
										kless = 0;
										f[0] += (pencoef * fc[0]);
										break;
									}
								}
							}
						}
						f[0] += (pencoef * fc[0]);
					}

					// No function value available:
					final int signf1 = (int) RealMath.sign(one, f1);
					if (Math.abs(f[0]) >= infty) {
						if (ksm || kc >= mxtc) {
							options[9 - 1] = -three;
							return;
						} else {
							++k2;
							k1 = 0;
							hp /= dq;
							System.arraycopy(x1, 0, x, 0, n);
							f[0] = f1;
							knan = true;
							if (constr) {
								fsbpnt = fsbpnt1;
								fp[0] = fp1;
							}
						}
					} else if (ii == n) {

						// STEP SIZE IS ZERO TO THE EXTENT OF EPSNORM
						++stepvanish;
						if (stepvanish >= 5) {
							options[9 - 1] = -ten - four;
							return;
						} else {
							System.arraycopy(x1, 0, x, 0, n);
							f[0] = f1;
							hp *= ten;
							ksm = true;
							if (constr) {
								fsbpnt = fsbpnt1;
								fp[0] = fp1;
							}
						}
					} else if (h1 * f[0] < h1 * f1 * RealMath.pow(gamma, signf1)) {

						// USE SMALLER STEP
						if (ksm) {
							break;
						}
						++k2;
						k1 = 0;
						hp /= dq;
						System.arraycopy(x1, 0, x, 0, n);
						f[0] = f1;
						if (constr) {
							fsbpnt = fsbpnt1;
							fp[0] = fp1;
						}
						if (kc >= mxtc) {
							break;
						}
					} else {

						// 1-D OPTIMIZER IS LEFT BEHIND
						if (h1 * f[0] <= h1 * f1) {
							break;
						}

						// USE LARGER STEP
						++k1;
						if (k2 > 0) {
							++kc;
						}
						k2 = 0;
						if (k1 >= 20) {
							hp *= du20;
						} else if (k1 >= 10) {
							hp *= du10;
						} else if (k1 >= 3) {
							hp *= du03;
						}
					}
				}
				// End of 1-D search

				// ADJUST THE TRIAL STEP SIZE
				dx = zero;
				for (i = 1; i <= n; ++i) {
					dx += (xopt[i - 1] - x[i - 1]) * (xopt[i - 1] - x[i - 1]);
				}
				dx = Math.sqrt(dx);
				if (kg < kstore) {
					++kg;
				}
				if (kg >= 2) {
					for (i = kg; i >= 2; --i) {
						nsteps[i - 1] = nsteps[i - 1 - 1];
					}
				}
				d = BlasMath.denorm(n, g0);
				nsteps[1 - 1] = dx / (Math.abs(h) * d);
				kk = d = zero;
				for (i = 1; i <= kg; ++i) {
					dd = (double) (kg - i + 1);
					d += dd;
					kk += (nsteps[i - 1] * dd);
				}
				kk /= d;
				if (kk > des) {
					if (kg == 1) {
						h *= (kk - des + one);
					} else {
						h *= Math.sqrt(kk - des + one);
					}
				} else if (kk < des) {
					h *= Math.sqrt(kk / des);
				}
				if (ksm) {
					++stepvanish;
				}

				// COMPUTE THE GRADIENT
				if (app) {
					for (j = 1; j <= n; ++j) {
						if (g0[j - 1] >= zero) {
							deltax[j - 1] = h1 * ddx;
						} else {
							deltax[j - 1] = -h1 * ddx;
						}
					}
					obj = true;
					if (constr) {
						apprgrdn(n, g, x, fp, fun, deltax, obj);
					} else {
						apprgrdn(n, g, x, f, fun, deltax, obj);
					}
					options[10 - 1] += n_float;
				} else {
					System.arraycopy(grad.apply(x), 0, g, 0, n);
					options[11 - 1] += one;
				}
				ng = BlasMath.denorm(n, g);
				if (ng >= infty) {
					options[9 - 1] = -four;
					return;
				} else if (ng < zerograd) {
					ng = zerograd;
				}

				// Constraints:
				if (constr) {
					if (!fsbpnt) {
						if (ng < 1.0e-2 * pencoef) {
							++kless;
							if (kless >= 20) {
								pencoef /= ten;
								reset = true;
								kless = 0;
							}
						} else {
							kless = 0;
						}
						if (appconstr) {
							for (j = 1; j <= n; ++j) {
								if (x[j - 1] >= zero) {
									deltax[j - 1] = ddx;
								} else {
									deltax[j - 1] = -ddx;
								}
							}
							obj = false;
							apprgrdn(n, gc, x, fc, func, deltax, obj);
							options[12 - 1] += n_float;
						} else {
							System.arraycopy(gradc.apply(x), 0, gc, 0, n);
							options[13 - 1] += one;
						}
						ngc = BlasMath.denorm(n, gc);
						if (ngc >= infty) {
							options[9 - 1] = -six;
							return;
						} else if (ngc < zerograd && !appconstr) {
							options[9 - 1] = -six;
							return;
						}
						BlasMath.daxpym(n, pencoef, gc, 1, g, 1);
						ng = BlasMath.denorm(n, g);
						if (reset) {
							h = h1 * dx / three;
							--k;
							nng = ng;
							break;
						}
					}
				}
				if (h1 * f[0] > h1 * frec) {
					frec = f[0];
					System.arraycopy(x, 0, xrec, 0, n);
					System.arraycopy(g, 0, grec, 0, n);
				}

				if (ng > zerograd) {
					if (knorms < 10) {
						++knorms;
					}
					if (knorms >= 2) {
						for (i = knorms; i >= 2; --i) {
							gnorms[i - 1] = gnorms[i - 1 - 1];
						}
					}
					gnorms[1 - 1] = ng;
					nng = one;
					for (i = 1; i <= knorms; ++i) {
						nng *= gnorms[i - 1];
					}
					nng = Math.pow(nng, one / knorms);
				}

				// Norm X
				nx = BlasMath.denorm(n, x);

				// DISPLAY THE CURRENT VALUES
				if (k == ld) {
					ld = k + dispdata;
				}

				// CHECK THE STOPPING CRITERIA
				termflag = true;
				if (constr) {
					if (!fsbpnt) {
						termflag = false;
					}
				}
				if (kcheck <= 5 || kcheck <= 12 && ng > one) {
					termflag = false;
				}
				if (kc >= mxtc || knan) {
					termflag = false;
				}

				// ARGUMENT
				if (termflag) {
					ii = 0;
					stopping = true;
					for (i = 1; i <= n; ++i) {
						if (Math.abs(x[i - 1]) >= lowxbound) {
							++ii;
							idx[ii - 1] = i;
							final double absx = options[2 - 1] * Math.abs(x[i - 1]);
							if (Math.abs(xopt[i - 1] - x[i - 1]) > absx) {
								stopping = false;
							}
						}
					}
					if (ii == 0 || stopping) {
						stopping = true;
						++termx;
						d = zero;
						for (i = 1; i <= n; ++i) {
							d += (x[i - 1] - xrec[i - 1]) * (x[i - 1] - xrec[i - 1]);
						}
						d = Math.sqrt(d);

						// FUNCTION
						if (Math.abs(f[0] - frec) > detfr * Math.abs(f[0])
								&& Math.abs(f[0] - fopt) <= options[3 - 1] * Math.abs(f[0]) && krerun <= 3 && !constr) {
							stopping = false;
							if (ii > 0) {
								for (i = 1; i <= ii; ++i) {
									j = idx[i - 1];
									final double absx = detxr * Math.abs(x[j - 1]);
									if (Math.abs(xrec[j - 1] - x[j - 1]) > absx) {
										stopping = true;
										break;
									}
								}
							}
							if (stopping) {
								System.arraycopy(xrec, 0, x, 0, n);
								System.arraycopy(grec, 0, g, 0, n);
								ng = BlasMath.denorm(n, g);
								f[0] = frec;
								++krerun;
								h = h1 * Math.max(dx, detxr * nx) / krerun;
								warnno = 2;
								break;
							} else {
								h *= ten;
							}
						} else if (Math.abs(f[0] - frec) > options[3 - 1] * Math.abs(f[0]) && d < options[2 - 1] * nx
								&& constr) {
							// continue;
						} else if (Math.abs(f[0] - fopt) <= options[3 - 1] * Math.abs(f[0])
								|| Math.abs(f[0]) <= lowfbound
								|| (Math.abs(f[0] - fopt) <= options[3 - 1] && termx >= limxterm)) {
							if (stopf) {
								if (dx <= laststep) {
									if (warnno == 1 && ng < Math.sqrt(options[3 - 1])) {
										warnno = 0;
									}
									if (!app) {
										for (i = 1; i <= n; ++i) {
											if (Math.abs(g[i - 1]) <= epsnorm2) {
												warnno = 3;
												break;
											}
										}
									}
									if (warnno != 0) {
										options[9 - 1] = -(double) warnno - ten;
									} else {
										options[9 - 1] = (double) k;
									}
									return;
								}
							} else {
								stopf = true;
							}
						} else if (dx < powerm12 * Math.max(nx, one) && termx >= limxterm) {
							options[9 - 1] = -four - ten;
							if (dispwarn) {
								f[0] = frec;
								System.arraycopy(xrec, 0, x, 0, n);
							}
							return;
						}
					}
				}

				// ITERATIONS LIMIT
				if (k == iterlimit) {
					options[9 - 1] = -nine;
					return;
				}

				// ZERO GRADIENT
				if (constr) {
					if (ng <= zerograd) {
						options[9 - 1] = -eight;
						return;
					}
				} else if (ng <= zerograd) {
					++nzero;
					if (nzero >= 3) {
						options[9 - 1] = -eight;
						return;
					}
					BlasMath.dscalm(n, -h / two, g0, 1);
					for (i = 1; i <= 10; ++i) {
						BlasMath.dxpym(n, g0, 1, x, 1);
						f[0] = fun.apply(x);
						options[10 - 1] += one;
						if (Math.abs(f[0]) >= infty) {
							options[9 - 1] = -three;
							return;
						}
						if (app) {
							for (j = 1; j <= n; ++j) {
								if (g0[j - 1] >= zero) {
									deltax[j - 1] = h1 * ddx;
								} else {
									deltax[j - 1] = -h1 * ddx;
								}
							}
							obj = true;
							apprgrdn(n, g, x, f, fun, deltax, obj);
							options[10 - 1] += n_float;
						} else {
							System.arraycopy(grad.apply(x), 0, g, 0, n);
							options[11 - 1] += one;
						}
						ng = BlasMath.denorm(n, g);
						if (ng >= infty) {
							options[9 - 1] = -four;
							return;
						}
						if (ng > zerograd) {
							break;
						}
					}
					if (ng <= zerograd) {
						options[9 - 1] = -eight;
						return;
					}
					h = h1 * dx;
					break;
				}

				// FUNCTION IS FLAT AT THE POINT
				if (!constr && Math.abs(f[0] - fopt) < Math.abs(fopt) * options[3 - 1] && kcheck > 5 && ng < one) {
					ni = 0;
					for (i = 1; i <= n; ++i) {
						if (Math.abs(g[i - 1]) <= epsnorm2) {
							++ni;
							idx[ni - 1] = i;
						}
					}
					if (ni >= 1 && ni <= n / 2 && kflat <= 3) {
						++kflat;
						warnno = 1;
						System.arraycopy(x, 0, x1, 0, n);
						fm[0] = f[0];
						for (i = 1; i <= ni; ++i) {
							j = idx[i - 1];
							f2 = fm[0];
							y = x[j - 1];
							if (y == zero) {
								x1[j - 1] = one;
							} else if (Math.abs(y) < one) {
								x1[j - 1] = RealMath.sign(one, y);
							} else {
								x1[j - 1] = y;
							}
							for (ip = 1; ip <= 20; ++ip) {
								x1[j - 1] /= 1.15;
								f1 = fun.apply(x1);
								options[10 - 1] += one;
								if (Math.abs(f1) < infty) {
									if (h1 * f1 > h1 * fm[0]) {
										y = x1[j - 1];
										fm[0] = f1;
									} else if (h1 * f2 > h1 * f1) {
										break;
									} else if (f2 == f1) {
										x1[j - 1] /= 1.5;
									}
									f2 = f1;
								}
							}
							x1[j - 1] = y;
						}
						if (h1 * fm[0] > h1 * f[0]) {
							if (app) {
								Arrays.fill(deltax, 0, n, h1 * ddx);
								obj = true;
								apprgrdn(n, gt, x1, fm, fun, deltax, obj);
								options[10 - 1] += n_float;
							} else {
								System.arraycopy(grad.apply(x1), 0, gt, 0, n);
								options[11 - 1] += one;
							}
							ngt = BlasMath.ddotm(n, gt, 1, gt, 1);
							if (ngt > epsnorm2 && ngt < infty) {
								System.arraycopy(x1, 0, x, 0, n);
								System.arraycopy(gt, 0, g, 0, n);
								ng = ngt;
								f[0] = fm[0];
								h = h1 * dx / three;
								options[3 - 1] /= five;
								break;
							}
						}
					}
				}
			}
		}
	}

	private static void soptions(final double[] def) {
		def[1 - 1] = -1.0;
		def[2 - 1] = 1.0e-4;
		def[3 - 1] = 1.0e-6;
		def[4 - 1] = 15e3;
		def[5 - 1] = 0.0;
		def[6 - 1] = 1.0e-8;
		def[7 - 1] = 2.5;
		def[8 - 1] = 1.0e-11;
		def[9 - 1] = def[10 - 1] = def[11 - 1] = def[12 - 1] = def[13 - 1] = 0.0;
	}

	private static void apprgrdn(final int n, final double[] g, final double[] x, final double[] f,
			final Function<? super double[], Double> fun, final double[] deltax, final boolean obj) {

		final double lowbndobj = 2.0e-10, lowbndcnt = 5.0e-15, one = 1.0, ten = 10.0, half = 0.5;
		double d, y, fi;
		int i, j;
		boolean center = false;

		for (i = 1; i <= n; ++i) {
			y = x[i - 1];
			d = Math.max(lowbndcnt, Math.abs(y));
			d *= deltax[i - 1];
			if (obj) {
				if (Math.abs(d) < lowbndobj) {
					d = lowbndobj * RealMath.sign(one, deltax[i - 1]);
					center = true;
				} else {
					center = false;
				}
			} else if (Math.abs(d) < lowbndcnt) {
				d = lowbndcnt * RealMath.sign(one, deltax[i - 1]);
			}
			x[i - 1] = y + d;
			fi = fun.apply(x);
			if (obj) {
				if (fi == f[0]) {
					for (j = 1; j <= 3; ++j) {
						d *= ten;
						x[i - 1] = y + d;
						fi = fun.apply(x);
						if (fi != f[0]) {
							return;
						}
					}
				}
			}
			g[i - 1] = (fi - f[0]) / d;
			if (obj) {
				if (center) {
					x[i - 1] = y - d;
					fi = fun.apply(x);
					g[i - 1] = half * (g[i - 1] + (f[0] - fi) / d);
				}
			}
			x[i - 1] = y;
		}
	}
}
