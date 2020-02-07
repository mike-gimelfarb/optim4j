package min4j.multivariate.unconstrained.leastsquares;

import java.util.Arrays;
import java.util.function.Function;

import min4j.utils.BlasMath;
import min4j.utils.RealMath;

/**
 *
 * @author Michael
 */
public final class LevenbergMarquardtAlgorithm extends LeastSquaresOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	@FunctionalInterface
	private interface Fcn {

		void fcn(int[] iflag, int m, int n, double[] x, double[] fvec, double[][] fjac, int ldfjac);
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int myMaxEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public LevenbergMarquardtAlgorithm(final double tolerance, final int maxEvaluations) {
		super(tolerance);
		myMaxEvals = maxEvaluations;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], double[]> func, final double[] guess) {

		// prepare variables
		final int[] info = new int[1];
		final int[] nfev = new int[1];
		final int[] njev = new int[1];

		// call main subroutine
		final double[] result = dnlse1(func, guess, myTol, myTol, myTol, myMaxEvals, 0.0, 100.0, info, nfev, njev);
		if (info[0] >= 1 && info[0] <= 4) {
			return result;
		} else {
			return null;
		}
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param func
	 * @param jacobian
	 * @param guess
	 * @return
	 */
	public final double[] optimize(final Function<double[], double[]> func,
			final Function<double[], double[][]> jacobian, final double[] guess) {

		// prepare variables
		final int[] info = new int[1];
		final int[] nfev = new int[1];
		final int[] njev = new int[1];

		// call main subroutine
		final double[] result = dnlse2(func, jacobian, guess, myTol, myTol, myTol, myMaxEvals, 0.0, 100.0, info, nfev,
				njev);
		if (info[0] >= 1 && info[0] <= 4) {
			return result;
		} else {
			return null;
		}
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double[] dnlse1(final Function<? super double[], double[]> func, final double[] x, final double ftol,
			final double xtol, final double gtol, final int maxfev, final double epsfcn, final double factor,
			final int[] info, final int[] nfev, final int[] njev) {

		// prepare fcn function
		final Fcn fcn = (iflag, m, n, x1, fvec, fjac, ldfjac) -> {
			System.arraycopy(func.apply(x1), 0, fvec, 0, m);
		};

		// prepare variables
		final double[] guess = Arrays.copyOf(x, x.length), fvec = func.apply(guess);
		final int m = fvec.length, n = x.length, iopt = 1, ldfjac = m, mode = 1;
		final double[][] fjac = new double[m][n];
		final double[] diag = new double[n];
		final int[] ipvt = new int[n];
		final double[] qtf = new double[n], wa1 = new double[n], wa2 = new double[n], wa3 = new double[n],
				wa4 = new double[m];

		// call main subroutine
		dnls1(fcn, iopt, m, n, guess, fvec, fjac, ldfjac, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor, info,
				nfev, njev, ipvt, qtf, wa1, wa2, wa3, wa4);
		return guess;
	}

	private static double[] dnlse2(final Function<? super double[], double[]> func,
			final Function<? super double[], double[][]> jac, final double[] x, final double ftol, final double xtol,
			final double gtol, final int maxfev, final double epsfcn, final double factor, final int[] info,
			final int[] nfev, final int[] njev) {

		// prepare fcn function
		final Fcn fcn = (iflag, m, n, x1, fvec, fjac, ldfjac) -> {
			System.arraycopy(func.apply(x1), 0, fvec, 0, m);
			final double[][] xjac = jac.apply(x1);
			for (int i = 0; i < m; ++i) {
				System.arraycopy(xjac[i], 0, fjac[i], 0, n);
			}
		};

		// prepare variables
		final double[] guess = Arrays.copyOf(x, x.length), fvec = func.apply(guess);
		final int m = fvec.length, n = x.length, iopt = 2, ldfjac = m, mode = 1;
		final double[][] fjac = new double[m][n];
		final double[] diag = new double[n];
		final int[] ipvt = new int[n];
		final double[] qtf = new double[n], wa1 = new double[n], wa2 = new double[n], wa3 = new double[n],
				wa4 = new double[m];

		// call main subroutine
		dnls1(fcn, iopt, m, n, guess, fvec, fjac, ldfjac, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor, info,
				nfev, njev, ipvt, qtf, wa1, wa2, wa3, wa4);
		return guess;
	}

	private static void dnls1(final Fcn fcn, final int iopt, final int m, final int n, final double[] x,
			final double[] fvec, final double[][] fjac, final int ldfjac, final double ftol, final double xtol,
			final double gtol, final int maxfev, final double epsfcn, final double[] diag, final int mode,
			final double factor, final int[] info, final int[] nfev, final int[] njev, final int[] ipvt,
			final double[] qtf, final double[] wa1, final double[] wa2, final double[] wa3, final double[] wa4) {

		boolean sing;
		int i, iter, j, l, modech, ijunk, nrow;
		double actred, chklim = 0.1, delta = 0.0, dirder, epsmch, fnorm, fnorm1, gnorm, one = 1.0, pnorm, prered,
				p1 = 0.1, p5 = 0.5, p25 = 0.25, p75 = 0.75, p0001 = 1.0e-4, ratio, sum, temp1, temp2, xnorm = 0.0,
				zero = 0.0;
		final int[] iflag = new int[1];
		final double[] err = new double[m], temp = new double[1], par = new double[1], fjaccol = new double[n],
				stmp1 = new double[1], stmp2 = new double[2];

		// FIRST EXECUTABLE STATEMENT DNLS1
		epsmch = BlasMath.D1MACH[4 - 1];
		info[0] = iflag[0] = nfev[0] = njev[0] = 0;

		// CHECK THE INPUT PARAMETERS FOR ERRORS
		if (iopt < 1 || iopt > 3 || n <= 0 || m < n || ldfjac < n || ftol < zero || xtol < zero || gtol < zero
				|| maxfev <= 0 || factor <= zero) {

			// TERMINATION, EITHER NORMAL OR USER IMPOSED
			if (iflag[0] < 0) {
				info[0] = iflag[0];
			}
			return;
		}
		if (iopt < 3 && ldfjac < m) {

			// TERMINATION, EITHER NORMAL OR USER IMPOSED
			if (iflag[0] < 0) {
				info[0] = iflag[0];
			}
			return;
		}
		if (mode == 2) {
			for (j = 1; j <= n; ++j) {
				if (diag[j - 1] <= zero) {

					// TERMINATION, EITHER NORMAL OR USER IMPOSED
					if (iflag[0] < 0) {
						info[0] = iflag[0];
					}
					return;
				}
			}
		}

		// EVALUATE THE FUNCTION AT THE STARTING POINT AND CALCULATE ITS NORM
		iflag[0] = ijunk = 1;
		fcn.fcn(iflag, m, n, x, fvec, fjac, ijunk);
		nfev[0] = 1;
		if (iflag[0] < 0) {

			// TERMINATION, EITHER NORMAL OR USER IMPOSED
			if (iflag[0] < 0) {
				info[0] = iflag[0];
			}
			return;
		}
		fnorm = BlasMath.denorm(m, fvec);

		// INITIALIZE LEVENBERG-MARQUARDT PARAMETER AND ITERATION COUNTER
		par[0] = zero;
		iter = 1;

		// BEGINNING OF THE OUTER LOOP
		while (true) {

			// CALCULATE THE JACOBIAN MATRIX
			if (iopt == 3) {

				// ACCUMULATE THE JACOBIAN BY ROWS IN ORDER TO SAVE STORAGE.
				// COMPUTE THE QR FACTORIZATION OF THE JACOBIAN MATRIX
				// CALCULATED ONE ROW AT A TIME, WHILE SIMULTANEOUSLY
				// FORMING (Q TRANSPOSE)*FVEC AND STORING THE FIRST
				// N COMPONENTS IN QTF
				Arrays.fill(qtf, 0, n, zero);
				for (j = 1; j <= n; ++j) {
					for (i = 1; i <= n; ++i) {
						fjac[i - 1][j - 1] = zero;
					}
				}
				for (i = 1; i <= m; ++i) {
					nrow = i;
					iflag[0] = 3;
					fcn.fcn(iflag, m, n, x, fvec, new double[][] { wa3 }, nrow);
					if (iflag[0] < 0) {

						// TERMINATION, EITHER NORMAL OR USER IMPOSED
						if (iflag[0] < 0) {
							info[0] = iflag[0];
						}
						return;
					}

					// ON THE FIRST ITERATION, CHECK THE USER SUPPLIED JACOBIAN
					if (iter <= 1) {

						// GET THE INCREMENTED X-VALUES INTO WA1(*)
						modech = 1;
						dckder(m, n, x, fvec, fjac, ldfjac, wa1, wa4, modech, err);

						// EVALUATE AT INCREMENTED VALUES, IF NOT ALREADY EVALUATED
						if (i == 1) {

							// EVALUATE FUNCTION AT INCREMENTED VALUE AND PUT INTO
							// WA4(*)
							iflag[0] = 1;
							fcn.fcn(iflag, m, n, wa1, wa4, fjac, nrow);
							++nfev[0];
							if (iflag[0] < 0) {

								// TERMINATION, EITHER NORMAL OR USER IMPOSED
								if (iflag[0] < 0) {
									info[0] = iflag[0];
								}
								return;
							}
						}

						// 495
						modech = 2;
						stmp1[0] = fvec[i - 1];
						stmp2[0] = wa4[i - 1];
						dckder(1, n, x, stmp1, new double[][] { wa3 }, 1, wa1, stmp2, modech, err);
						fvec[i - 1] = stmp1[0];
						wa4[i - 1] = stmp2[0];
						if (err[0] < chklim) {
							System.err.println("Derivative may be wrong.");
						}
					}
					temp[0] = fvec[i - 1];
					dwupdt(n, fjac, ldfjac, wa3, qtf, temp, wa1, wa2);
				}
				++njev[0];

				// IF THE JACOBIAN IS RANK DEFICIENT, CALL DQRFAC TO
				// REORDER ITS COLUMNS AND UPDATE THE COMPONENTS OF QTF
				sing = false;
				for (j = 1; j <= n; ++j) {
					if (fjac[j - 1][j - 1] == zero) {
						sing = true;
					}
					ipvt[j - 1] = j;
					for (int jj = 1; jj <= j; ++jj) {
						fjaccol[jj - 1] = fjac[jj - 1][j - 1];
					}
					wa2[j - 1] = BlasMath.denorm(j, fjaccol);
				}
				if (sing) {
					BlasMath.dqrfac(n, n, fjac, ldfjac, true, ipvt, n, wa1, wa2, wa3);
					for (j = 1; j <= n; ++j) {
						if (fjac[j - 1][j - 1] != zero) {
							sum = zero;
							for (i = j; i <= n; ++i) {
								sum += (fjac[i - 1][j - 1] * qtf[i - 1]);
							}
							temp[0] = -sum / fjac[j - 1][j - 1];
							for (i = j; i <= n; ++i) {
								qtf[i - 1] += (fjac[i - 1][j - 1] * temp[0]);
							}
						}
						fjac[j - 1][j - 1] = wa1[j - 1];
					}
				}
			} else {

				// STORE THE FULL JACOBIAN USING M*N STORAGE
				if (iopt == 1) {

					// THE CODE APPROXIMATES THE JACOBIAN
					iflag[0] = 1;
					dfdjc3(fcn, m, n, x, fvec, fjac, ldfjac, iflag, epsfcn, wa4);
					nfev[0] += n;
				} else {

					// THE USER SUPPLIES THE JACOBIAN
					iflag[0] = 2;
					fcn.fcn(iflag, m, n, x, fvec, fjac, ldfjac);
					++njev[0];

					// ON THE FIRST ITERATION, CHECK THE USER SUPPLIED JACOBIAN
					if (iter <= 1) {
						if (iflag[0] < 0) {

							// TERMINATION, EITHER NORMAL OR USER IMPOSED
							if (iflag[0] < 0) {
								info[0] = iflag[0];
							}
							return;
						}

						// GET THE INCREMENTED X-VALUES INTO WA1(*)
						modech = 1;
						dckder(m, n, x, fvec, fjac, ldfjac, wa1, wa4, modech, err);

						// EVALUATE FUNCTION AT INCREMENTED VALUE AND PUT IN WA4(*)
						iflag[0] = 1;
						fcn.fcn(iflag, m, n, wa1, wa4, fjac, ldfjac);
						++nfev[0];
						if (iflag[0] < 0) {

							// TERMINATION, EITHER NORMAL OR USER IMPOSED
							if (iflag[0] < 0) {
								info[0] = iflag[0];
							}
							return;
						}
						for (i = 1; i <= m; ++i) {
							modech = 2;
							stmp1[0] = fvec[i - 1];
							stmp2[0] = wa4[i - 1];
							dckder(1, n, x, stmp1, new double[][] { fjac[i - 1] }, ldfjac, wa1, stmp2, modech, err);
							fvec[i - 1] = stmp1[0];
							wa4[i - 1] = stmp2[0];
							if (err[0] < chklim) {
								System.err.println("Derivative may be wrong.");
							}
						}
					}
				}
				if (iflag[0] < 0) {

					// TERMINATION, EITHER NORMAL OR USER IMPOSED
					if (iflag[0] < 0) {
						info[0] = iflag[0];
					}
					return;
				}

				// COMPUTE THE QR FACTORIZATION OF THE JACOBIAN
				BlasMath.dqrfac(m, n, fjac, ldfjac, true, ipvt, n, wa1, wa2, wa3);

				// FORM (Q TRANSPOSE)*FVEC AND STORE THE FIRST N COMPONENTS IN QTF
				System.arraycopy(fvec, 0, wa4, 0, m);
				for (j = 1; j <= n; ++j) {
					if (fjac[j - 1][j - 1] != zero) {
						sum = zero;
						for (i = j; i <= m; ++i) {
							sum += (fjac[i - 1][j - 1] * wa4[i - 1]);
						}
						temp[0] = -sum / fjac[j - 1][j - 1];
						for (i = j; i <= m; ++i) {
							wa4[i - 1] += (fjac[i - 1][j - 1] * temp[0]);
						}
					}
					fjac[j - 1][j - 1] = wa1[j - 1];
					qtf[j - 1] = wa4[j - 1];
				}
			}

			// ON THE FIRST ITERATION AND IF MODE IS 1, SCALE ACCORDING
			// TO THE NORMS OF THE COLUMNS OF THE INITIAL JACOBIAN
			if (iter == 1) {
				if (mode != 2) {
					for (j = 1; j <= n; ++j) {
						diag[j - 1] = wa2[j - 1];
						if (wa2[j - 1] == zero) {
							diag[j - 1] = one;
						}
					}
				}

				// ON THE FIRST ITERATION, CALCULATE THE NORM OF THE SCALED X
				// AND INITIALIZE THE STEP BOUND DELTA
				for (j = 1; j <= n; ++j) {
					wa3[j - 1] = diag[j - 1] * x[j - 1];
				}
				xnorm = BlasMath.denorm(n, wa3);
				delta = factor * xnorm;
				if (delta == zero) {
					delta = factor;
				}
			}

			// COMPUTE THE NORM OF THE SCALED GRADIENT
			gnorm = zero;
			if (fnorm != zero) {
				for (j = 1; j <= n; ++j) {
					l = ipvt[j - 1];
					if (wa2[l - 1] != zero) {
						sum = zero;
						for (i = 1; i <= j; ++i) {
							sum += (fjac[i - 1][j - 1] * (qtf[i - 1] / fnorm));
						}
						gnorm = Math.max(gnorm, Math.abs(sum / wa2[l - 1]));
					}
				}
			}

			// TEST FOR CONVERGENCE OF THE GRADIENT NORM
			if (gnorm <= gtol) {
				info[0] = 4;
			}
			if (info[0] != 0) {

				// TERMINATION, EITHER NORMAL OR USER IMPOSED
				if (iflag[0] < 0) {
					info[0] = iflag[0];
				}
				return;
			}

			// RESCALE IF NECESSARY
			if (mode != 2) {
				for (j = 1; j <= n; ++j) {
					diag[j - 1] = Math.max(diag[j - 1], wa2[j - 1]);
				}
			}

			// BEGINNING OF THE INNER LOOP
			while (true) {

				// DETERMINE THE LEVENBERG-MARQUARDT PARAMETER
				dmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4);

				// STORE THE DIRECTION P AND X + P. CALCULATE THE NORM OF P
				for (j = 1; j <= n; ++j) {
					wa1[j - 1] = -wa1[j - 1];
					wa2[j - 1] = x[j - 1] + wa1[j - 1];
					wa3[j - 1] = diag[j - 1] * wa1[j - 1];
				}
				pnorm = BlasMath.denorm(n, wa3);

				// ON THE FIRST ITERATION, ADJUST THE INITIAL STEP BOUND
				if (iter == 1) {
					delta = Math.min(delta, pnorm);
				}

				// EVALUATE THE FUNCTION AT X + P AND CALCULATE ITS NORM
				iflag[0] = 1;
				fcn.fcn(iflag, m, n, wa2, wa4, fjac, ijunk);
				++nfev[0];
				if (iflag[0] < 0) {

					// TERMINATION, EITHER NORMAL OR USER IMPOSED
					if (iflag[0] < 0) {
						info[0] = iflag[0];
					}
					return;
				}
				fnorm1 = BlasMath.denorm(m, wa4);

				// COMPUTE THE SCALED ACTUAL REDUCTION
				actred = -one;
				if (p1 * fnorm1 < fnorm) {
					actred = one - (fnorm1 / fnorm) * (fnorm1 / fnorm);
				}

				// COMPUTE THE SCALED PREDICTED REDUCTION AND
				// THE SCALED DIRECTIONAL DERIVATIVE
				for (j = 1; j <= n; ++j) {
					wa3[j - 1] = zero;
					l = ipvt[j - 1];
					temp[0] = wa1[l - 1];
					for (i = 1; i <= j; ++i) {
						wa3[i - 1] += (fjac[i - 1][j - 1] * temp[0]);
					}
				}
				temp1 = BlasMath.denorm(n, wa3) / fnorm;
				temp2 = (Math.sqrt(par[0]) * pnorm) / fnorm;
				prered = temp1 * temp1 + temp2 * temp2 / p5;
				dirder = -(temp1 * temp1 + temp2 * temp2);

				// COMPUTE THE RATIO OF THE ACTUAL TO THE PREDICTED REDUCTION
				ratio = zero;
				if (prered != zero) {
					ratio = actred / prered;
				}

				// UPDATE THE STEP BOUND
				if (ratio > p25) {
					if (par[0] != zero && ratio < p75) {

					} else {
						delta = pnorm / p5;
						par[0] *= p5;
					}
				} else {
					if (actred >= zero) {
						temp[0] = p5;
					}
					if (actred < zero) {
						temp[0] = p5 * dirder / (dirder + p5 * actred);
					}
					if (p1 * fnorm1 >= fnorm || temp[0] < p1) {
						temp[0] = p1;
					}
					delta = temp[0] * Math.min(delta, pnorm / p1);
					par[0] /= temp[0];
				}

				// TEST FOR SUCCESSFUL ITERATION
				if (ratio >= p0001) {

					// SUCCESSFUL ITERATION. UPDATE X, FVEC, AND THEIR NORMS
					System.arraycopy(wa2, 0, x, 0, n);
					for (j = 1; j <= n; ++j) {
						wa2[j - 1] = diag[j - 1] * x[j - 1];
					}
					System.arraycopy(wa4, 0, fvec, 0, m);
					xnorm = BlasMath.denorm(n, wa2);
					fnorm = fnorm1;
					++iter;
				}

				// TESTS FOR CONVERGENCE
				if (Math.abs(actred) <= ftol && prered <= ftol && p5 * ratio <= one) {
					info[0] = 1;
				}
				if (delta <= xtol * xnorm) {
					info[0] = 2;
				}
				if (Math.abs(actred) <= ftol && prered <= ftol && p5 * ratio <= one && info[0] == 2) {
					info[0] = 3;
				}
				if (info[0] != 0) {

					// TERMINATION, EITHER NORMAL OR USER IMPOSED
					if (iflag[0] < 0) {
						info[0] = iflag[0];
					}
					return;
				}

				// TESTS FOR TERMINATION AND STRINGENT TOLERANCES
				if (nfev[0] >= maxfev) {
					info[0] = 5;
				}
				if (Math.abs(actred) <= epsmch && prered <= epsmch && p5 * ratio <= one) {
					info[0] = 6;
				}
				if (delta <= epsmch * xnorm) {
					info[0] = 7;
				}
				if (gnorm <= epsmch) {
					info[0] = 8;
				}
				if (info[0] != 0) {

					// TERMINATION, EITHER NORMAL OR USER IMPOSED
					if (iflag[0] < 0) {
						info[0] = iflag[0];
					}
					return;
				}

				// END OF THE INNER LOOP. REPEAT IF ITERATION UNSUCCESSFUL
				if (ratio >= p0001) {
					break;
				}
			}
		}
	}

	private static void dmpar(final int n, final double[][] r, final int ldr, final int[] ipvt, final double[] diag,
			final double[] qtb, final double delta, final double[] par, final double[] x, final double[] sigma,
			final double[] wa1, final double[] wa2) {

		int i, iter, j, jm1, jp1, k, l, nsing;
		double dxnorm, dwarf, fp, gnorm, parc, parl, paru, p1 = 0.1, p001 = 1.0e-3, sum, temp, zero = 0.0;

		// FIRST EXECUTABLE STATEMENT DMPAR
		dwarf = BlasMath.D1MACH[1 - 1];

		// COMPUTE AND STORE IN X THE GAUSS-NEWTON DIRECTION. IF THE
		// JACOBIAN IS RANK-DEFICIENT, OBTAIN A LEAST SQUARES SOLUTION
		nsing = n;
		for (j = 1; j <= n; ++j) {
			wa1[j - 1] = qtb[j - 1];
			if (r[j - 1][j - 1] == zero && nsing == n) {
				nsing = j - 1;
			}
			if (nsing < n) {
				wa1[j - 1] = zero;
			}
		}
		if (nsing >= 1) {
			for (k = 1; k <= nsing; ++k) {
				j = nsing - k + 1;
				wa1[j - 1] /= r[j - 1][j - 1];
				temp = wa1[j - 1];
				jm1 = j - 1;
				if (jm1 >= 1) {
					for (i = 1; i <= jm1; ++i) {
						wa1[i - 1] -= (r[i - 1][j - 1] * temp);
					}
				}
			}
		}
		for (j = 1; j <= n; ++j) {
			l = ipvt[j - 1];
			x[l - 1] = wa1[j - 1];
		}

		// INITIALIZE THE ITERATION COUNTER.
		// EVALUATE THE FUNCTION AT THE ORIGIN, AND TEST
		// FOR ACCEPTANCE OF THE GAUSS-NEWTON DIRECTION
		iter = 0;
		for (j = 1; j <= n; ++j) {
			wa2[j - 1] = diag[j - 1] * x[j - 1];
		}
		dxnorm = BlasMath.denorm(n, wa2);
		fp = dxnorm - delta;
		if (fp <= p1 * delta) {

			// TERMINATION
			if (iter == 0) {
				par[0] = zero;
			}
			return;
		}

		// IF THE JACOBIAN IS NOT RANK DEFICIENT, THE NEWTON
		// STEP PROVIDES A LOWER BOUND, PARL, FOR THE ZERO OF
		// THE FUNCTION. OTHERWISE SET THIS BOUND TO ZERO
		parl = zero;
		if (nsing >= n) {
			for (j = 1; j <= n; ++j) {
				l = ipvt[j - 1];
				wa1[j - 1] = diag[l - 1] * (wa2[l - 1] / dxnorm);
			}
			for (j = 1; j <= n; ++j) {
				sum = zero;
				jm1 = j - 1;
				if (jm1 >= 1) {
					for (i = 1; i <= jm1; ++i) {
						sum += (r[i - 1][j - 1] * wa1[i - 1]);
					}
				}
				wa1[j - 1] = (wa1[j - 1] - sum) / r[j - 1][j - 1];
			}
			temp = BlasMath.denorm(n, wa1);
			parl = ((fp / delta) / temp) / temp;
		}

		// CALCULATE AN UPPER BOUND, PARU, FOR THE ZERO OF THE FUNCTION
		for (j = 1; j <= n; ++j) {
			sum = zero;
			for (i = 1; i <= j; ++i) {
				sum += (r[i - 1][j - 1] * qtb[i - 1]);
			}
			l = ipvt[j - 1];
			wa1[j - 1] = sum / diag[l - 1];
		}
		gnorm = BlasMath.denorm(n, wa1);
		paru = gnorm / delta;
		if (paru == zero) {
			paru = dwarf / Math.min(delta, p1);
		}

		// IF THE INPUT PAR LIES OUTSIDE OF THE INTERVAL (PARL,PARU),
		// SET PAR TO THE CLOSER ENDPOINT
		par[0] = Math.max(par[0], parl);
		par[0] = Math.min(par[0], paru);
		if (par[0] == zero) {
			par[0] = gnorm / dxnorm;
		}

		while (true) {

			// BEGINNING OF AN ITERATION
			++iter;

			// EVALUATE THE FUNCTION AT THE CURRENT VALUE OF PAR
			if (par[0] == zero) {
				par[0] = Math.max(dwarf, p001 * paru);
			}
			temp = Math.sqrt(par[0]);
			BlasMath.dscal1(n, temp, diag, 1, wa1, 1);
			dqrslv(n, r, ldr, ipvt, wa1, qtb, x, sigma, wa2);
			for (j = 1; j <= n; ++j) {
				wa2[j - 1] = diag[j - 1] * x[j - 1];
			}
			dxnorm = BlasMath.denorm(n, wa2);
			temp = fp;
			fp = dxnorm - delta;

			// IF THE FUNCTION IS SMALL ENOUGH, ACCEPT THE CURRENT VALUE
			// OF PAR. ALSO TEST FOR THE EXCEPTIONAL CASES WHERE PARL
			// IS ZERO OR THE NUMBER OF ITERATIONS HAS REACHED 10
			if (Math.abs(fp) <= p1 * delta || parl == zero && fp <= temp && temp < zero || iter == 10) {

				// TERMINATION
				if (iter == 0) {
					par[0] = zero;
				}
				return;
			}

			// COMPUTE THE NEWTON CORRECTION
			for (j = 1; j <= n; ++j) {
				l = ipvt[j - 1];
				wa1[j - 1] = diag[l - 1] * (wa2[l - 1] / dxnorm);
			}
			for (j = 1; j <= n; ++j) {
				wa1[j - 1] /= sigma[j - 1];
				temp = wa1[j - 1];
				jp1 = j + 1;
				if (n >= jp1) {
					for (i = jp1; i <= n; ++i) {
						wa1[i - 1] -= (r[i - 1][j - 1] * temp);
					}
				}
			}
			temp = BlasMath.denorm(n, wa1);
			parc = ((fp / delta) / temp) / temp;

			// DEPENDING ON THE SIGN OF THE FUNCTION, UPDATE PARL OR PARU
			if (fp > zero) {
				parl = Math.max(parl, par[0]);
			} else if (fp < zero) {
				paru = Math.min(paru, par[0]);
			}

			// COMPUTE AN IMPROVED ESTIMATE FOR PAR
			par[0] = Math.max(parl, par[0] + parc);
		}
	}

	private static void dckder(final int m, final int n, final double[] x, final double[] fvec, final double[][] fjac,
			final int ldfjac, final double[] xp, final double[] fvecp, final int mode, final double[] err) {
		int i, j;
		double eps, epsf, epslog, epsmch, factor = 1.0e2, one = 1.0, temp, zero = 0.0;

		// EPSMCH IS THE MACHINE PRECISION
		epsmch = BlasMath.D1MACH[4 - 1];
		eps = Math.sqrt(epsmch);
		if (mode == 2) {

			// MODE = 2
			epsf = factor * epsmch;
			epslog = Math.log10(eps);
			Arrays.fill(err, 0, m, zero);
			for (j = 1; j <= n; ++j) {
				temp = Math.abs(x[j - 1]);
				if (temp == zero) {
					temp = one;
				}
				for (i = 1; i <= m; ++i) {
					err[i - 1] += (temp * fjac[i - 1][j - 1]);
				}
			}
			for (i = 1; i <= m; ++i) {
				temp = one;
				if (fvec[i - 1] != zero && fvecp[i - 1] != zero
						&& Math.abs(fvecp[i - 1] - fvec[i - 1]) >= epsf * Math.abs(fvec[i - 1])) {
					final double num = Math.abs((fvecp[i - 1] - fvec[i - 1]) / eps - err[i - 1]);
					final double den = Math.abs(fvec[i - 1]) + Math.abs(fvecp[i - 1]);
					temp = eps * num / den;
				}
				err[i - 1] = one;
				if (temp > epsmch && temp < eps) {
					err[i - 1] = (Math.log10(temp) - epslog) / epslog;
				}
				if (temp >= eps) {
					err[i - 1] = zero;
				}
			}
		} else {

			// MODE = 1
			for (j = 1; j <= n; ++j) {
				temp = eps * Math.abs(x[j - 1]);
				if (temp == zero) {
					temp = eps;
				}
				xp[j - 1] = x[j - 1] + temp;
			}
		}
	}

	private static void dfdjc3(final Fcn fcn, final int m, final int n, final double[] x, final double[] fvec,
			final double[][] fjac, final int ldfjac, final int[] iflag, final double epsfcn, final double[] wa) {
		int i, j;
		double eps, epsmch, h, temp, zero = 0.0;

		// FIRST EXECUTABLE STATEMENT DFDJC3
		epsmch = BlasMath.D1MACH[4 - 1];
		eps = Math.sqrt(Math.max(epsfcn, epsmch));

		// SET IFLAG=1 TO INDICATE THAT FUNCTION VALUES ARE TO BE RETURNED BY FCN
		iflag[0] = 1;
		for (j = 1; j <= n; ++j) {
			temp = x[j - 1];
			h = eps * Math.abs(temp);
			if (h == zero) {
				h = eps;
			}
			x[j - 1] = temp + h;
			fcn.fcn(iflag, m, n, x, wa, fjac, ldfjac);
			if (iflag[0] < 0) {
				break;
			}
			x[j - 1] = temp;
			for (i = 1; i <= m; ++i) {
				fjac[i - 1][j - 1] = (wa[i - 1] - fvec[i - 1]) / h;
			}
		}
	}

	private static void dwupdt(final int n, final double[][] r, final int ldr, final double[] w, final double[] b,
			final double[] alpha, final double[] cos, final double[] sin) {
		int i, j, jm1;
		double cotan, one = 1.0, rowj, tan, temp, zero = 0.0;

		// FIRST EXECUTABLE STATEMENT DWUPDT
		for (j = 1; j <= n; ++j) {
			rowj = w[j - 1];
			jm1 = j - 1;

			// APPLY THE PREVIOUS TRANSFORMATIONS TO
			// R(I,J), I=1,2,...,J-1, AND TO W(J)
			if (jm1 >= 1) {
				for (i = 1; i <= jm1; ++i) {
					temp = cos[i - 1] * r[i - 1][j - 1] + sin[i - 1] * rowj;
					rowj = -sin[i - 1] * r[i - 1][j - 1] + cos[i - 1] * rowj;
					r[i - 1][j - 1] = temp;
				}
			}

			// DETERMINE A GIVENS ROTATION WHICH ELIMINATES W(J)
			cos[j - 1] = one;
			sin[j - 1] = zero;
			if (rowj == zero) {
				continue;
			}
			if (Math.abs(r[j - 1][j - 1]) >= Math.abs(rowj)) {
				tan = rowj / r[j - 1][j - 1];
				cos[j - 1] = 1.0 / RealMath.hypot(1.0, tan);
				sin[j - 1] = cos[j - 1] * tan;
			} else {
				cotan = r[j - 1][j - 1] / rowj;
				sin[j - 1] = 1.0 / RealMath.hypot(1.0, cotan);
				cos[j - 1] = sin[j - 1] * cotan;
			}

			// APPLY THE CURRENT TRANSFORMATION TO R(J,J), B(J), AND ALPHA
			r[j - 1][j - 1] = cos[j - 1] * r[j - 1][j - 1] + sin[j - 1] * rowj;
			temp = cos[j - 1] * b[j - 1] + sin[j - 1] * alpha[0];
			alpha[0] = -sin[j - 1] * b[j - 1] + cos[j - 1] * alpha[0];
			b[j - 1] = temp;
		}
	}

	private static void dqrslv(final int n, final double[][] r, final int ldr, final int[] ipvt, final double[] diag,
			final double[] qtb, final double[] x, final double[] sigma, final double[] wa) {
		int i, j, jp1, k, kp1, l, nsing;
		double cos, cotan, qtbpj, sin, sum, tan, temp, zero = 0.0;

		// FIRST EXECUTABLE STATEMENT DQRSLV
		for (j = 1; j <= n; ++j) {
			for (i = j; i <= n; ++j) {
				r[i - 1][j - 1] = r[j - 1][i - 1];
			}
			x[j - 1] = r[j - 1][j - 1];
			wa[j - 1] = qtb[j - 1];
		}

		// ELIMINATE THE DIAGONAL MATRIX D USING A GIVENS ROTATION
		for (j = 1; j <= n; ++j) {

			// PREPARE THE ROW OF D TO BE ELIMINATED, LOCATING THE
			// DIAGONAL ELEMENT USING P FROM THE QR FACTORIZATION
			l = ipvt[j - 1];
			if (diag[l - 1] == zero) {

				// STORE THE DIAGONAL ELEMENT OF S AND RESTORE
				// THE CORRESPONDING DIAGONAL ELEMENT OF R
				sigma[j - 1] = r[j - 1][j - 1];
				r[j - 1][j - 1] = x[j - 1];
				continue;
			}
			Arrays.fill(sigma, j - 1, n, zero);
			sigma[j - 1] = diag[l - 1];

			// THE TRANSFORMATIONS TO ELIMINATE THE ROW OF D
			// MODIFY ONLY A SINGLE ELEMENT OF (Q TRANSPOSE)*B
			// BEYOND THE FIRST N, WHICH IS INITIALLY ZERO
			qtbpj = zero;
			for (k = j; k <= n; ++k) {

				// DETERMINE A GIVENS ROTATION WHICH ELIMINATES THE
				// APPROPRIATE ELEMENT IN THE CURRENT ROW OF D
				if (sigma[k - 1] == zero) {
					continue;
				}
				if (Math.abs(r[k - 1][k - 1]) >= Math.abs(sigma[k - 1])) {
					tan = sigma[k - 1] / r[k - 1][k - 1];
					cos = 1.0 / RealMath.hypot(1.0, tan);
					sin = cos * tan;
				} else {
					cotan = r[k - 1][k - 1] / sigma[k - 1];
					sin = 1.0 / RealMath.hypot(1.0, cotan);
					cos = sin * cotan;
				}

				// COMPUTE THE MODIFIED DIAGONAL ELEMENT OF R AND
				// THE MODIFIED ELEMENT OF ((Q TRANSPOSE)*B,0)
				r[k - 1][k - 1] = cos * r[k - 1][k - 1] + sin * sigma[k - 1];
				temp = cos * wa[k - 1] + sin * qtbpj;
				qtbpj = -sin * wa[k - 1] + cos * qtbpj;
				wa[k - 1] = temp;

				// ACCUMULATE THE TRANSFORMATION IN THE ROW OF S
				kp1 = k + 1;
				if (n >= kp1) {
					for (i = kp1; i <= n; ++i) {
						temp = cos * r[i - 1][k - 1] + sin * sigma[i - 1];
						sigma[i - 1] = -sin * r[i - 1][k - 1] + cos * sigma[i - 1];
						r[i - 1][k - 1] = temp;
					}
				}
			}

			// STORE THE DIAGONAL ELEMENT OF S AND RESTORE
			// THE CORRESPONDING DIAGONAL ELEMENT OF R
			sigma[j - 1] = r[j - 1][j - 1];
			r[j - 1][j - 1] = x[j - 1];
		}

		// SOLVE THE TRIANGULAR SYSTEM FOR Z. IF THE SYSTEM IS
		// SINGULAR, THEN OBTAIN A LEAST SQUARES SOLUTION
		nsing = n;
		for (j = 1; j <= n; ++j) {
			if (sigma[j - 1] == zero && nsing == n) {
				nsing = j - 1;
			}
			if (nsing < j) {
				wa[j - 1] = zero;
			}
		}

		if (nsing >= 1) {
			for (k = 1; k <= nsing; ++k) {
				j = nsing - k + 1;
				sum = zero;
				jp1 = j + 1;
				if (nsing >= jp1) {
					for (i = jp1; i <= nsing; ++i) {
						sum += (r[i - 1][j - 1] * wa[i - 1]);
					}
				}
				wa[j - 1] = (wa[j - 1] - sum) / sigma[j - 1];
			}
		}

		// PERMUTE THE COMPONENTS OF Z BACK TO COMPONENTS OF X
		for (j = 1; j <= n; ++j) {
			l = ipvt[j - 1];
			x[l - 1] = wa[j - 1];
		}
	}
}
