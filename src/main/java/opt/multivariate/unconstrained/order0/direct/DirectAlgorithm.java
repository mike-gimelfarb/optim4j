package opt.multivariate.unconstrained.order0.direct;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.unconstrained.order0.GradientFreeOptimizer;

/**
 *
 * @author Michael
 */
public final class DirectAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	private static final int DEF_MAXDEEP = 600;
	private static final int DEF_MAXDIV = 3000;
	private static final int DEF_MAXDIM = 64;

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int maxfunc, maxiters, maxdeep, maxdiv, maxdim, method;
	private final double[] lx, ux;
	private int jones;
	private double volper, sigmaper;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param volperParam
	 * @param sigmaperParam
	 * @param maxEvals
	 * @param maxIterations
	 * @param maxDepth
	 * @param maxDivs
	 * @param algorithmMethod
	 * @param lower
	 * @param upper
	 */
	public DirectAlgorithm(final double tolerance, final double volperParam, final double sigmaperParam,
			final int maxEvals, final int maxIterations, final int maxDepth, final int maxDivs,
			final int algorithmMethod, final double[] lower, final double[] upper) {
		super(tolerance);
		maxfunc = maxEvals + 21;
		maxiters = maxIterations;
		maxdeep = maxDepth;
		maxdiv = maxDivs;
		maxdim = DEF_MAXDIM;
		volper = volperParam;
		sigmaper = sigmaperParam;
		method = algorithmMethod;
		lx = lower;
		ux = upper;
	}

	/**
	 *
	 * @param tolerance
	 * @param volperParam
	 * @param sigmaperParam
	 * @param lower
	 * @param upper
	 */
	public DirectAlgorithm(final double tolerance, final double volperParam, final double sigmaperParam,
			final int maxEvals, final double[] lower, final double[] upper) {
		this(tolerance, volperParam, sigmaperParam, maxEvals, maxEvals, DEF_MAXDEEP, DEF_MAXDIV, 1, lower, upper);
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
		final double[] x = Arrays.copyOf(guess, n);
		final double[] eps = { myTol }, fmin = { func.apply(x) };
		final int[] maxf = { maxfunc - 21 }, maxT = { maxiters }, Ierror = new int[1];
		final double fglobal = -1.0e100, fglper = 0.0;

		// call main subroutine
		Direct(func, x, n, eps, maxf, maxT[0], fmin, lx, ux, method, Ierror, fglobal, fglper, volper, sigmaper);
		myEvals += maxf[0];
		return Ierror[0] > 0 ? x : null;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void Direct(final Function<? super double[], Double> fcn, final double[] x, final int n, final double[] eps,
			final int[] maxf, final int maxT, final double[] fmin, final double[] l, final double[] u,
			final int algmethod, final int[] Ierror, final double fglobal, final double fglper, final double volper,
			final double sigmaper) {

		final int[] iepschange = new int[1], free = new int[1], oops = new int[1], actdeep = new int[1],
				maxi = new int[1], minpos = new int[1], Ifeasiblef = new int[1], IInfeasiblef = new int[1],
				maxpos = new int[1], start = new int[1];
		final double[] epsfix = new double[1], fmax = new double[1];
		double divfactor, delta, kmax;
		int t, i, j, help, numfunc, cheat, tstart, newtosample, pos1, mdeep, oldmaxf, increase, freeold = 0,
				actdeep_div, actmaxdeep, oldpos;

		final double[][] f = new double[maxfunc][2], c = new double[maxfunc][maxdim];
		final double[] thirds = new double[maxdeep + 1], levels = new double[maxdeep + 1], w = new double[maxdim],
				oldl = new double[maxdim], oldu = new double[maxdim];
		final int[][] S = new int[maxdiv][2], length = new int[maxfunc][maxdim], list2 = new int[maxdim][2];
		final int[] anchor = new int[maxdeep + 2], point = new int[maxfunc], arrayI = new int[maxdim];

		// Start of code
		jones = algmethod;

		// Save the upper and lower bounds
		System.arraycopy(u, 0, oldu, 0, n);
		System.arraycopy(l, 0, oldl, 0, n);

		// Set parameters
		cheat = 0;
		kmax = 1.0e10;
		mdeep = maxdeep;

		// Write the header of the logfile
		DIRheader(n, eps, maxf[0], l, u, maxfunc, Ierror, epsfix, iepschange);

		// If an error has occured while writing the header
		if (Ierror[0] < 0) {
			return;
		}

		// If the known global minimum is equal 0, we cannot divide by it
		if (fglobal == 0.0) {
			divfactor = 1.0;
		} else {
			divfactor = Math.abs(fglobal);
		}

		// Start of application-specific initialisation
		// End of application-specific initialisation
		//
		// Save the budget given by the user
		oldmaxf = maxf[0];
		increase = 0;

		// Initialiase the lists
		DIRInitList(anchor, free, point, f, maxfunc, maxdeep);

		// Call the routine to initialise the mapping of x
		DIRpreprc(u, l, n, l, u, oops);
		if (oops[0] > 0) {
			Ierror[0] = -3;
			return;
		}
		// tstart = 2;

		// Initialise the algorithm DIRECT
		// Added variable to keep track of the maximum value found
		DIRInit(f, fcn, c, length, actdeep, point, anchor, free, arrayI, maxi, list2, w, x, l, u, fmin, minpos, thirds,
				levels, maxfunc, maxdeep, n, maxdim, fmax, Ifeasiblef, IInfeasiblef, Ierror);

		// Added error checking
		if (Ierror[0] < 0) {
			if (Ierror[0] == -4) {
				return;
			}
			if (Ierror[0] == -5) {
				return;
			}
		}
		numfunc = 1 + maxi[0] + maxi[0];
		actmaxdeep = 1;
		oldpos = 0;
		tstart = 2;

		// Main loop!
		for (t = tstart; t <= maxT; ++t) {

			// Choose the sample points
			actdeep[0] = actmaxdeep;
			DIRChoose(anchor, S, maxdeep, f, fmin[0], eps[0], levels, maxpos, length, maxdeep, n, cheat, kmax,
					Ifeasiblef[0]);

			// Add other hyperrectangles to S
			if (algmethod == 0) {
				DIRDoubleInsert(anchor, S, maxpos, point, f, maxdiv, Ierror);
				if (Ierror[0] == -6) {
					return;
				}
			}
			oldpos = minpos[0];

			// Initialise the number of sample points in this outer loop
			newtosample = 0;
			for (j = 1; j <= maxpos[0]; ++j) {
				actdeep[0] = S[j - 1][2 - 1];

				// If the actual index is a point to sample, do it
				if (S[j - 1][1 - 1] > 0) {

					// Calculate the value delta used for sampling points
					actdeep_div = DIRGetmaxDeep(S[j - 1][1 - 1], length, n);
					delta = thirds[actdeep_div + 1];
					actdeep[0] = S[j - 1][2 - 1];

					// If the current dept of division is only one
					// under the maximal allowed
					if (actdeep[0] + 1 >= mdeep) {
						Ierror[0] = -6;

						// Store the position of the minimum in x
						for (i = 1; i <= n; ++i) {
							x[i - 1] = c[minpos[0] - 1][i - 1] * l[i - 1] + l[i - 1] * u[i - 1];
						}
						System.arraycopy(oldu, 0, u, 0, n);
						System.arraycopy(oldl, 0, l, 0, n);

						// Store the number of function evaluations in maxf
						maxf[0] = numfunc;
						return;
					}
					actmaxdeep = Math.max(actdeep[0], actmaxdeep);
					help = S[j - 1][1 - 1];
					if (!(anchor[actdeep[0] + 1] == help)) {
						pos1 = anchor[actdeep[0] + 1];
						while (!(point[pos1 - 1] == help)) {
							pos1 = point[pos1 - 1];
						}
						point[pos1 - 1] = point[help - 1];
					} else {
						anchor[actdeep[0] + 1] = point[help - 1];
					}
					if (actdeep[0] < 0) {
						actdeep[0] = (int) f[help - 1][1 - 1];
					}

					// Get the Directions in which to decrease the interval-length
					DIRGet_I(length, help, arrayI, maxi, n);

					// Sample the function
					DIRSamplepoints(c, arrayI, delta, help, start, length, free, maxi[0], point, n, oops);
					if (oops[0] > 0) {
						Ierror[0] = -4;
						return;
					}
					newtosample += maxi[0];

					// Added variable to keep track of the maximum value found
					DIRSamplef(c, start[0], length, f, free[0], maxi[0], point, fcn, x, l, fmin, minpos, u, n, fmax,
							Ifeasiblef, IInfeasiblef);
					if (oops[0] > 0) {
						Ierror[0] = -5;
						return;
					}

					// Divide the intervalls
					DIRDivide(start[0], actdeep_div, length, point, arrayI, help, list2, w, maxi[0], f);

					// Insert the new intervalls into the list (sorted)
					DIRInsertList(start, anchor, point, f, maxi[0], length, maxfunc, n, help);

					// Increase the number of function evaluations
					numfunc += (maxi[0] + maxi[0]);
				}
			}
			// End of main loop

			// Termination Checks
			// Calculate the index for the hyperrectangle
			Ierror[0] = jones;
			jones = 0;
			actdeep_div = DIRGetlevel(minpos[0], length, n);
			jones = Ierror[0];

			// Use precalculated values to calculate volume
			delta = thirds[actdeep_div] * 100.0;
			if (delta <= volper) {
				Ierror[0] = 4;

				// Store the position of the minimum in x
				for (i = 1; i <= n; ++i) {
					x[i - 1] = c[minpos[0] - 1][i - 1] * l[i - 1] + l[i - 1] * u[i - 1];
				}
				System.arraycopy(oldu, 0, u, 0, n);
				System.arraycopy(oldl, 0, l, 0, n);

				// Store the number of function evaluations in maxf
				maxf[0] = numfunc;
				return;
			}

			// Calculate the measure for the hyperrectangle
			actdeep_div = DIRGetlevel(minpos[0], length, n);
			delta = levels[actdeep_div];
			if (delta <= sigmaper) {
				Ierror[0] = 5;

				// Store the position of the minimum in x
				for (i = 1; i <= n; ++i) {
					x[i - 1] = c[minpos[0] - 1][i - 1] * l[i - 1] + l[i - 1] * u[i - 1];
				}
				System.arraycopy(oldu, 0, u, 0, n);
				System.arraycopy(oldl, 0, l, 0, n);

				// Store the number of function evaluations in maxf
				maxf[0] = numfunc;
				return;
			}

			// If the best found function value is within fglper
			if ((100.0 * (fmin[0] - fglobal) / divfactor) <= fglper) {
				Ierror[0] = 3;

				// Store the position of the minimum in x
				for (i = 1; i <= n; ++i) {
					x[i - 1] = c[minpos[0] - 1][i - 1] * l[i - 1] + l[i - 1] * u[i - 1];
				}
				System.arraycopy(oldu, 0, u, 0, n);
				System.arraycopy(oldl, 0, l, 0, n);

				// Store the number of function evaluations in maxf
				maxf[0] = numfunc;
				return;
			}

			// Find out if there are infeasible points which are near feasible ones
			if (IInfeasiblef[0] > 0) {
				DIRreplaceInf(free[0], freeold, f, c, thirds, length, anchor, point, u, l, maxfunc, n, fmax[0]);
			}
			freeold = free[0];

			// If iepschange = 1, we use the epsilon change formula from Jones
			if (iepschange[0] == 1) {
				eps[0] = Math.max(1.0e-4 * Math.abs(fmin[0]), epsfix[0]);
			}

			// If no feasible point has been found yet
			if (increase == 1) {
				maxf[0] = numfunc + oldmaxf;
				if (Ifeasiblef[0] == 0) {
					increase = 0;
				}
			}

			// Check if the number of function evaluations done is larger
			if (numfunc > maxf[0]) {
				if (Ifeasiblef[0] == 0) {
					Ierror[0] = 1;

					// Store the position of the minimum in x
					for (i = 1; i <= n; ++i) {
						x[i - 1] = c[minpos[0] - 1][i - 1] * l[i - 1] + l[i - 1] * u[i - 1];
					}
					System.arraycopy(oldu, 0, u, 0, n);
					System.arraycopy(oldl, 0, l, 0, n);

					// Store the number of function evaluations in maxf
					maxf[0] = numfunc;
					return;
				} else {
					increase = 1;
					maxf[0] = numfunc + oldmaxf;
				}
			}
		}

		// The algorithm stopped after maxT iterations
		Ierror[0] = 2;

		// Store the position of the minimum in x
		for (i = 1; i <= n; ++i) {
			x[i - 1] = c[minpos[0] - 1][i - 1] * l[i - 1] + l[i - 1] * u[i - 1];
		}
		System.arraycopy(oldu, 0, u, 0, n);
		System.arraycopy(oldl, 0, l, 0, n);

		// Store the number of function evaluations in maxf
		maxf[0] = numfunc;
	}

	private static void DIRSamplef(final double[][] c, final int nnew, final int[][] length, final double[][] f,
			final int free, final int maxI, final int[] point, final Function<? super double[], Double> fcn,
			final double[] x, final double[] l, final double[] fmin, final int[] minpos, final double[] u, final int n,
			final double[] fmax, final int[] IFeasiblef, final int[] IInfeasiblef) {
		int j, kret = 0;
		int pos = nnew;
		int helppoint = pos;
		for (j = 1; j <= maxI + maxI; ++j) {
			System.arraycopy(c[pos - 1], 0, x, 0, n);
			final double[] farr = { f[pos - 1][1 - 1] };
			DIRinfcn(fcn, x, l, u, n, farr);
			f[pos - 1][1 - 1] = farr[0];
			IInfeasiblef[0] = Math.max(IInfeasiblef[0], kret);
			if (kret == 0) {
				f[pos - 1][2 - 1] = 0.0;
				IFeasiblef[0] = 0;
				fmax[0] = Math.max(f[pos - 1][1 - 1], fmax[0]);
			}
			if (kret >= 1) {
				f[pos - 1][2 - 1] = 2.0;
				f[pos - 1][1 - 1] = fmax[0];
			}
			if (kret == -1) {
				f[pos - 1][2 - 1] = -1.0;
			}
			pos = point[pos - 1];
		}
		pos = helppoint;
		for (j = 1; j <= maxI + maxI; ++j) {
			if ((f[pos - 1][1 - 1] < fmin[0]) && (f[pos - 1][2 - 1] == 0)) {
				fmin[0] = f[pos - 1][1 - 1];
				minpos[0] = pos;
			}
			pos = point[pos - 1];
		}
	}

	private void DIRChoose(final int[] anchor, final int[][] S, final int actdeep, final double[][] f,
			final double fmin, final double eps, final double[] thirds, final int[] maxpos, final int[][] length,
			final int maxdeep, final int n, final int cheat, final double kmax, final int Ifeasiblef) {
		double maxlower = 1.0e20;
		int i, j, k, i_, j_;
		double help2, helplower, helpgreater;
		int novalue, novaluedeep = 0;
		// helplower = maxlower;
		// helpgreater = 0.0;
		k = 1;
		if (Ifeasiblef >= 1) {
			for (j = 0; j <= actdeep; ++j) {
				if (anchor[j + 1] > 0) {
					S[k - 1][1 - 1] = anchor[j + 1];
					S[k - 1][2 - 1] = DIRGetlevel(S[k - 1][1 - 1], length, n);
					break;
				}
			}
			++k;
			maxpos[0] = 1;
			return;
		} else {
			for (j = 0; j <= actdeep; ++j) {
				if (anchor[j + 1] > 0) {
					S[k - 1][1 - 1] = anchor[j + 1];
					S[k - 1][2 - 1] = DIRGetlevel(S[k - 1][1 - 1], length, n);
					++k;
				}
			}
		}

		novalue = 0;
		if (anchor[0] > 0) {
			novalue = anchor[0];
			novaluedeep = DIRGetlevel(novalue, length, n);
		}
		maxpos[0] = k - 1;
		for (j = k - 1; j <= maxdeep; ++j) {
			S[k - 1][1 - 1] = 0;
		}

		for (j = maxpos[0]; j >= 1; --j) {
			helplower = maxlower;
			helpgreater = 0.0;
			j_ = S[j - 1][1 - 1];
			boolean breakflag = false;
			for (i = 1; i <= j - 1; ++i) {
				i_ = S[i - 1][1 - 1];
				if ((i_ > 0) && !(i == j)) {
					if (f[i_ - 1][2 - 1] <= 1.0) {
						help2 = thirds[S[i - 1][2 - 1]] - thirds[S[j - 1][2 - 1]];
						help2 = (f[i_ - 1][1 - 1] - f[j_ - 1][1 - 1]) / help2;
						if (help2 <= 0.0) {
							breakflag = true;
							break;
						}
						if (help2 < helplower) {
							helplower = help2;
						}
					}
				}
			}
			if (breakflag) {
				S[j - 1][1 - 1] = 0;
				continue;
			}
			for (i = j + 1; i <= maxpos[0]; ++i) {
				i_ = S[i - 1][1 - 1];
				if ((i_ > 0) && !(i == j)) {
					if (f[i_ - 1][2 - 1] <= 1.0) {
						help2 = thirds[S[i - 1][2 - 1]] - thirds[S[j - 1][2 - 1]];
						help2 = (f[i_ - 1][1 - 1] - f[j_ - 1][1 - 1]) / help2;
						if (help2 <= 0.0) {
							breakflag = true;
							break;
						}
						if (help2 > helpgreater) {
							helpgreater = help2;
						}
					}
				}
			}
			if (breakflag) {
				S[j - 1][1 - 1] = 0;
				continue;
			}
			if ((helplower > maxlower) && (helpgreater > 0)) {
				helplower = helpgreater;
				helpgreater -= 1.0;
			}
			if (helpgreater <= helplower) {
				if ((cheat == 1) && (helplower > kmax)) {
					helplower = kmax;
				}
				final double temp = f[j_ - 1][1 - 1] - helplower * thirds[S[j - 1][2 - 1]];
				if (temp > (fmin - eps * Math.abs(fmin))) {
					S[j - 1][1 - 1] = 0;
				}
			} else {
				S[j - 1][1 - 1] = 0;
			}
		}
		if (novalue > 0) {
			++maxpos[0];
			S[maxpos[0] - 1][1 - 1] = novalue;
			S[maxpos[0] - 1][2 - 1] = novaluedeep;
		}
	}

	private static int DIRGetmaxDeep(final int pos, final int[][] length, final int n) {
		int help = length[pos - 1][1 - 1];
		for (int i = 2; i <= n; ++i) {
			help = Math.min(help, length[pos - 1][i - 1]);
		}
		return help;
	}

	private int DIRGetlevel(final int pos, final int[][] length, final int n) {
		int help, i, p, k, DIRGetLevel;
		if (jones == 0) {
			help = length[pos - 1][1 - 1];
			k = help;
			p = 1;
			for (i = 2; i <= n; ++i) {
				if (length[pos - 1][i - 1] < k) {
					k = length[pos - 1][i - 1];
				}
				if (length[pos - 1][i - 1] == help) {
					++p;
				}
			}
			if (k == help) {
				DIRGetLevel = k * n + n - p;
			} else {
				DIRGetLevel = k * n + p;
			}
		} else {
			help = length[pos - 1][1 - 1];
			for (i = 2; i <= n; ++i) {
				if (length[pos - 1][i - 1] < help) {
					help = length[pos - 1][i - 1];
				}
			}
			DIRGetLevel = help;
		}
		return DIRGetLevel;
	}

	private static void DIRDoubleInsert(final int[] anchor, final int[][] S, final int[] maxpos, final int[] point,
			final double[][] f, final int maxdiv, final int[] ierror) {
		int iflag, i, pos, help, actdeep;
		int oldmaxpos = maxpos[0];
		for (i = 1; i <= oldmaxpos; ++i) {
			if (S[i - 1][1 - 1] > 0) {
				actdeep = S[i - 1][2 - 1];
				help = anchor[actdeep + 1];
				pos = point[help - 1];
				iflag = 0;
				while ((pos > 0) && (iflag == 0)) {
					if (f[pos - 1][1 - 1] - f[help - 1][1 - 1] <= 1.0e-13) {
						if (maxpos[0] < maxdiv) {
							++maxpos[0];
							S[maxpos[0] - 1][1 - 1] = pos;
							S[maxpos[0] - 1][2 - 1] = actdeep;
							pos = point[pos - 1];
						} else {
							ierror[0] = -6;
							return;
						}
					} else {
						iflag = 1;
					}
				}
			}
		}
	}

	private void DIRreplaceInf(final int free, final int freeold, final double[][] f, final double[][] c,
			final double[] thirds, final int[][] length, final int[] anchor, final int[] point, final double[] c1,
			final double[] c2, final int maxfunc, final int n, final double fmax) {
		int LMaxDim = 32;
		double sidelength;
		final double[] a = new double[LMaxDim], b = new double[LMaxDim], x = new double[LMaxDim];
		int i, j, k, l, help;
		for (i = 1; i <= free - 1; ++i) {
			if (f[i - 1][2 - 1] > 0) {
				help = DIRGetmaxDeep(i, length, n);
				// sidelength = thirds[help] * 2.0;
				for (j = 1; j <= n; ++j) {
					sidelength = thirds[length[i - 1][j - 1]];
					a[j - 1] = c[i - 1][j - 1] - sidelength;
					b[j - 1] = c[i - 1][j - 1] + sidelength;
				}
				f[i - 1][1 - 1] = 1.0e6;
				f[i - 1][2 - 1] = 2.0;
				for (k = 1; k <= free - 1; ++k) {
					if (f[k - 1][2 - 1] == 0) {
						System.arraycopy(c[k - 1], 0, x, 0, n);
						if (Isinbox(x, a, b, n) == 1) {
							f[i - 1][1 - 1] = Math.min(f[i - 1][1 - 1], f[k - 1][1 - 1]);
							f[i - 1][2 - 1] = 1.0;
						}
					}
				}
				if (f[i - 1][2 - 1] == 1.0) {
					f[i - 1][1 - 1] += (1.0e-6 * Math.abs(f[i - 1][1 - 1]));
					for (l = 1; l <= n; ++l) {
						x[l - 1] = c[i - 1][l - 1] * c1[l - 1] + c[i - 1][l - 1] * c2[l - 1];
					}
					DIRResortlist(i, anchor, f, point, length, n, maxfunc);
				} else if (!(fmax == f[i - 1][1 - 1])) {
					f[i - 1][1 - 1] = Math.max(fmax + 1.0, f[i - 1][1 - 1]);
				}
			}
		}
	}

	private void DIRResortlist(final int replace, final int[] anchor, final double[][] f, final int[] point,
			final int[][] length, final int n, final int maxfunc) {
		int start, l, i, pos;
		l = DIRGetlevel(replace, length, n);
		start = anchor[l + 1];
		if (replace != start) {
			pos = start;
			for (i = 1; i <= maxfunc; ++i) {
				if (point[pos - 1] == replace) {
					point[pos - 1] = point[replace - 1];
					break;
				} else {
					pos = point[pos - 1];
				}
				if (pos == 0) {
					break;
				}
			}
			if (f[start - 1][1 - 1] > f[replace - 1][1 - 1]) {
				anchor[l + 1] = replace;
				point[replace - 1] = start;
			} else {
				pos = start;
				for (i = 1; i <= maxfunc; ++i) {
					if (point[pos - 1] == 0) {
						point[replace - 1] = point[pos - 1];
						point[pos - 1] = replace;
						break;
					} else {
						if (f[point[pos - 1] - 1][1 - 1] > f[replace - 1][1 - 1]) {
							point[replace - 1] = point[pos - 1];
							point[pos - 1] = replace;
							break;
						}
						pos = point[pos - 1];
					}
				}
			}
		}
	}

	private void DIRInsertList(final int[] nnew, final int[] anchor, final int[] point, final double[][] f,
			final int maxI, final int[][] length, final int maxfunc, final int n, final int samp) {
		final int[] pos = new int[1];
		int j, pos1, pos2, deep;
		for (j = 1; j <= maxI; ++j) {
			pos1 = nnew[0];
			pos2 = point[pos1 - 1];
			nnew[0] = point[pos2 - 1];
			deep = DIRGetlevel(pos1, length, n);
			if (anchor[deep + 1] == 0) {
				if (f[pos2 - 1][1 - 1] < f[pos1 - 1][1 - 1]) {
					anchor[deep + 1] = pos2;
					point[pos2 - 1] = pos1;
					point[pos1 - 1] = 0;
				} else {
					anchor[deep + 1] = pos1;
					point[pos2 - 1] = 0;
				}
			} else {
				pos[0] = anchor[deep + 1];
				if (f[pos2 - 1][1 - 1] < f[pos1 - 1][1 - 1]) {
					if (f[pos2 - 1][1 - 1] < f[pos[0] - 1][1 - 1]) {
						anchor[deep + 1] = pos2;
						if (f[pos1 - 1][1 - 1] < f[pos[0] - 1][1 - 1]) {
							point[pos2 - 1] = pos1;
							point[pos1 - 1] = pos[0];
						} else {
							point[pos2 - 1] = pos[0];
							DIRInsert(pos, pos1, point, f, maxfunc);
						}
					} else {
						DIRInsert(pos, pos2, point, f, maxfunc);
						DIRInsert(pos, pos1, point, f, maxfunc);
					}
				} else if (f[pos1 - 1][1 - 1] < f[pos[0] - 1][1 - 1]) {
					anchor[deep + 1] = pos1;
					if (f[pos[0] - 1][1 - 1] < f[pos2 - 1][1 - 1]) {
						point[pos1 - 1] = pos[0];
						DIRInsert(pos, pos2, point, f, maxfunc);
					} else {
						point[pos1 - 1] = pos2;
						point[pos2 - 1] = pos[0];
					}
				} else {
					DIRInsert(pos, pos1, point, f, maxfunc);
					DIRInsert(pos, pos2, point, f, maxfunc);
				}
			}
		}

		deep = DIRGetlevel(samp, length, n);
		pos[0] = anchor[deep + 1];
		if (f[samp - 1][1 - 1] < f[pos[0] - 1][1 - 1]) {
			anchor[deep + 1] = samp;
			point[samp - 1] = pos[0];
		} else {
			DIRInsert(pos, samp, point, f, maxfunc);
		}
	}

	private static void DIRInsertList_2(final int[] start, final int j, final int k, final int[][] list2,
			final double[] w, final int maxI) {
		int pos = start[0];
		if (start[0] == 0) {
			list2[j - 1][1 - 1] = 0;
			start[0] = j;
			list2[j - 1][2 - 1] = k;
			return;
		}
		if (w[start[0] - 1] > w[j - 1]) {
			list2[j - 1][1 - 1] = start[0];
			start[0] = j;
		} else {
			for (int i = 1; i <= maxI; ++i) {
				if (list2[pos - 1][1 - 1] == 0) {
					list2[j - 1][1 - 1] = 0;
					list2[pos - 1][1 - 1] = j;
					list2[j - 1][2 - 1] = k;
					return;
				} else if (w[j - 1] < w[list2[pos - 1][1 - 1] - 1]) {
					list2[j - 1][1 - 1] = list2[pos - 1][1 - 1];
					list2[pos - 1][1 - 1] = j;
					list2[j - 1][2 - 1] = k;
					return;
				}
				pos = list2[pos - 1][1 - 1];
			}
		}
		list2[j - 1][2 - 1] = k;
	}

	private static void DIRSearchmin(final int[] start, final int[][] list2, final int[] pos, final int[] k) {
		k[0] = start[0];
		pos[0] = list2[start[0] - 1][2 - 1];
		start[0] = list2[start[0] - 1][1 - 1];
	}

	private void DIRInit(final double[][] f, final Function<? super double[], Double> fcn, final double[][] c,
			final int[][] length, final int[] actdeep, final int[] point, final int[] anchor, final int[] free,
			final int[] arrayI, final int[] maxI, final int[][] list2, final double[] w, final double[] x,
			final double[] l, final double[] u, final double[] fmin, final int[] minpos, final double[] thirds,
			final double[] levels, final int maxfunc, final int maxdeep, final int n, final int maxor,
			final double[] fmax, final int[] Ifeasiblef, final int[] IInfeasible, final int[] Ierror) {
		final int[] nnew = new int[1], oops = new int[1];
		int i, j, help = 0;
		double help2, delta;
		fmin[0] = 1.0e20;
		if (jones == 0) {
			for (j = 0; j <= n - 1; ++j) {
				w[j + 1 - 1] = 0.5 * Math.sqrt(n - j + j / 9.0);
			}
			help2 = 1.0;
			for (i = 1; i <= maxdeep / n; ++i) {
				for (j = 0; j <= n - 1; ++j) {
					levels[(i - 1) * n + j] = w[j + 1 - 1] / help2;
				}
				help2 *= 3.0;
			}
		} else {
			help2 = 3.0;
			for (i = 1; i <= maxdeep; ++i) {
				levels[i] = 1.0 / help2;
				help2 *= 3.0;
			}
			levels[0] = 1.0;
		}

		help2 = 3.0;
		for (i = 1; i <= maxdeep; ++i) {
			thirds[i] = 1.0 / help2;
			help2 *= 3.0;
		}
		thirds[0] = 1.0;
		for (i = 1; i <= n; ++i) {
			c[1 - 1][i - 1] = 0.5;
			x[i - 1] = 0.5;
			length[1 - 1][i - 1] = 0;
		}
		final double[] farr = { f[1 - 1][1 - 1] };
		DIRinfcn(fcn, x, l, u, n, farr);
		f[1 - 1][1 - 1] = farr[0];
		f[1 - 1][2 - 1] = help;
		IInfeasible[0] = help;
		fmax[0] = f[1 - 1][1 - 1];
		if (f[1 - 1][2 - 1] > 0.0) {
			f[1 - 1][1 - 1] = 1.0e6;
			fmax[0] = f[1 - 1][1 - 1];
			Ifeasiblef[0] = 1;
		} else {
			Ifeasiblef[0] = 0;
		}
		fmin[0] = f[1 - 1][1 - 1];
		minpos[0] = 1;
		actdeep[0] = 2;
		point[1 - 1] = 0;
		free[0] = 2;
		delta = thirds[1];
		DIRGet_I(length, 1, arrayI, maxI, n);
		nnew[0] = free[0];
		DIRSamplepoints(c, arrayI, delta, 1, nnew, length, free, maxI[0], point, n, oops);
		if (oops[0] > 0) {
			Ierror[0] = -4;
			return;
		}
		DIRSamplef(c, nnew[0], length, f, free[0], maxI[0], point, fcn, x, l, fmin, minpos, u, n, fmax, Ifeasiblef,
				IInfeasible);
		if (oops[0] > 0) {
			Ierror[0] = -5;
			return;
		}
		DIRDivide(nnew[0], 0, length, point, arrayI, 1, list2, w, maxI[0], f);
		DIRInsertList(nnew, anchor, point, f, maxI[0], length, maxfunc, n, 1);
	}

	private static void DIRDivide(final int nnew, final int currentlength, final int[][] length, final int[] point,
			final int[] arrayI, final int sample, final int[][] list2, final double[] w, final int maxI,
			final double[][] f) {
		final int[] start = { 0 }, pos = { nnew }, k = new int[1];
		int i, j, pos2;
		for (i = 1; i <= maxI; ++i) {
			j = arrayI[i - 1];
			w[j - 1] = f[pos[0] - 1][1 - 1];
			k[0] = pos[0];
			pos[0] = point[pos[0] - 1];
			w[j - 1] = Math.min(f[pos[0] - 1][1 - 1], w[j - 1]);
			pos[0] = point[pos[0] - 1];
			DIRInsertList_2(start, j, k[0], list2, w, maxI);
		}
		if (pos[0] > 0) {
			System.out.println("STOP");
			return;
		}
		for (j = 1; j <= maxI; ++j) {
			DIRSearchmin(start, list2, pos, k);
			pos2 = start[0];
			length[sample - 1][k[0] - 1] = currentlength + 1;
			for (i = 1; i <= maxI - j + 1; ++i) {
				length[pos[0] - 1][k[0] - 1] = currentlength + 1;
				pos[0] = point[pos[0] - 1];
				length[pos[0] - 1][k[0] - 1] = currentlength + 1;
				if (pos2 > 0) {
					pos[0] = list2[pos2 - 1][2 - 1];
					pos2 = list2[pos2 - 1][1 - 1];
				}
			}
		}
	}

	private static void DIRSamplepoints(final double[][] c, final int[] arrayI, final double delta, final int sample,
			final int[] start, final int[][] length, final int[] free, final int maxI, final int[] point, final int n,
			final int[] oops) {
		oops[0] = 0;
		int pos = free[0];
		start[0] = free[0];
		for (int k = 1; k <= maxI + maxI; ++k) {
			System.arraycopy(length[sample - 1], 0, length[free[0] - 1], 0, n);
			System.arraycopy(c[sample - 1], 0, c[free[0] - 1], 0, n);
			pos = free[0];
			free[0] = point[free[0] - 1];
			if (free[0] == 0) {
				oops[0] = 1;
				return;
			}
		}
		point[pos - 1] = 0;
		pos = start[0];
		for (int j = 1; j <= maxI; ++j) {
			c[pos - 1][arrayI[j - 1] - 1] = c[sample - 1][arrayI[j - 1] - 1] + delta;
			pos = point[pos - 1];
			c[pos - 1][arrayI[j - 1] - 1] = c[sample - 1][arrayI[j - 1] - 1] - delta;
			pos = point[pos - 1];
		}
		if (pos > 0) {
			System.out.println("STOP");
		}
	}

	private static void DIRGet_I(final int[][] length, final int pos, final int[] arrayI, final int[] maxi,
			final int n) {
		int i, help, j = 1;
		help = length[pos - 1][1 - 1];
		for (i = 2; i <= n; ++i) {
			if (length[pos - 1][i - 1] < help) {
				help = length[pos - 1][i - 1];
			}
		}
		for (i = 1; i <= n; ++i) {
			if (length[pos - 1][i - 1] == help) {
				arrayI[j - 1] = i;
				++j;
			}
		}
		maxi[0] = j - 1;
	}

	private static void DIRInitList(final int[] anchor, final int[] free, final int[] point, final double[][] f,
			final int maxfunc, final int maxdeep) {
		Arrays.fill(anchor, 0, maxdeep + 2, 0);
		for (int i = 1; i <= maxfunc; ++i) {
			f[i - 1][1 - 1] = f[i - 1][2 - 1] = 0.0;
			point[i - 1] = i + 1;
		}
		point[maxfunc - 1] = 0;
		free[0] = 1;
	}

	private static void DIRInsert(final int[] start, final int ins, final int[] point, final double[][] f,
			final int maxfunc) {
		for (int i = 1; i <= maxfunc; ++i) {
			if (point[start[0] - 1] == 0) {
				point[start[0] - 1] = ins;
				point[ins - 1] = 0;
				return;
			} else if (f[ins - 1][1 - 1] < f[point[start[0] - 1] - 1][1 - 1]) {
				final int help = point[start[0] - 1];
				point[start[0] - 1] = ins;
				point[ins - 1] = help;
				return;
			}
			start[0] = point[start[0] - 1];
		}
	}

	private static void DIRpreprc(final double[] u, final double[] l, final int n, final double[] xs1,
			final double[] xs2, final int[] oops) {
		oops[0] = 0;
		for (int i = 1; i <= n; ++i) {
			if (u[i - 1] <= l[i - 1]) {
				oops[0] = 1;
				return;
			}
		}
		for (int i = 1; i <= n; ++i) {
			final double help = (u[i - 1] - l[i - 1]);
			xs2[i - 1] = l[i - 1] / help;
			xs1[i - 1] = help;
		}
	}

	private static void DIRinfcn(final Function<? super double[], Double> fcn, final double[] x, final double[] c1,
			final double[] c2, final int n, final double[] f) {
		for (int i = 1; i <= n; ++i) {
			x[i - 1] = (x[i - 1] + c2[i - 1]) * c1[i - 1];
		}
		f[0] = fcn.apply(x);
		for (int i = 1; i <= n; ++i) {
			x[i - 1] = x[i - 1] / c1[i - 1] - c2[i - 1];
		}
	}

	private static void DIRheader(final int n, final double[] eps, final int maxf, final double[] l, final double[] u,
			final int maxfunc, final int[] Ierror, final double[] epsfix, final int[] iepschange) {
		Ierror[0] = 0;
		if (eps[0] < 0.0) {
			iepschange[0] = 1;
			epsfix[0] = -eps[0];
			eps[0] = -eps[0];
		} else {
			iepschange[0] = 0;
			epsfix[0] = 1.0e100;
		}
		for (int i = 1; i <= n; ++i) {
			if (u[i - 1] <= l[i - 1]) {
				Ierror[0] = -1;
			}
		}
		if ((maxf + 20) > maxfunc) {
			Ierror[0] = -2;
		}
	}

	private static int Isinbox(final double[] x, final double[] a, final double[] b, final int n) {
		for (int i = 1; i <= n; ++i) {
			if ((a[i - 1] > x[i - 1]) || (b[i - 1] < x[i - 1])) {
				return 0;
			}
		}
		return 1;
	}
}
