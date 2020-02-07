package opt.multivariate.unconstrained.order1;

import java.util.function.Function;

import utils.BlasMath;

/**
 *
 * @author Michael
 */
public final class TrustRegionNewtonAlgorithm extends GradientOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double myE1 = 0.25;
	private final double myE2 = 0.25;
	private final double myE3 = 0.75;
	private final double myT1 = 0.25;
	private final double myT2 = 2.0;
	private final double myDeltaM, myDelta0;
	private final int myMaxEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param minDelta
	 * @param maxDelta
	 * @param maxEvaluations
	 */
	public TrustRegionNewtonAlgorithm(final double tolerance, final double minDelta, final double maxDelta,
			final int maxEvaluations) {
		super(tolerance);
		myDelta0 = minDelta;
		myDeltaM = maxDelta;
		myMaxEvals = maxEvaluations;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final double[] optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] guess) {

		// prepare work arrays
		final int[] fev = new int[1];
		final int[] dfev = new int[1];

		// call main subroutine
		final double[] result = trust(f, df, null, guess.length, guess, 1, myDelta0, myDeltaM, myE1, myE2, myE3, myT1,
				myT2, myTol, fev, dfev, myMaxEvals);
		myEvals += fev[0];
		myGEvals += dfev[0];
		return result;
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param f
	 * @param df
	 * @param hess
	 * @param guess
	 * @return
	 */
	public final double[] optimize(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final Function<? super double[], double[][]> hess,
			final double[] guess) {

		// prepare work arrays
		final int[] fev = new int[1];
		final int[] dfev = new int[1];

		// call main subroutine
		final double[] result = trust(f, df, hess, guess.length, guess, 0, myDelta0, myDeltaM, myE1, myE2, myE3, myT1,
				myT2, myTol, fev, dfev, myMaxEvals);
		myEvals += fev[0];
		myGEvals += dfev[0];
		return result;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static double[] trust(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final Function<? super double[], double[][]> d2f,
			final int n, final double[] x0, final int mode, final double delta0, final double delmax, final double e1,
			final double e2, final double e3, final double t1, final double t2, final double tol, final int[] fev,
			final int[] dfev, final int maxfev) {

		// INITIALIZE POSITION, GRADIENT AND WORK ARRAYS
		final double[] x = x0;
		final double[] g = df.apply(x);
		final double[] d = new double[n];
		final double[] x1 = new double[n];
		final double[] b = new double[n];
		final double[] wa1 = new double[n];
		final double[] wa2 = new double[n];
		final double[] p = new double[n];
		double delta = delta0;
		double y = f.apply(x);
		fev[0] = dfev[0] = 1;

		// INITIALIZE THE HESSIAN
		double[][] B;
		if (mode == 0) {
			B = d2f.apply(x);
		} else if (mode == 1) {
			B = new double[n][n];
			for (int i = 0; i < n; ++i) {
				for (int j = i; j < n; ++j) {
					if (i == j) {
						B[i][j] = 1.0;
					} else {
						B[i][j] = B[j][i] = 0.0;
					}
				}
			}
		} else {
			return null;
		}

		// MAIN LOOP OF TRUST REGION METHOD STARTS HERE
		while (true) {

			// COMPUTE THE NEW POSITION AND STEP INFORMATION BY SOLVING THE
			// TRUST REGION SUBPROBLEM
			//
			// MIN_{||P||<=DELTA} F(X)+DF(X)*P+0.5*P*B(X)*P
			//
			// USING A CONJUGATE GRADIENT METHOD
			trcg(n, delta, g, p, b, B, wa1, wa2);
			for (int j = 0; j < n; ++j) {
				b[j] = BlasMath.ddotm(n, B[j], 1, p, 1);
				x1[j] = x[j] + p[j];
			}
			final double y1 = f.apply(x1);
			++fev[0];
			if (fev[0] > maxfev) {
				break;
			}

			// COMPUTE THE PREDICTION REDUCTION UNDER THE QUADRATIC MODEL,
			// THE ACTUAL REDUCTION OF FUNCTION VALUES, AND COMPUTE RHO,
			// THE RATIO OF THE PREDICTED TO ACTUAL REDUCTION.
			final double actred = y - y1;
			final double gp = BlasMath.ddotm(n, g, 1, p, 1);
			final double pbp = BlasMath.ddotm(n, p, 1, b, 1);
			final double prered = -gp - 0.5 * pbp;
			if (prered == 0.0 || prered == -0.0) {
				break;
			}
			final double rho = actred / prered;

			// USE THE REDUCTION RATIO TO UPDATE THE SIZE OF THE TRUST REGION
			// RADIUS FOR THE NEXT ITERATION.
			final double pnorm = BlasMath.denorm(n, p);
			if (rho < e2) {
				delta *= t1;
			} else if (rho > e3 && pnorm == delta) {
				delta = Math.min(t2 * delta, delmax);
			}

			// IF THE REDUCTION IS SUFFICIENT, WE ADVANCE THE POINT AND
			// RECOMPUTE THE MODEL HESSIAN
			if (rho > e1) {

				// ADVANCE THE POSITION, AND RECOMPUTE GRADIENT AND FUNCTION
				// VALUES AT THE NEW POINT. FURTHERMORE, PREPARE THE NECESSARY
				// QUANTITIES IN ORDER TO UPDATE THE HESSIAN AT THE NEW POINT.
				final double[] g1 = df.apply(x1);
				++dfev[0];
				for (int i = 0; i < n; ++i) {
					d[i] = g1[i] - g[i] - b[i];
				}
				System.arraycopy(x1, 0, x, 0, n);
				System.arraycopy(g1, 0, g, 0, n);
				y = y1;

				if (mode == 0) {

					// COMPUTE THE HESSIAN DIRECTLY FROM GIVEN FUNCTION
					B = d2f.apply(x);
				} else if (mode == 1) {

					// UPDATE THE HESSIAN USING A SYMMETRIC RANK-ONE UPDATE
					final double dp = BlasMath.ddotm(n, d, 1, p, 1);
					final double dnorm = BlasMath.denorm(n, d);
					if (Math.abs(dp) >= 1.0e-8 * pnorm * dnorm) {
						for (int i = 0; i < n; ++i) {
							for (int j = 0; j <= i; ++j) {
								final double correc = d[i] * d[j] / dp;
								B[i][j] += correc;
								B[j][i] = B[i][j];
							}
						}
					}
				}

				// PERFORM A TEST FOR CONVERGENCE HERE. THE TEST IS SATISFIED
				// WHEN
				// ||DF(X)|| <= MAX(1.0, ||X||)
				// HOLDS. ALSO LEAVE IF THE GRADIENT CANNOT BE COMPUTED
				final double ng = BlasMath.denorm(n, g1);
				if (ng != ng) {
					break;
				} else {
					final double nx = BlasMath.denorm(n, x1);
					if (ng <= tol * Math.max(1.0, nx)) {
						return x;
					}
				}
			}
		}
		return null;
	}

	// from pytron
	private static void trcg(final int n, final double delta, final double[] g, final double[] s, final double[] r,
			final double[][] H, final double[] wa1, final double[] wa2) {
		final double delsq = delta * delta;
		int i, j;
		double rtr, rtrnew, alpha, beta, cgtol;

		// INITIALIZE THE KEY QUANTITIES AND WORK ARRAYS
		for (i = 0; i < n; ++i) {
			s[i] = 0.0;
			r[i] = -g[i];
		}
		System.arraycopy(r, 0, wa1, 0, n);
		cgtol = 0.1 * BlasMath.denorm(n, g);
		rtr = BlasMath.ddotm(n, r, 1, r, 1);

		// MAIN LOOP OF THE CONJUGATE GRADIENT METHOD TO SOLVE THE
		// TRUST REGION SUBPROBLEM
		while (true) {

			// PERFORM A TEST FOR CONVERGENCE
			if (rtr <= cgtol * cgtol) {
				return;
			}

			// IF CONVERGENCE IS NOT SATISFIED, CONTINUE WITH THE METHOD
			for (j = 0; j < n; ++j) {
				wa2[j] = BlasMath.ddotm(n, H[j], 1, wa1, 1);
			}
			alpha = rtr / BlasMath.ddotm(n, wa1, 1, wa2, 1);
			BlasMath.daxpym(n, alpha, wa1, 1, s, 1);
			if (BlasMath.denorm(n, s) > delta) {
				alpha = -alpha;
				BlasMath.daxpym(n, alpha, wa1, 1, s, 1);
				final double std = BlasMath.ddotm(n, s, 1, wa1, 1);
				final double sts = BlasMath.ddotm(n, s, 1, s, 1);
				final double dtd = BlasMath.ddotm(n, wa1, 1, wa1, 1);
				final double rad = Math.sqrt(std * std + dtd * (delsq - sts));
				if (std >= 0.0) {
					alpha = (delsq - sts) / (std + rad);
				} else {
					alpha = (rad - std) / dtd;
				}
				BlasMath.daxpym(n, alpha, wa1, 1, s, 1);
				alpha = -alpha;
				BlasMath.daxpym(n, alpha, wa2, 1, r, 1);
				return;
			} else {
				alpha = -alpha;
				BlasMath.daxpym(n, alpha, wa2, 1, r, 1);
				rtrnew = BlasMath.ddotm(n, r, 1, r, 1);
				beta = rtrnew / rtr;
				BlasMath.dscalm(n, beta, wa1, 1);
				BlasMath.dxpym(n, r, 1, wa1, 1);
				rtr = rtrnew;
			}
		}
	}
}
