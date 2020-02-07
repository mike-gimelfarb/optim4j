package min4j.multivariate.unconstrained.order0.cmaes;

import java.util.Arrays;
import java.util.function.Function;

import min4j.utils.BlasMath;
import min4j.utils.RealMath;

/**
 *
 * @author Michael
 */
public class CmaesAlgorithm extends AbstractCmaesOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	/**
	 * 
	 * @author Michael
	 *
	 */
	public static class CmaesFactory implements AbstractCmaesFactory {

		@Override
		public CmaesAlgorithm createCmaStrategy(double tolerance, int populationSize, double initialSigma,
				int maxEvaluations) {
			return new CmaesAlgorithm(tolerance, populationSize, initialSigma, maxEvaluations);
		}
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// convergence parameters
	protected int myFlag;

	// other algorithm memory
	protected double updateEigenFrequency;
	protected int updateEigenLastEval;
	protected double[] diagD;
	protected double[][] B, C, invsqrtC;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public CmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations) {
		super(tolerance, populationSize, initialSigma, maxEvaluations);
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public CmaesAlgorithm(final double tolerance, final double initialSigma, final int maxEvaluations) {
		super(tolerance, initialSigma, maxEvaluations);
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 */
	public CmaesAlgorithm(final double tolerance, final double initialSigma) {
		super(tolerance, initialSigma);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
		super.initialize(func, guess);

		// we perform an eigenvalue decomposition every O(d) iterations
		updateEigenFrequency = (1.0 / (c1 + cmu)) / (10.0 * D);
		updateEigenLastEval = 0;

		// Initialize dynamic (internal) strategy parameters and constants
		diagD = new double[D];
		B = new double[D][D];
		C = new double[D][D];
		invsqrtC = new double[D][D];
		for (int d = 0; d < D; ++d) {
			diagD[d] = 1.0;
			C[d][d] = invsqrtC[d][d] = B[d][d] = 1.0;
		}

		// Initialize convergence parameters
		myFlag = 0;
	}

	@Override
	public void updateDistribution() {

		// compute weighted mean into xmean
		System.arraycopy(xmean, 0, xold, 0, D);
		for (int i = 0; i < D; ++i) {
			double sum = 0.0;
			for (int n = 0; n < myMu; ++n) {
				final int j = arfitness[n].index;
				sum += weights[n] * arx[j][i];
			}
			xmean[i] = sum;
		}

		// Cumulation: Update evolution paths
		final double csc = Math.sqrt(cs * (2.0 - cs) * mueff);
		for (int i = 0; i < D; ++i) {
			ps[i] *= (1.0 - cs);
			for (int j = 0; j < D; ++j) {
				ps[i] += csc * invsqrtC[i][j] * (xmean[j] - xold[j]) / sigma;
			}
		}

		// compute hsig
		final double pslen = BlasMath.denorm(D, ps);
		final double denom = 1.0 - Math.pow(1.0 - cs, 2.0 * myEvals / myLambda);
		final int hsig;
		if (pslen / Math.sqrt(denom) / chi < 1.4 + 2.0 / (D + 1.0)) {
			hsig = 1;
		} else {
			hsig = 0;
		}

		// update pc
		final double ccc = Math.sqrt(cc * (2.0 - cc) * mueff);
		for (int i = 0; i < D; ++i) {
			pc[i] = (1.0 - cc) * pc[i] + hsig * ccc * (xmean[i] - xold[i]) / sigma;
		}

		// Adapt covariance matrix C
		final double c2 = (1.0 - hsig) * cc * (2.0 - cc);
		for (int i = 0; i < D; ++i) {
			for (int j = 0; j <= i; ++j) {

				// old matrix plus rank-one update
				double sum = (1.0 - c1 - cmu) * C[i][j] + c1 * (pc[i] * pc[j] + c2 * C[i][j]);

				// rank mu update
				for (int k = 0; k < myMu; ++k) {
					final int m = arfitness[k].index;
					final double di = (arx[m][i] - xold[i]) / sigma;
					final double dj = (arx[m][j] - xold[j]) / sigma;
					sum += cmu * weights[k] * di * dj;
				}
				C[i][j] = sum;
			}
		}

		// update sigma parameters
		updateSigma();

		// Decomposition of C into B*diag(D.^2)*B' (diagonalization)
		eigenDecomposition();
	}

	@Override
	public void samplePopulation() {
		for (int n = 0; n < myLambda; ++n) {
			for (int i = 0; i < D; ++i) {
				artmp[i] = diagD[i] * RAND.nextGaussian();
			}
			for (int i = 0; i < D; ++i) {
				final double sum = BlasMath.ddotm(D, B[i], 1, artmp, 1);
				arx[n][i] = xmean[i] + sigma * sum;
			}
		}
	}

	@Override
	public boolean converged() {

		// MaxIter
		if (myIteration >= myMaxIters) {
			myFlag = 1;
			return true;
		}

		// TolHistFun
		if (myIteration >= myHistoryLength && myHistoryWorstFit - myHistoryBestFit < myTol) {
			myFlag = 2;
			return true;
		}

		// EqualFunVals
		if (myHistoryBest.length >= D && myHistoryKth.length >= D) {
			int countEq = 0;
			for (int i = 0; i < D; ++i) {
				if (myHistoryBest.get(i) == myHistoryKth.get(i)) {
					++countEq;
					if (3 * countEq >= D) {
						myFlag = 3;
						return true;
					}
				}
			}
		}

		// TolX
		boolean converged = true;
		for (int i = 0; i < D; ++i) {
			if (Math.max(pc[i], Math.sqrt(C[i][i])) * sigma / mySigma0 >= myTol) {
				converged = false;
				break;
			}
		}
		if (converged) {
			myFlag = 4;
			return true;
		}

		// TolUpSigma
		if (sigma / mySigma0 > 1.0e20 * diagD[D - 1]) {
			myFlag = 5;
			return true;
		}

		// ConditionCov
		if (diagD[D - 1] > 1.0e7 * diagD[0]) {
			myFlag = 7;
			return true;
		}

		// NoEffectAxis
		final int iaxis = D - 1 - ((myIteration - 1) % D);
		converged = true;
		for (int i = 0; i < D; ++i) {
			if (xmean[i] != xmean[i] + 0.1 * sigma * diagD[iaxis] * B[iaxis][i]) {
				converged = false;
				break;
			}
		}
		if (converged) {
			myFlag = 8;
			return true;
		}

		// NoEffectCoor
		for (int i = 0; i < D; ++i) {
			if (xmean[i] == xmean[i] + 0.2 * sigma * Math.sqrt(C[i][i])) {
				myFlag = 9;
				return true;
			}
		}
		return false;
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @return
	 */
	public final int convergenceFlag() {
		return myFlag;
	}

	/**
	 * 
	 * @return
	 */
	public void eigenDecomposition() {

		// skip the eigenvalue-decomposition O(D^3) until condition is reached
		// this is done once every O(D) iterations making the algorithm O(D^2)
		if (myEvals - updateEigenLastEval <= myLambda * updateEigenFrequency) {
			return;
		}

		// enforce symmetry
		for (int i = 0; i < D; ++i) {
			for (int j = 0; j <= i; ++j) {
				B[i][j] = B[j][i] = C[i][j];
			}
		}

		// eigenvalue decomposition, B==normalized eigenvectors
		updateEigenLastEval = myEvals;
		tred2(D, B, diagD, artmp);
		tql2(D, diagD, artmp, B);

		// limit condition number of covariance matrix
		if (diagD[0] <= 0.0) {
			for (int i = 0; i < D; ++i) {
				diagD[i] = Math.max(diagD[i], 0.0);
			}
			final double shift = diagD[D - 1] / 1e14;
			for (int i = 0; i < D; ++i) {
				C[i][i] += shift;
				diagD[i] += shift;
			}
		}
		if (diagD[D - 1] > 1e14 * diagD[0]) {
			final double shift = diagD[D - 1] / 1e14 - diagD[0];
			for (int i = 0; i < D; ++i) {
				C[i][i] += shift;
				diagD[i] += shift;
			}
		}

		// take square root of eigenvalues
		for (int i = 0; i < D; ++i) {
			diagD[i] = Math.sqrt(diagD[i]);
		}

		// invsqrtC = B * diag(D^-1) * B^T
		for (int i = 0; i < D; ++i) {
			for (int j = 0; j <= i; ++j) {
				double sum = 0.0;
				for (int k = 0; k < D; ++k) {
					sum += B[i][k] / diagD[k] * B[j][k];
				}
				invsqrtC[i][j] = invsqrtC[j][i] = sum;
			}
		}
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	protected static final void tred2(int n, final double[][] V, final double[] d, final double[] e) {

		// This is derived from the Algol procedures tred2 by
		// Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
		// Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
		// Fortran subroutine in EISPACK.
		System.arraycopy(V[n - 1], 0, d, 0, n);

		// Householder reduction to tridiagonal form.
		for (int i = n - 1; i > 0; i--) {

			// Scale to avoid under/overflow.
			double scale = 0.0;
			double h = 0.0;
			for (int k = 0; k < i; k++) {
				scale += Math.abs(d[k]);
			}
			if (scale == 0.0) {
				e[i] = d[i - 1];
				for (int j = 0; j < i; j++) {
					d[j] = V[i - 1][j];
					V[i][j] = V[j][i] = 0.0;
				}
			} else {

				// Generate Householder vector.
				for (int k = 0; k < i; k++) {
					d[k] /= scale;
					h += d[k] * d[k];
				}
				double f = d[i - 1];
				double g = Math.sqrt(h);
				if (f > 0) {
					g = -g;
				}
				e[i] = scale * g;
				h = h - f * g;
				d[i - 1] = f - g;
				Arrays.fill(e, 0, i, 0.0);

				// Apply similarity transformation to remaining columns.
				for (int j = 0; j < i; j++) {
					f = d[j];
					V[j][i] = f;
					g = e[j] + V[j][j] * f;
					for (int k = j + 1; k <= i - 1; k++) {
						g += V[k][j] * d[k];
						e[k] += V[k][j] * f;
					}
					e[j] = g;
				}
				BlasMath.dscalm(i, 1.0 / h, e, 1);
				f = BlasMath.ddotm(i, e, 1, d, 1);
				double hh = f / (h + h);
				BlasMath.daxpym(i, -hh, d, 1, e, 1);
				for (int j = 0; j < i; j++) {
					f = d[j];
					g = e[j];
					for (int k = j; k <= i - 1; k++) {
						V[k][j] -= (f * e[k] + g * d[k]);
					}
					d[j] = V[i - 1][j];
					V[i][j] = 0.0;
				}
			}
			d[i] = h;
		}

		// Accumulate transformations.
		for (int i = 0; i < n - 1; i++) {
			V[n - 1][i] = V[i][i];
			V[i][i] = 1.0;
			double h = d[i + 1];
			if (h != 0.0) {
				for (int k = 0; k <= i; k++) {
					d[k] = V[k][i + 1] / h;
				}
				for (int j = 0; j <= i; j++) {
					double g = 0.0;
					for (int k = 0; k <= i; k++) {
						g += V[k][i + 1] * V[k][j];
					}
					for (int k = 0; k <= i; k++) {
						V[k][j] -= g * d[k];
					}
				}
			}
			for (int k = 0; k <= i; k++) {
				V[k][i + 1] = 0.0;
			}
		}
		System.arraycopy(V[n - 1], 0, d, 0, n);
		Arrays.fill(V[n - 1], 0, n, 0.0);
		V[n - 1][n - 1] = 1.0;
		e[0] = 0.0;
	}

	// Symmetric tridiagonal QL algorithm, taken from JAMA package.
	protected static final void tql2(int n, double[] d, double[] e, double[][] V) {

		// This is derived from the Algol procedures tql2, by
		// Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
		// Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
		// Fortran subroutine in EISPACK.
		System.arraycopy(e, 1, e, 0, n - 1);
		e[n - 1] = 0.0;

		double f = 0.0;
		double tst1 = 0.0;
		double eps = BlasMath.D1MACH[3 - 1];
		for (int l = 0; l < n; l++) {

			// Find small subdiagonal element
			tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
			int m = l;
			while (m < n) {
				if (Math.abs(e[m]) <= eps * tst1) {
					break;
				}
				m++;
			}

			// If m == l, d[l] is an eigenvalue, otherwise, iterate.
			if (m > l) {
				do {

					// Compute implicit shift
					double g = d[l];
					double p = (d[l + 1] - g) / (2.0 * e[l]);
					double r = RealMath.hypot(p, 1.0);
					if (p < 0) {
						r = -r;
					}
					d[l] = e[l] / (p + r);
					d[l + 1] = e[l] * (p + r);
					double dl1 = d[l + 1];
					double h = g - d[l];
					for (int i = l + 2; i < n; i++) {
						d[i] -= h;
					}
					f += h;

					// Implicit QL transformation.
					p = d[m];
					double c = 1.0;
					double c2 = c;
					double c3 = c;
					double el1 = e[l + 1];
					double s = 0.0;
					double s2 = 0.0;
					for (int i = m - 1; i >= l; i--) {
						c3 = c2;
						c2 = c;
						s2 = s;
						g = c * e[i];
						h = c * p;
						r = RealMath.hypot(p, e[i]);
						e[i + 1] = s * r;
						s = e[i] / r;
						c = p / r;
						p = c * d[i] - s * g;
						d[i + 1] = h + s * (c * g + s * d[i]);

						// Accumulate transformation.
						for (int k = 0; k < n; k++) {
							h = V[k][i + 1];
							V[k][i + 1] = s * V[k][i] + c * h;
							V[k][i] = c * V[k][i] - s * h;
						}
					}
					p = -s * s2 * c3 * el1 * e[l] / dl1;
					e[l] = s * p;
					d[l] = c * p;

					// Check for convergence.
				} while (Math.abs(e[l]) > eps * tst1);
			}
			d[l] = d[l] + f;
			e[l] = 0.0;
		}

		// Sort eigenvalues and corresponding vectors.
		for (int i = 0; i < n - 1; i++) {
			int k = i;
			double p = d[i];
			for (int j = i + 1; j < n; j++) {
				if (d[j] < p) {

					// NH find smallest k>i
					k = j;
					p = d[j];
				}
			}
			if (k != i) {

				// swap k and i
				d[k] = d[i];
				d[i] = p;
				for (int j = 0; j < n; j++) {
					p = V[j][i];
					V[j][i] = V[j][k];
					V[j][k] = p;
				}
			}
		}
	}
}
