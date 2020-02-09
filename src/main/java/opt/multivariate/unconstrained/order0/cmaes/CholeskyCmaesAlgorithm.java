/*
Copyright (c) 2020 Mike Gimelfarb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the > "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, > subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package opt.multivariate.unconstrained.order0.cmaes;

import java.util.function.Function;

import utils.BlasMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] Krause, Oswin, Dídac Rodríguez Arbonès, and Christian Igel. "CMA-ES with
 * optimal covariance update and storage complexity." Advances in Neural
 * Information Processing Systems. 2016.
 */
public final class CholeskyCmaesAlgorithm extends AbstractCmaesOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	/**
	 * 
	 * @author Michael
	 *
	 */
	public static final class CholeskyCmaesFactory implements AbstractCmaesFactory {

		@Override
		public CholeskyCmaesAlgorithm createCmaStrategy(double tolerance, int populationSize, double initialSigma,
				int maxEvaluations) {
			return new CholeskyCmaesAlgorithm(tolerance, tolerance, populationSize, initialSigma, maxEvaluations);
		}
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final double mySigmaTol;

	private double[] dmean;
	private double[][] A, mattmp;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public CholeskyCmaesAlgorithm(final double tolerance, final double stdevTolerance, final int populationSize,
			final double initialSigma, final int maxEvaluations) {
		super(tolerance, populationSize, initialSigma, maxEvaluations);
		mySigmaTol = stdevTolerance;
	}

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public CholeskyCmaesAlgorithm(final double tolerance, final double stdevTolerance, final double initialSigma,
			final int maxEvaluations) {
		super(tolerance, initialSigma, maxEvaluations);
		mySigmaTol = stdevTolerance;
	}

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param initialSigma
	 */
	public CholeskyCmaesAlgorithm(final double tolerance, final double stdevTolerance, final double initialSigma) {
		super(tolerance, initialSigma);
		mySigmaTol = stdevTolerance;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {
		super.initialize(func, guess);

		// Initialize dynamic (internal) strategy parameters and constants
		dmean = new double[D];
		A = new double[D][D];
		mattmp = new double[D][D];
		for (int d = 0; d < D; ++d) {
			A[d][d] = 1.0;
		}
	}

	@Override
	public final void updateDistribution() {

		// cache the old xmean
		System.arraycopy(xmean, 0, xold, 0, D);

		// compute weighted mean into xmean
		for (int i = 0; i < D; ++i) {
			double sum = 0.0;
			for (int n = 0; n < myMu; ++n) {
				final int j = arfitness[n].index;
				sum += weights[n] * arx[j][i];
			}
			xmean[i] = sum;
			dmean[i] = (xmean[i] - xold[i]) / sigma;
		}

		// update pc
		final double ccc = Math.sqrt(cc * (2.0 - cc) * mueff);
		for (int i = 0; i < D; ++i) {
			pc[i] = (1.0 - cc) * pc[i] + ccc * dmean[i];
		}

		// apply formula (2) to A(t)
		final double acoeff = Math.sqrt(1.0 - c1 - cmu);
		for (int i = 0; i < D; ++i) {
			for (int j = 0; j <= i; ++j) {
				mattmp[i][j] = acoeff * A[i][j];
			}
		}

		// perform the rank 1 updates to A(t+1)
		rank1update(D, mattmp, c1, pc, artmp);
		for (int i = 0; i < myMu; ++i) {
			for (int j = 0; j < D; ++j) {
				artmp[j] = (arx[i][j] - xmean[j]) / sigma;
			}
			rank1update(D, mattmp, cmu * weights[i], artmp, artmp);
		}

		// do back-substitution to compute A^-1 * (m(t+1)-m(t))
		System.arraycopy(dmean, 0, artmp, 0, D);
		for (int i = 0; i < D; ++i) {
			artmp[i] -= BlasMath.ddotm(i, A[i], 1, artmp, 1);
			artmp[i] /= A[i][i];
		}

		// compute the vector pc
		final double csc = Math.sqrt(cs * (2.0 - cs) * mueff);
		for (int i = 0; i < D; ++i) {
			ps[i] = (1.0 - cs) * ps[i] + csc * artmp[i];
		}

		// update A(t) to A(t+1)
		for (int i = 0; i < D; ++i) {
			System.arraycopy(mattmp[i], 0, A[i], 0, D);
		}

		// update the step size
		updateSigma();
	}

	@Override
	public final void samplePopulation() {
		for (int n = 0; n < myLambda; ++n) {
			for (int i = 0; i < D; ++i) {
				artmp[i] = RAND.nextGaussian();
			}
			for (int i = 0; i < D; ++i) {
				final double sum = BlasMath.ddotm(D, A[i], 1, artmp, 1);
				arx[n][i] = xmean[i] + sigma * sum;
			}
		}
	}

	@Override
	public boolean converged() {

		// check convergence in fitness difference between best and worst points
		final double y0 = ybw[0];
		final double y3 = ybw[3];
		final double toly = 0.5 * RELEPS * (Math.abs(y0) + Math.abs(y3));
		if (Math.abs(y0 - y3) > myTol + toly) {
			return false;
		}

		// compute standard deviation of swarm radiuses
		int count = 0;
		double mean = 0.0;
		double m2 = 0.0;
		for (final double[] pt : arx) {
			final double x = BlasMath.denorm(D, pt);
			++count;
			final double delta = x - mean;
			mean += delta / count;
			final double delta2 = x - mean;
			m2 += delta * delta2;
		}

		// test convergence in standard deviation
		return m2 <= (myLambda - 1) * mySigmaTol * mySigmaTol;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static void rank1update(final int D, final double[][] a, final double beta, final double[] v,
			final double[] artmp) {
		System.arraycopy(v, 0, artmp, 0, D);
		double b = 1.0;
		for (int j = 0; j < D; ++j) {
			final double ajj = a[j][j];
			final double alfaj = artmp[j];
			final double gam = ajj * ajj * b + beta * alfaj * alfaj;
			final double a1jj = a[j][j] = Math.sqrt(gam / b);
			for (int k = j + 1; k < D; ++k) {
				artmp[k] -= (alfaj / ajj) * a[k][j];
				a[k][j] = (a1jj / ajj) * a[k][j] + (a1jj * beta * alfaj) / gam * artmp[k];
			}
			b += beta * (alfaj / ajj) * (alfaj / ajj);
		}
	}
}
