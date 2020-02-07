package min4j.multivariate.unconstrained.order0.cmaes;

import java.util.function.Function;

import min4j.utils.BlasMath;
import min4j.utils.Constants;

/**
 * Ros, Raymond, and Nikolaus Hansen. "A simple modification in CMA-ES achieving
 * linear time and space complexity." International Conference on Parallel
 * Problem Solving from Nature. Springer, Berlin, Heidelberg, 2008.
 * 
 * @author Michael
 *
 */
public final class SepCmaesAlgorithm extends AbstractCmaesOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	/**
	 * 
	 * @author Michael
	 *
	 */
	public static final class SepCmaesFactory implements AbstractCmaesFactory {

		final boolean myAdjustLr;

		/**
		 * 
		 * @param adjustLearningRate
		 */
		public SepCmaesFactory(final boolean adjustLearningRate) {
			myAdjustLr = adjustLearningRate;
		}

		@Override
		public SepCmaesAlgorithm createCmaStrategy(double tolerance, int populationSize, double initialSigma,
				int maxEvaluations) {
			return new SepCmaesAlgorithm(tolerance, populationSize, initialSigma, maxEvaluations, myAdjustLr);
		}
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// convergence parameters
	protected int myFlag;

	// additional algorithm parameters
	protected final boolean myAdjustLr;
	protected double ccov;
	protected double[] diagD, C;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 * @param applyLearningRateAdjustment
	 */
	public SepCmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations, final boolean applyLearningRateAdjustment) {
		super(tolerance, populationSize, initialSigma, maxEvaluations);
		myAdjustLr = applyLearningRateAdjustment;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 * @param applyLearningRateAdjustment
	 */
	public SepCmaesAlgorithm(final double tolerance, final double initialSigma, final int maxEvaluations,
			final boolean applyLearningRateAdjustment) {
		super(tolerance, initialSigma, maxEvaluations);
		myAdjustLr = applyLearningRateAdjustment;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
		super.initialize(func, guess);

		// Strategy parameter setting: Adaptation
		// we slightly modify the parameters given in other implementations of CMAES
		cc = 4.0 / (D + 4.0);
		cs = (mueff + 2.0) / (3.0 + D + mueff);
		damps = 1.0 + cs + 2.0 * Math.max(0.0, Math.sqrt((mueff - 1.0) / (D + 1.0)) - 1.0);

		// additional parameter ccov
		ccov = 2.0 / ((D + Constants.SQRT2) * (D + Constants.SQRT2) * mueff);
		ccov += Math.min(1.0, (2.0 * mueff - 1.0) / ((D + 2.0) * (D + 2.0) + mueff)) * (1.0 - 1.0 / mueff);

		// apply the empirical learning rate adjustment for separable functions
		// as in Hansen et. al (2008)
		if (myAdjustLr) {
			ccov *= (D + 2.0) / 3.0;
		}

		// Initialize dynamic (internal) strategy parameters and constants
		diagD = new double[D];
		C = new double[D];
		for (int d = 0; d < D; ++d) {
			diagD[d] = C[d] = 1.0;
		}

		// initialize convergence parameters
		myFlag = 0;
	}

	@Override
	public final void updateDistribution() {

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
			ps[i] += csc * C[i] * (xmean[i] - xold[i]) / sigma;
		}

		// compute hsig
		final double pslen = BlasMath.denorm(ps.length, ps);
		final double denom = 1.0 - Math.pow(1.0 - cs, 2.0 * myEvals / myLambda);
		final int hsig = pslen / Math.sqrt(denom) / chi < 1.4 + 2.0 / (D + 1.0) ? 1 : 0;

		// update pc
		final double ccc = Math.sqrt(cc * (2.0 - cc) * mueff);
		for (int i = 0; i < D; ++i) {
			pc[i] = (1.0 - cc) * pc[i] + hsig * ccc * (xmean[i] - xold[i]) / sigma;
		}

		// Adapt covariance matrix C
		for (int i = 0; i < D; ++i) {

			// old matrix plus rank-one update
			double sum = (1.0 - ccov) * C[i] + (ccov / mueff) * pc[i] * pc[i];

			// rank mu update
			for (int k = 0; k < myMu; ++k) {
				final int m = arfitness[k].index;
				final double di = (arx[m][i] - xold[i]) / sigma;
				sum += ccov * (1.0 - 1.0 / mueff) * weights[k] * di * di;
			}
			C[i] = sum;
			diagD[i] = Math.sqrt(C[i]);
		}

		// Adapt step size sigma
		updateSigma();
	}

	@Override
	public final void samplePopulation() {
		for (int n = 0; n < myLambda; ++n) {
			for (int i = 0; i < D; ++i) {
				arx[n][i] = xmean[i] + sigma * diagD[i] * RAND.nextGaussian();
			}
		}
	}

	@Override
	public boolean converged() {

		// we use the original Hansen convergence test but modified for the
		// diagonal representation of the covariance matrix
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
			if (Math.max(pc[i], diagD[i]) * sigma / mySigma0 >= myTol) {
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
		if (xmean[iaxis] == xmean[iaxis] + 0.1 * sigma * diagD[iaxis]) {
			myFlag = 8;
			return true;
		}

		// NoEffectCoor
		for (int i = 0; i < D; ++i) {
			if (xmean[i] == xmean[i] + 0.2 * sigma * diagD[i]) {
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
}
