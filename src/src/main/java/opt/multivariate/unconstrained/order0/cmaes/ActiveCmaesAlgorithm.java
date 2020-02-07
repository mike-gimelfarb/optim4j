package opt.multivariate.unconstrained.order0.cmaes;

import java.util.function.Function;

import utils.BlasMath;

/**
 * Hansen, Nikolaus, and Raymond Ros. "Benchmarking a weighted negative
 * covariance matrix update on the BBOB-2010 noiseless testbed." Proceedings of
 * the 12th annual conference companion on Genetic and evolutionary computation.
 * ACM, 2010.
 * 
 * Jastrebski, Grahame A., and Dirk V. Arnold. "Improving evolution strategies
 * through active covariance matrix adaptation." Evolutionary Computation, 2006.
 * CEC 2006. IEEE Congress on. IEEE, 2006.
 * 
 * @author Michael
 *
 */
public class ActiveCmaesAlgorithm extends CmaesAlgorithm {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	/**
	 * 
	 * @author Michael
	 *
	 */
	public static class ActiveCmaesFactory extends CmaesFactory {

		@Override
		public ActiveCmaesAlgorithm createCmaStrategy(double tolerance, int populationSize, double initialSigma,
				int maxEvaluations) {
			return new ActiveCmaesAlgorithm(tolerance, populationSize, initialSigma, maxEvaluations);
		}
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// additional algorithm parameters
	private final double myAlphaCov;

	// other algorithm memory
	private double cm, cneg, alphaold;
	private double[] ycoeff;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 * 
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 * @param alphaCov
	 */
	public ActiveCmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations, final double alphaCov) {
		super(tolerance, populationSize, initialSigma, maxEvaluations);
		myAlphaCov = alphaCov;
	}

	/**
	 * 
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public ActiveCmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations) {
		this(tolerance, populationSize, initialSigma, maxEvaluations, 2.0);
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public ActiveCmaesAlgorithm(final double tolerance, final double initialSigma, final int maxEvaluations) {
		super(tolerance, initialSigma, maxEvaluations);
		myAlphaCov = 2.0;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 */
	public ActiveCmaesAlgorithm(final double tolerance, final double initialSigma) {
		super(tolerance, initialSigma);
		myAlphaCov = 2.0;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
		super.initialize(func, guess);

		// re-define other parameters
		// note that some parameters may be re-defined in future implementations
		// as described in Hansen et al. (2010)
		cm = 1.0;
		alphaold = 0.5;
		cc = (4.0 + 0.0 * mueff / D) / (D + 4.0 + 0.0 * 2.0 * mueff / D);
		cs = (mueff + 2.0) / (3.0 + D + mueff);
		c1 = myAlphaCov * Math.min(1.0, myLambda / 6.0) / ((D + 1.3) * (D + 1.3) + mueff);
		cmu = 1.0 - c1;
		cmu = Math.min(cmu,
				myAlphaCov * (mueff - 2.0 + 1.0 / mueff) / ((2.0 + D) * (2.0 + D) + myAlphaCov * mueff / 2.0));
		cneg = (1.0 - cmu) * (myAlphaCov / 8.0) * mueff / (Math.pow(D + 2.0, 1.5) + 2.0 * mueff);
		damps = 1.0 + cs + 2.0 * Math.max(0.0, Math.sqrt((mueff - 1.0) / (D + 1.0)) - 1.0);

		// we perform an eigenvalue decomposition every O(d) iterations
		updateEigenFrequency = (1.0 / (c1 + cmu + cneg)) / (10.0 * D);
		updateEigenLastEval = 0;

		// other new storage
		ycoeff = new double[myMu];
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
			xmean[i] = xold[i] * (1.0 - cm) + sum * cm;
		}

		// Cumulation: Update evolution paths
		final double csc = Math.sqrt(cs * (2.0 - cs) * mueff);
		for (int i = 0; i < D; ++i) {
			ps[i] *= (1.0 - cs);
			for (int j = 0; j < D; ++j) {
				ps[i] += csc * invsqrtC[i][j] * (xmean[j] - xold[j]) / (cm * sigma);
			}
		}

		// compute hsig
		final double pslen = BlasMath.denorm(ps.length, ps);
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
			pc[i] = (1.0 - cc) * pc[i] + hsig * ccc * (xmean[i] - xold[i]) / (cm * sigma);
		}

		// compute the coefficients for the vectors for the negative update
		for (int i = 0; i < myMu; ++i) {
			final int mtop = arfitness[myLambda - myMu + 1 + i - 1].index;
			final int mbot = arfitness[myLambda - i - 1].index;
			double ssqtop = 0.0;
			double ssqbot = 0.0;
			for (int j = 0; j < D; ++j) {
				double termtop = 0.0;
				double termbot = 0.0;
				for (int l = 0; l < D; ++l) {
					termtop += invsqrtC[j][l] * (arx[mtop][l] - xold[l]);
					termbot += invsqrtC[j][l] * (arx[mbot][l] - xold[l]);
				}
				ssqtop += termtop * termtop;
				ssqbot += termbot * termbot;
			}
			ssqbot = Math.max(ssqbot, 1e-8);
			ycoeff[i] = ssqtop / ssqbot;
		}

		// Adapt covariance matrix C
		final double c2 = (1.0 - hsig) * cc * (2.0 - cc);
		final double cmu1 = cmu + cneg * (1.0 - alphaold);
		for (int i = 0; i < D; ++i) {
			for (int j = 0; j <= i; ++j) {

				// old matrix plus rank-one update
				double sum = (1.0 - c1 - cmu + cneg * alphaold) * C[i][j] + c1 * (pc[i] * pc[j] + c2 * C[i][j]);

				// rank mu update
				for (int k = 0; k < myMu; ++k) {
					final int m = arfitness[k].index;
					final double di = (arx[m][i] - xold[i]) / sigma;
					final double dj = (arx[m][j] - xold[j]) / sigma;
					sum += cmu1 * weights[k] * di * dj;
				}

				// active update: this is the main modification in active CMA-ES
				for (int k = 0; k < myMu; ++k) {
					final int m = arfitness[myLambda - k - 1].index;
					final double di = (arx[m][i] - xold[i]) / sigma;
					final double dj = (arx[m][j] - xold[j]) / sigma;
					sum -= cneg * weights[k] * ycoeff[k] * di * dj;
				}
				C[i][j] = sum;
			}
		}

		// update sigma
		updateSigma();

		// Decomposition of C into B*diag(D.^2)*B' (diagonalization)
		eigenDecomposition();
	}
}
