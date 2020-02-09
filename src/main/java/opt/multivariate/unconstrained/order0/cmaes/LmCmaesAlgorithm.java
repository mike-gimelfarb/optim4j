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

import java.util.Arrays;
import java.util.function.Function;

import utils.BlasMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] Loshchilov, Ilya. "LM-CMA: An alternative to L-BFGS for large-scale black
 * box optimization." Evolutionary computation 25.1 (2017): 143-171.
 * 
 * [2] Loshchilov, Ilya. "A computationally efficient limited memory CMA-ES for
 * large scale optimization." Proceedings of the 2014 Annual Conference on
 * Genetic and Evolutionary Computation. ACM, 2014.
 *
 */
public class LmCmaesAlgorithm extends AbstractCmaesOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	/**
	 * 
	 * @author Michael
	 *
	 */
	public static class LmCmaesFactory implements AbstractCmaesFactory {

		@Override
		public LmCmaesAlgorithm createCmaStrategy(double tolerance, int populationSize, double initialSigma,
				int maxEvaluations) {
			return new LmCmaesAlgorithm(tolerance, populationSize, initialSigma, maxEvaluations);
		}
	}

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	public static final Function<Integer, Integer> SMALL_MEMORY = d -> 4 + (int) Math.log(3.0 * d);

	public static final Function<Integer, Integer> LARGE_MEMORY = d -> (int) (2.0 * Math.sqrt(d));

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// algorithm parameters
	private final Function<Integer, Integer> myMemorySizeFunction;
	private final boolean myUseNewMethod;
	private final int mySampleMode;
	private final double mySigmaMinTol = 1e-16;

	// additional algorithm memory
	private int myMemoryLength, myMemorySize, myNSteps, myT;
	private double sqrt1mc1, zstar, s, ccc;
	private int[] jarr, larr;
	private double[] b, d, Az, prevFitness;
	private double[][] pcmat, vmat;
	private IntDoublePair[] mixed;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 * 
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 * @param sampleRademacher
	 * @param useNewMethod
	 * @param memorySize
	 */
	public LmCmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations, final boolean sampleRademacher, final boolean useNewMethod,
			final Function<Integer, Integer> memorySize) {
		super(tolerance, populationSize, initialSigma, maxEvaluations);
		mySampleMode = sampleRademacher ? 1 : 0;
		myUseNewMethod = useNewMethod;
		myMemorySizeFunction = memorySize;
	}

	/**
	 * 
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 * @param memorySize
	 */
	public LmCmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations, final Function<Integer, Integer> memorySize) {
		this(tolerance, populationSize, initialSigma, maxEvaluations, true, true, memorySize);
	}

	/**
	 * 
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public LmCmaesAlgorithm(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations) {
		this(tolerance, populationSize, initialSigma, maxEvaluations, LARGE_MEMORY);
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 * @param maxEvaluations
	 * @param memorySize
	 */
	public LmCmaesAlgorithm(final double tolerance, final double initialSigma, final int maxEvaluations,
			final Function<Integer, Integer> memorySize) {
		super(tolerance, initialSigma, maxEvaluations);
		mySampleMode = 1;
		myUseNewMethod = true;
		myMemorySizeFunction = memorySize;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public LmCmaesAlgorithm(final double tolerance, final double initialSigma, final int maxEvaluations) {
		this(tolerance, initialSigma, maxEvaluations, LARGE_MEMORY);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {

		// call super method
		super.initialize(func, guess);

		// adjust the learning parameters for LM-CMA-ES in Loshchilov (2015)
		myMemorySize = myMemorySizeFunction.apply(D);
		myMemoryLength = 0;
		if (myUseNewMethod) {
			myNSteps = D;
			myT = Math.max(1, (int) Math.log(D));
			cc = 0.5 / Math.sqrt(D);
		} else {
			myNSteps = myMemorySize;
			myT = 1;
			cc = 1.0 / myMemorySize;
		}
		cs = 0.3;
		c1 = 0.1 / Math.log(D + 1.0);
		cmu = Double.NaN;
		ccc = Math.sqrt(cc * (2.0 - cc) * mueff);
		damps = 1.0;
		sqrt1mc1 = Math.sqrt(1.0 - c1);
		zstar = 0.25;
		s = 0.0;

		// initialize additional memory
		jarr = new int[myMemorySize];
		larr = new int[myMemorySize];
		b = new double[myMemorySize];
		d = new double[myMemorySize];
		Az = new double[D];
		prevFitness = new double[myLambda];
		pcmat = new double[myMemorySize][D];
		vmat = new double[myMemorySize][D];

		// initialize pooled ranking
		mixed = new IntDoublePair[myLambda << 1];
		for (int n = 0; n < (myLambda << 1); ++n) {
			mixed[n] = new IntDoublePair();
		}
	}

	@Override
	public boolean converged() {

		// MaxIter
		if (myIteration >= myMaxIters) {
			return true;
		}

		// SigmaTooSmall
		if (sigma < mySigmaMinTol) {
			return true;
		}

		// TolHistFun
		if (myIteration >= myHistoryLength && myHistoryWorstFit - myHistoryBestFit < myTol) {
			return true;
		}

		// EqualFunVals
		if (myHistoryBest.length >= D && myHistoryKth.length >= D) {
			int countEq = 0;
			for (int i = 0; i < D; ++i) {
				if (myHistoryBest.get(i) == myHistoryKth.get(i)) {
					++countEq;
					if (3 * countEq >= D) {
						return true;
					}
				}
			}
		}
		return false;
	}

	@Override
	public void samplePopulation() {
		int sign = 1;
		for (int n = 0; n < myLambda; ++n) {
			if (sign == 1) {

				// sample a candidate vector
				if (mySampleMode == 0) {

					// sample from a Gaussian distribution
					for (int i = 0; i < D; ++i) {
						artmp[i] = Az[i] = RAND.nextGaussian();
					}
				} else {

					// sample from a Rademacher distribution
					for (int i = 0; i < D; ++i) {
						artmp[i] = Az[i] = RAND.nextBoolean() ? 1 : -1;
					}
				}

				// perform Cholesky factor vector update
				// this is algorithm 3 in Loshchilov (2015)
				final int i0;
				if (myUseNewMethod) {
					i0 = selectSubset(myMemoryLength, n);
				} else {
					i0 = 0;
				}
				for (int i = i0; i < myMemoryLength; ++i) {
					final int j = jarr[i];
					final double dot = b[j] * BlasMath.ddotm(D, vmat[j], 1, artmp, 1);
					BlasMath.dscalm(D, sqrt1mc1, Az, 1);
					BlasMath.daxpym(D, dot, pcmat[j], 1, Az, 1);
				}
			}
			BlasMath.daxpy1(D, sign * sigma, Az, 1, xmean, 1, arx[n], 1);
			sign = -sign;
		}
	}

	@Override
	public void updateDistribution() {

		// compute weighted mean into xmean
		System.arraycopy(xmean, 0, xold, 0, D);
		Arrays.fill(xmean, 0.0);
		for (int n = 0; n < myMu; ++n) {
			final int i = arfitness[n].index;
			BlasMath.daxpym(D, weights[n], arx[i], 1, xmean, 1);
		}

		// Cumulation: Update evolution paths
		for (int i = 0; i < D; ++i) {
			pc[i] = (1.0 - cc) * pc[i] + ccc * (xmean[i] - xold[i]) / sigma;
		}

		if (myIteration % myT == 0) {

			// select the direction vectors
			final int imin = updateSet(myMemorySize, jarr, larr, myIteration, myNSteps, myT);
			if (myMemoryLength < myMemorySize) {
				++myMemoryLength;
			}

			// copy cumulation path vector into matrix
			int jcur = jarr[myMemoryLength - 1];
			System.arraycopy(pc, 0, pcmat[jcur], 0, D);

			// recompute v vectors
			for (int i = imin; i < myMemoryLength; ++i) {

				// this part is adapted from the code by Loshchilov
				jcur = jarr[i];
				System.arraycopy(pcmat[jcur], 0, artmp, 0, D);
				Ainvz(D, i, 1.0 / sqrt1mc1, jarr, d, artmp, vmat);
				System.arraycopy(artmp, 0, vmat[jcur], 0, D);

				// compute b and d vectors
				final double vnrm2 = BlasMath.ddotm(D, artmp, 1, artmp, 1);
				final double sqrtc1 = Math.sqrt(1.0 + (c1 / (1.0 - c1)) * vnrm2);
				b[jcur] = (sqrt1mc1 / vnrm2) * (sqrtc1 - 1.0);
				d[jcur] = (1.0 / (sqrt1mc1 * vnrm2)) * (1.0 - 1.0 / sqrtc1);
			}
		}

		// update sigma parameters
		updateSigma();
	}

	@Override
	public void evaluateAndSortPopulation() {

		// cache the previous fitness
		if (myIteration > 0) {
			for (int n = 0; n < myLambda; ++n) {
				prevFitness[n] = arfitness[n].value;
			}
		}

		// now perform fitness evaluation
		super.evaluateAndSortPopulation();
	}

	@Override
	public void updateSigma() {
		if (myIteration == 0) {
			return;
		}

		// combine the members from the current and previous populations and sort
		for (int n = 0; n < myLambda; ++n) {
			mixed[n].index = n;
			mixed[n].value = prevFitness[n];
			mixed[n + myLambda].index = n + myLambda;
			mixed[n + myLambda].value = arfitness[n].value;
		}
		Arrays.sort(mixed);

		// compute normalized success measure
		double zpsr = 0.0;
		for (int n = 0; n < (myLambda << 1); ++n) {
			final double f = (1.0 * n) / myLambda;
			if (mixed[n].index < myLambda) {
				zpsr += f;
			} else {
				zpsr -= f;
			}
		}
		zpsr /= myLambda;
		zpsr -= zstar;

		// update sigma
		s = (1.0 - cs) * s + cs * zpsr;
		sigma *= Math.exp(s / damps);
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static void Ainvz(final int n, int jlen, final double c, final int[] jarr, final double[] d,
			final double[] Av, final double[][] v) {

		// this is algorithm 4 in Loshchilov (2015)
		for (int i = 0; i < jlen; ++i) {
			final int idx = jarr[i];
			final double dot = d[idx] * BlasMath.ddotm(n, v[idx], 1, Av, 1);
			BlasMath.dscalm(n, c, Av, 1);
			BlasMath.daxpym(n, -dot, v[idx], 1, Av, 1);
		}
	}

	private static int updateSet(final int m, final int[] j, final int[] l, int iter, final int N, final int T) {

		// this is algorithm 5 in Loshchilov (2015)
		iter = Math.floorDiv(iter, T);
		int imin = 1;
		if (iter < m) {
			j[iter] = iter;
		} else if (m > 1) {
			int iminval = l[j[1]] - l[j[0]];
			for (int i = 1; i < m - 1; ++i) {
				final int val = l[j[i + 1]] - l[j[i]];
				if (val < iminval) {
					iminval = val;
					imin = i + 1;
				}
			}
			if (iminval >= N) {
				imin = 0;
			}
			final int jtmp = j[imin];
			for (int i = imin; i < m - 1; ++i) {
				j[i] = j[i + 1];
			}
			j[m - 1] = jtmp;
		}
		final int jcur = j[Math.min(iter, m - 1)];
		l[jcur] = iter * T;
		return imin == 1 ? 0 : imin;
	}

	private static int selectSubset(final int m, final int k) {

		// this is algorithm 6 in Loshchilov (2015)
		if (m <= 1) {
			return 0;
		}
		int msigma = 4;
		if (k == 0) {
			msigma *= 10;
		}
		int mstar = (int) (msigma * Math.abs(RAND.nextGaussian()));
		mstar = Math.min(mstar, m);
		mstar = m - mstar;
		return mstar;
	}
}