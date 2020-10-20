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
package opt.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.GradientFreeOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.Sequences;

/**
 * 
 * REFERENCES:
 * 
 * [1] Rao, R. "Jaya: A simple and new optimization algorithm for solving
 * constrained and unconstrained optimization problems." International Journal
 * of Industrial Engineering Computations 7.1 (2016): 19-34.
 * 
 * [2] Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population
 * and Lévy flights for solving economic load dispatch problems." IEEE Access 7
 * (2019): 21372-21384.
 * 
 */
public final class AMLJayaAlgorithm extends GradientFreeOptimizer {

	// particle object
	private final class Particle {

		private double myF;
		private double[] myX;

		public Particle() {
			myX = new double[myD];
			for (int i = 0; i < myD; ++i) {
				myX[i] = (myUpper[i] - myLower[i]) * RAND.nextDouble() + myLower[i];
			}
			myF = myFunc.apply(myX);
		}
	}

	// parameters
	private final int myNp, myMinNp;
	private final int myMaxEvals;
	private final Function<Integer, Double> myScale;
	private final double myBeta;
	private final int myK0;
	private final boolean myAdaptK;

	// problem setting
	private Function<? super double[], Double> myFunc;
	private double[] myLower, myUpper;
	private int myD;

	// counters
	private int myIters, myEvals;
	private double myBestFit, myPrevBestFit;
	private Particle myBestX;

	// algorithm memory
	private int myK;
	private int[] myLens;
	private double[] myTemp;
	private double[][] myR;
	private Particle[] myPool;

	/**
	 * 
	 * @param populationSize
	 * @param scale
	 * @param beta
	 * @param numPopulations
	 * @param maxEvaluations
	 * @param useSubpopulations
	 * @param minSubpopulationSize
	 */
	public AMLJayaAlgorithm(final int populationSize, final Function<Integer, Double> scale, final double beta,
			final int numPopulations, final int maxEvaluations, final boolean useSubpopulations,
			final int minSubpopulationSize) {
		super(0.0);
		myNp = populationSize;
		myMaxEvals = maxEvaluations;
		myScale = scale;
		myBeta = beta;
		myK0 = numPopulations;
		myAdaptK = useSubpopulations;
		myMinNp = minSubpopulationSize;
	}

	/**
	 * 
	 * @param populationSize
	 * @param maxEvaluations
	 * @param minSubpopulationSize
	 */
	public AMLJayaAlgorithm(final int populationSize, final int maxEvaluations, final int minSubpopulationSize) {
		this(populationSize, t -> 0.01, 1.5, 2, maxEvaluations, true, minSubpopulationSize);
	}

	@Override
	public void initialize(Function<? super double[], Double> func, double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		initialize(func, lo, hi);
	}

	@Override
	public void iterate() {

		// divide population into K sub-populations
		divideSubPopulations();

		// prepare for the current iteration
		myBestFit = Double.POSITIVE_INFINITY;
		final double scale = myScale.apply(myIters);
		setUniformNumbers();

		// apply the evolution algorithm to each sub-population
		int idx_next_subpop = 0;
		for (int q = 0; q < myK; ++q) {

			// get best and worst members
			Particle best = myPool[0], worst = myPool[0];
			for (int i = idx_next_subpop; i < idx_next_subpop + myLens[q]; ++i) {
				if (myPool[i].myF < best.myF) {
					best = myPool[i];
				}
				if (myPool[i].myF > worst.myF) {
					worst = myPool[i];
				}
			}

			// perform update
			for (int i = idx_next_subpop; i < idx_next_subpop + myLens[q]; ++i) {
				evolveParticle(i, scale, best, worst);
			}
			idx_next_subpop += myLens[q];
		}

		// adapt the number of sub-populations
		adaptK();

		// increment counters
		myPrevBestFit = myBestFit;
		System.out.println(myBestFit);
		System.out.println(myK);
		++myIters;
	}

	@Override
	public MultivariateOptimizerSolution optimize(Function<? super double[], Double> func, double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, lo, hi);
	}

	/**
	 * 
	 * @param func
	 * @param lower
	 * @param upper
	 */
	public void initialize(final Function<? super double[], Double> func, final double[] lower, final double[] upper) {

		// initialize problem
		myFunc = func;
		myLower = lower;
		myUpper = upper;
		myD = lower.length;

		// initialize counters
		myIters = 0;
		myEvals = myNp;
		myPrevBestFit = Double.POSITIVE_INFINITY;
		myBestFit = Double.POSITIVE_INFINITY;

		// initialize algorithm memory
		myPool = new Particle[myNp];
		for (int i = 0; i < myNp; ++i) {
			myPool[i] = new Particle();
		}
		myTemp = new double[myD];
		myLens = new int[myNp];
		myK = myK0;
		myR = new double[2][myD];
	}

	/**
	 * 
	 * @param func
	 * @param lower
	 * @param upper
	 * @return
	 */
	public MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func, final double[] lower,
			final double[] upper) {

		// initialization
		initialize(func, lower, upper);

		// main loop
		while (true) {
			iterate();

			// check max number of evaluations
			if (myEvals >= myMaxEvals) {
				break;
			}
		}
		return new MultivariateOptimizerSolution(myBestX.myX, myEvals, 0, false);
	}

	private final void divideSubPopulations() {

		// allocate N_subpop = NP // K elements at random to each sub-population
		Sequences.shuffle(RAND, 0, myNp - 1, myPool);
		final int base_len = Math.floorDiv(myNp, myK);
		Arrays.fill(myLens, base_len);

		// since the total number of elements must be NP, we fill the remaining
		// sub-populations randomly
		if (base_len * myK < myNp) {
			for (int i = 0; i < myNp - base_len * myK; ++i) {
				final int idx = RAND.nextInt(myK);
				++myLens[idx];
			}
		}

		// check that the total number of elements is equal to NP among all K
		// sub-population
		int total = 0;
		for (int k = 0; k < myK; ++k) {
			total += myLens[k];
		}
		if (total != myNp) {
			System.err.println("Problem with the sub-population assignment. Please file an issue on github.");
		}
	}

	private final void evolveParticle(final int i, final double scale, final Particle best, final Particle worst) {

		// evolve the position
		for (int j = 0; j < myD; ++j) {
			final double step = sampleLevyStep(myBeta);
			final double stepSize = scale * step * (myPool[i].myX[j] - best.myX[j]);
			final double levy_ij = myPool[i].myX[j] + stepSize * RAND.nextDouble();
			myTemp[j] = levy_ij + myR[0][j] * (best.myX[j] - Math.abs(myPool[i].myX[j]))
					- myR[1][j] * (worst.myX[j] - Math.abs(myPool[i].myX[j]));
			myTemp[j] = Math.max(myTemp[j], myLower[j]);
			myTemp[j] = Math.min(myTemp[j], myUpper[j]);
		}

		// evaluate fitness of new position
		final double ytemp = myFunc.apply(myTemp);
		++myEvals;

		// copy the particle back to the swarm if it is an improvement
		if (ytemp < myPool[i].myF) {
			System.arraycopy(myTemp, 0, myPool[i].myX, 0, myD);
			myPool[i].myF = ytemp;
		}

		// update the global best for the current iteration
		if (myPool[i].myF < myBestFit) {
			myBestFit = myPool[i].myF;
			myBestX = myPool[i];
		}
	}

	private final void adaptK() {
		if (myAdaptK) {
			if (myBestFit < myPrevBestFit) {
				if (myNp >= myMinNp * (myK + 1)) {
					++myK;
				}
			} else if (myBestFit > myPrevBestFit) {
				if (myK > 1) {
					--myK;
				}
			}
		}
	}

	private final void setUniformNumbers() {
		for (int j = 0; j < myD; ++j) {
			myR[0][j] = RAND.nextDouble();
			myR[1][j] = RAND.nextDouble();
		}
	}

	private static final double sampleLevyStep(final double beta) {

		// Mantegna's algorithm
		final double sigma_u = Math.pow((gamma(1.0 + beta) * Math.sin(beta * Math.PI / 2.0))
				/ (gamma((1.0 + beta) / 2.0) * beta * Math.pow(2.0, (beta - 1.0) / 2.0)), 1.0 / beta);
		final double sigma_v = 1.0;
		final double u = RAND.nextGaussian() * sigma_u;
		final double v = RAND.nextGaussian() * sigma_v;
		return u / (Math.pow(Math.abs(v), 1.0 / beta));
	}

	private static final double gamma(double x) {
		final double u = 1.0 + 76.18009173 / x - 86.50532033 / (x + 1) + 24.01409822 / (x + 2) - 1.231739516 / (x + 3)
				+ 0.00120858003 / (x + 4) - 0.00000536382 / (x + 5);
		final double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
		final double lngamma = tmp + Math.log(u * Math.sqrt(2 * Math.PI));
		return Math.exp(lngamma);
	}
}
