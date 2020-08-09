package opt.multivariate.unconstrained.order0.evol;

import java.util.function.Function;

import opt.multivariate.GradientFreeOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.BlasMath;
import utils.Sequences;

/**
 * 
 * REFERENCES:
 * 
 * [1] Wang, Hui, et al. "Randomly attracted firefly algorithm with neighborhood
 * search and dynamic parameter adjustment mechanism." Soft Computing 21.18
 * (2017): 5325-5339.
 * 
 * [2] I. Fister Jr., X.-S. Yang, I. Fister, J. Brest, Memetic firefly algorithm
 * for combinatorial optimization, in Bioinspired Optimization Methods and their
 * Applications (BIOMA 2012), B. Filipic and J.Silc, Eds. Jozef Stefan
 * Institute, Ljubljana, Slovenia, 2012.
 * 
 * [3] Yu S, Zhu S, Ma Y, Mao D. Enhancing firefly algorithm using generalized
 * opposition-based learning. Computing. 2015 :97(7) 741–754.
 * 
 * [4] Shakarami MR, Sedaghati R. A new approach for network reconfiguration
 * problem in order to deviation bus voltage minimization with regard to
 * probabilistic load model and DGs. International Journal of Electrical,
 * Computer, Energetic, Electronic and Communication Engineering.
 * 2014;8(2):430–5.
 * 
 */
public final class AdaptiveFireflyAlgorithm extends GradientFreeOptimizer {

	// firefly object
	private final class Firefly implements Comparable<Firefly> {

		double intensity, alpha;
		double[] position, pbest;

		@Override
		public final int compareTo(Firefly o) {
			return Double.compare(intensity, o.intensity);
		}

		final void randomize() {
			final double r = RAND.nextDouble();
			position = new double[myD];
			for (int i = 0; i < myD; ++i) {
				position[i] = r * (myUpper[i] - myLower[i]) + myLower[i];
			}
			intensity = Double.POSITIVE_INFINITY;
			alpha = Double.NaN;
			pbest = position.clone();
			updateIntensity();
		}

		final void updateIntensity() {
			intensity = myFunc.apply(position);
			++myEvals;
		}
	}

	// adaptation strategy
	public static abstract class StepSizeStrategy {

		public abstract double updateStepSize(final AdaptiveFireflyAlgorithm parent, final int i);
	}

	public static class Geometric extends StepSizeStrategy {

		private final double myAlpha0, myDecay;

		public Geometric(final double alphaInit, final double decay) {
			myAlpha0 = alphaInit;
			myDecay = decay;
		}

		@Override
		public double updateStepSize(final AdaptiveFireflyAlgorithm parent, final int i) {
			final Firefly fly = parent.mySwarm[i];
			final double alpha;
			if (Double.isNaN(fly.alpha)) {
				alpha = myAlpha0;
			} else {
				alpha = fly.alpha * myDecay;
			}
			fly.alpha = alpha;
			return alpha;
		}
	}

	public static class Sh2014 extends StepSizeStrategy {

		private final double myAlpha0;

		public Sh2014(final double alphaInit) {
			myAlpha0 = alphaInit;
		}

		@Override
		public double updateStepSize(final AdaptiveFireflyAlgorithm parent, final int i) {
			final Firefly fly = parent.mySwarm[i];
			final double decay = Math.pow(0.5 / parent.myMaxIters, 1.0 / parent.myMaxIters);
			final double alpha;
			if (Double.isNaN(fly.alpha)) {
				alpha = myAlpha0;
			} else {
				alpha = fly.alpha * decay;
			}
			fly.alpha = alpha;
			return alpha;
		}
	}

	public static final class Memetic extends StepSizeStrategy {

		private final double myAlpha0, myDecay;

		public Memetic(final double alphaInit, final double decay) {
			myAlpha0 = alphaInit;
			myDecay = decay;
		}

		public Memetic(final double alphaInit) {
			this(alphaInit, Math.pow(10.0, -4.0) / 0.9);
		}

		@Override
		public double updateStepSize(final AdaptiveFireflyAlgorithm parent, final int i) {
			final Firefly fly = parent.mySwarm[i];
			final double alpha;
			if (Double.isNaN(fly.alpha)) {
				alpha = myAlpha0;
			} else {
				alpha = fly.alpha * Math.pow(myDecay, 1.0 / parent.myMaxIters);
			}
			fly.alpha = alpha;
			return alpha;
		}
	}

	// randomization strategy
	public enum Noise {
		UNIFORM, GAUSSIAN, CAUCHY, NONE
	}

	// function properties
	private Function<? super double[], Double> myFunc;
	private double[] myLower, myUpper;
	private int myD;

	// counters
	private int myIter, myEvals;
	private Firefly myCurrentBest;

	// swarm
	private final int myN;
	private Firefly[] mySwarm;

	// exploration parameters
	private final boolean myDoNeighborhoodSearch, myUseOpposition;
	private final int myK;
	private final double myWorstToBestProb;
	private final Noise myNoiseStrategy;

	// step size parameters
	private final StepSizeStrategy myStepSizeStrategy;
	private double myMinAlpha, myMaxAlpha;

	// algorithm parameters
	private final int myMaxEvals, myMaxIters;
	private final double myBetaMin, myBetaMax, myGamma;

	// working memory
	private int[] tempK;
	private double[] tempD, temp4;
	private double[][] tempX;

	/**
	 * 
	 * @param swarmSize
	 * @param betaMin
	 * @param betaMax
	 * @param gamma
	 * @param noiseStrategy
	 * @param stepStrategy
	 * @param doNeighborhoodSearch
	 * @param neighborhoodSize
	 * @param useOppositionSearch
	 * @param worstToBestProb
	 * @param maxEvaluations
	 * @param maxIterations
	 */
	public AdaptiveFireflyAlgorithm(final int swarmSize, final double betaMin, final double betaMax, final double gamma,
			final Noise noiseStrategy, final StepSizeStrategy stepStrategy, final boolean doNeighborhoodSearch,
			final int neighborhoodSize, final boolean useOppositionSearch, final double worstToBestProb,
			final int maxEvaluations, final int maxIterations) {
		super(0.0);
		myN = swarmSize;
		myBetaMin = betaMin;
		myBetaMax = betaMax;
		myGamma = gamma;
		myNoiseStrategy = noiseStrategy;
		myStepSizeStrategy = stepStrategy;
		myDoNeighborhoodSearch = doNeighborhoodSearch;
		myUseOpposition = useOppositionSearch;
		myWorstToBestProb = worstToBestProb;
		myK = Math.min(neighborhoodSize, (myN - 1) / 2);
		myMaxEvals = maxEvaluations;
		myMaxIters = maxIterations;
	}

	/**
	 * 
	 * @param swarmSize
	 * @param betaMin
	 * @param betaMax
	 * @param gamma
	 * @param stepStrategy
	 * @param neighborhoodSize
	 * @param worstToBestProb
	 * @param maxEvaluations
	 */
	public AdaptiveFireflyAlgorithm(final int swarmSize, final double betaMin, final double betaMax, final double gamma,
			final StepSizeStrategy stepStrategy, final int neighborhoodSize, final double worstToBestProb,
			final int maxEvaluations) {
		this(swarmSize, betaMin, betaMax, gamma, Noise.UNIFORM, stepStrategy, true, neighborhoodSize, true,
				worstToBestProb, maxEvaluations, maxEvaluations / swarmSize);
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
		updateFireflies();
		updateStatistics();
		if (myIter % 100 == 0) {
			System.out.println(
					myIter + "\t" + myEvals + "\t" + myCurrentBest.intensity + "\t" + myMinAlpha + "\t" + myMaxAlpha);
		}
		++myIter;
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
		initializeProblem(func, lower, upper);
		initializeMemory();
		initializeFireflies();
		updateStatistics();
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

			// check max number of generations
			if (myIter >= myMaxIters) {
				break;
			}
		}
		return new MultivariateOptimizerSolution(myCurrentBest.position, myEvals, 0, false);
	}

	private final void initializeProblem(final Function<? super double[], Double> func, final double[] lower,
			final double[] upper) {
		myFunc = func;
		myD = lower.length;
		myLower = lower;
		myUpper = upper;
	}

	private final void initializeFireflies() {
		for (int i = 0; i < myN; ++i) {
			mySwarm[i] = new Firefly();
			mySwarm[i].randomize();
		}
	}

	private final void initializeMemory() {
		myEvals = 0;
		myIter = 0;
		tempD = new double[myD];
		tempK = new int[2];
		temp4 = new double[4];
		tempX = new double[3][myD];
		mySwarm = new Firefly[myN];
	}

	private final void updateFireflies() {

		// main loop of firefly update
		for (int i = 0; i < myN; ++i) {

			// choose a random index not equal to j
			final int random_j = sample1FromSwarm(i);

			// random attraction model
			if (mySwarm[i].intensity > mySwarm[random_j].intensity) {

				// compute the distance between fireflies i and j
				final double[] ffi = mySwarm[i].position;
				final double[] ffj = mySwarm[random_j].position;
				final double r = computeDistance(ffi, ffj);

				// update the step size alpha
				final double alpha = myStepSizeStrategy.updateStepSize(this, i);

				// compute the attraction coefficient beta
				double beta = (myBetaMax - myBetaMin) * Math.exp(-myGamma * r * r) + myBetaMin;
				beta = beta * myEvals / myMaxEvals;

				// move the firefly i closer to j
				mySwarm[i].pbest = ffi.clone();
				for (int k = 0; k < myD; ++k) {
					final double scale_k = myUpper[k] - myLower[k];
					final double noise = alpha * sampleNoise() * scale_k;
					ffi[k] = ffi[k] * (1.0 - beta) + ffj[k] * beta + noise;
				}

				// correct the coordinates of firefly i if out of bounds
				rectifyBounds(mySwarm[i].position);

				// update the fitness of firefly i
				mySwarm[i].updateIntensity();

				// update counters and check for termination
				if (myEvals >= myMaxEvals) {
					break;
				}
			} else {

				// neighborhood search
				if (myDoNeighborhoodSearch) {

					// generate local trial solution
					sample2FromNeighborhood(i);
					sample3Uniform();
					final double[] X = mySwarm[i].position;
					final double[] pbest = mySwarm[i].pbest;
					final double[] Xi1 = mySwarm[tempK[0]].position;
					final double[] Xi2 = mySwarm[tempK[1]].position;
					for (int j = 0; j < myD; ++j) {
						tempX[0][j] = temp4[0] * X[j] + temp4[1] * pbest[j] + temp4[2] * (Xi1[j] - Xi2[j]);
					}
					rectifyBounds(tempX[0]);

					// generate global trial solution
					sample2FromSwarm(i);
					sample3Uniform();
					final double[] gbest = myCurrentBest.position;
					final double[] Xi3 = mySwarm[tempK[0]].position;
					final double[] Xi4 = mySwarm[tempK[1]].position;
					for (int j = 0; j < myD; ++j) {
						tempX[1][j] = temp4[0] * X[j] + temp4[1] * gbest[j] + temp4[2] * (Xi3[j] - Xi4[j]);
					}
					rectifyBounds(tempX[1]);

					// generate Cauchy trial solution
					for (int j = 0; j < myD; ++j) {
						tempX[2][j] = X[j] + sampleCauchy();
					}
					rectifyBounds(tempX[2]);

					// calculate the fitness values of X1, X2 and X3
					temp4[0] = mySwarm[i].intensity;
					temp4[1] = myFunc.apply(tempX[0]);
					temp4[2] = myFunc.apply(tempX[1]);
					temp4[3] = myFunc.apply(tempX[2]);
					myEvals += 3;

					// select the best solution among X, X1, X2, X3 as the new X
					final int i_min = Sequences.argmin(4, temp4);
					if (i_min >= 1) {
						mySwarm[i].pbest = mySwarm[i].position.clone();
						mySwarm[i].position = tempX[i_min - 1].clone();
						mySwarm[i].intensity = temp4[i_min];
					}
				}
			}
		}

		// compute the dimmest and strongest firefly
		Firefly worst = null, best = null;
		for (final Firefly fly : mySwarm) {
			if (worst == null) {
				worst = fly;
			} else if (fly.intensity > worst.intensity) {
				worst = fly;
			}
			if (best == null) {
				best = fly;
			} else if (fly.intensity < best.intensity) {
				best = fly;
			}
		}

		// opposition-based update of the dimmest firefly
		if (myUseOpposition) {
			if (RAND.nextDouble() < myWorstToBestProb) {
				worst.position = best.position.clone();
				worst.pbest = best.pbest.clone();
				worst.alpha = best.alpha;
				worst.intensity = best.intensity;
			} else {
				worst.pbest = worst.position.clone();
				for (int i = 0; i < myD; ++i) {
					worst.position[i] = myLower[i] + myUpper[i] - worst.position[i];
				}
				worst.alpha = Double.NaN;
				worst.updateIntensity();
			}
		}
	}

	private final double computeDistance(final double[] x, final double[] y) {
		for (int i = 0; i < myD; ++i) {
			tempD[i] = x[i] - y[i];
		}
		return BlasMath.denorm(myD, tempD);
	}

	private final void rectifyBounds(final double[] pos) {
		for (int i = 0; i < myD; ++i) {
			if (pos[i] < myLower[i]) {
				pos[i] = myLower[i];
			} else if (pos[i] > myUpper[i]) {
				pos[i] = myUpper[i];
			}
		}
	}

	private final void updateStatistics() {
		myCurrentBest = null;
		myMinAlpha = 1.0;
		myMaxAlpha = 0.0;
		for (final Firefly fly : mySwarm) {
			if (myCurrentBest == null) {
				myCurrentBest = fly;
			} else {
				if (fly.intensity < myCurrentBest.intensity) {
					myCurrentBest = fly;
				}
			}
			final double alpha = fly.alpha;
			if (!Double.isNaN(alpha)) {
				myMinAlpha = Math.min(myMinAlpha, alpha);
				myMaxAlpha = Math.max(myMaxAlpha, alpha);
			}
		}
	}

	private final void sample2FromNeighborhood(final int i) {
		for (int idx = 0; idx < 2; ++idx) {
			int iidx = i;
			while ((idx == 0 && iidx == i) || (idx == 1 && (iidx == i || iidx == tempK[0]))) {
				final int r = RAND.nextInt(myK) + 1;
				if (RAND.nextBoolean()) {
					iidx = (i + r) % myN;
				} else {
					iidx = (i - r + myN) % myN;
				}
			}
			tempK[idx] = iidx;
		}
	}

	private final int sample1FromSwarm(final int i) {
		int j = i;
		while (j == i) {
			j = RAND.nextInt(myN);
		}
		return j;
	}

	private final void sample2FromSwarm(final int i) {
		for (int idx = 0; idx <= 1; ++idx) {
			int iidx = i;
			while ((idx == 0 && iidx == i) || (idx == 1 && (iidx == i || iidx == tempK[0]))) {
				iidx = RAND.nextInt(myN);
			}
			tempK[idx] = iidx;
		}
	}

	private final void sample3Uniform() {
		final double r1 = RAND.nextDouble();
		final double r2 = RAND.nextDouble();
		final double r3 = RAND.nextDouble();
		final double sum = r1 + r2 + r3;
		temp4[0] = r1 / sum;
		temp4[1] = r2 / sum;
		temp4[2] = r3 / sum;
	}

	private final double sampleNoise() {
		switch (myNoiseStrategy) {
		case UNIFORM:
			return RAND.nextDouble() - 0.5;
		case GAUSSIAN:
			return RAND.nextGaussian();
		case CAUCHY:
			return sampleCauchy();
		case NONE:
			return 0.0;
		default:
			return 0.0;
		}
	}

	private static double sampleCauchy() {
		return Math.tan(Math.PI * (RAND.nextDouble() - 0.5));
	}
}
