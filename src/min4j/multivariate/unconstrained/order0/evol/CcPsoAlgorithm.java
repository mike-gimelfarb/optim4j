package min4j.multivariate.unconstrained.order0.evol;

import java.util.function.Function;

import min4j.multivariate.unconstrained.order0.GradientFreeOptimizer;
import min4j.utils.BlasMath;
import min4j.utils.Constants;

/**
 * Li, Xiaodong, and Xin Yao. "Cooperatively coevolving particle swarms for
 * large scale optimization." IEEE Transactions on Evolutionary Computation 16.2
 * (2012): 210-224.
 * 
 * @author Michael
 *
 */
public class CcPsoAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// function properties
	private Function<? super double[], Double> myFunc;
	private double[] myLower;
	private double[] myUpper;
	private int myD;

	// algorithm parameters
	private final boolean myApplyBoundsConstr;
	private final int myMaxEvals;
	private final int mySwarmSize;
	private final double mySigmaTol;
	private int mySwarmCount;
	private int myGenr;
	private int myCompsPerSwarm;
	private int myIs;
	private int[] myS;

	// temporary storage for the swarm
	private int[][] myK;
	private double[][] myPos;
	private double[][] myPersBestPos;
	private double[][] myPersBestFit;
	private double[][] myLocalBestPos;
	private double[] mySwarmBestPos;
	private double[] myBestPos;
	private double myBestFit;
	private double[] myWork;
	private boolean[][] mySampledCauchy;

	// temporary storage for the swarm topology
	private int[] myTopIndices;
	private double[] myTopFit;

	// data for adaptive parameter updates
	private int myUpdateFreq;
	private int myCSucc, myCFail, myGSucc, myGFail;
	private double myF;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 * 
	 * 
	 * @param tolerance
	 * @param stdevTolerance
	 * @param maxEvaluations
	 * @param particlesPerSwarm
	 * @param partitionSizes
	 * @param applyBoundsConstraints
	 * @param updateParamsEveryGens
	 */
	public CcPsoAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvaluations,
			final int particlesPerSwarm, final int[] partitionSizes, final boolean applyBoundsConstraints,
			final int updateParamsEveryGens) {
		super(tolerance);
		mySigmaTol = stdevTolerance;
		mySwarmSize = particlesPerSwarm;
		myMaxEvals = maxEvaluations;
		myS = partitionSizes;
		myApplyBoundsConstr = applyBoundsConstraints;
		myUpdateFreq = updateParamsEveryGens;
	}

	/**
	 * @param tolerance
	 * @params stdevTolerance
	 * @param maxEvaluations
	 * @param particlesPerSwarm
	 * @param partitionSizes
	 * @param applyBoundsConstraints
	 */
	public CcPsoAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvaluations,
			final int particlesPerSwarm, final int[] partitionSizes, final boolean applyBoundsConstraints) {
		this(tolerance, stdevTolerance, maxEvaluations, particlesPerSwarm, partitionSizes, applyBoundsConstraints, 30);
	}

	/**
	 * @param stdevTolerance
	 * @param tolerance
	 * @param maxEvaluations
	 * @param particlesPerSwarm
	 * @param partitionSizes
	 */
	public CcPsoAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvaluations,
			final int particlesPerSwarm, final int[] partitionSizes) {
		this(tolerance, stdevTolerance, maxEvaluations, particlesPerSwarm, partitionSizes, true);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
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

		// save the old best fitness value of swarms to track improvement
		final double myOldBestFit = myBestFit;

		// update each swarm's personal bests
		for (int is = 0; is < mySwarmCount; ++is) {
			updateSwarm(is);
		}

		// update each swarm's particle positions
		for (int is = 0; is < mySwarmCount; ++is) {
			updatePositions(is);
		}

		// check if a randomization of the components is required
		if (myGenr > 0 && myBestFit == myOldBestFit) {
			randomizeComponents();
		}

		// check if we need to reset the counters for exploration
		updateParameters();
	}

	@Override
	public double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, lo, hi);
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 * 
	 * @param func
	 * @param lower
	 * @param upper
	 */
	public void initialize(final Function<? super double[], Double> func, final double[] lower, final double[] upper) {

		// initialize domain
		myFunc = func;
		myLower = lower;
		myUpper = upper;
		myD = lower.length;

		// uses a ring topology
		myTopIndices = new int[3];
		myTopFit = new double[3];

		// initialize adaptive params
		myCSucc = myCFail = myGSucc = myGFail = 0;
		myF = 0.5;

		// initialize algorithm primitives and initialize swarms' components
		myEvals = myGenr = 0;
		randomizeComponents();

		// initialize the swarms
		myPos = new double[mySwarmSize][myD];
		myPersBestPos = new double[mySwarmSize][myD];
		myLocalBestPos = new double[mySwarmSize][];
		mySwarmBestPos = new double[myD];
		myBestPos = new double[myD];
		myWork = new double[myD];
		randomizeSwarmPositions();
	}

	/**
	 * 
	 * @param func
	 * @param lower
	 * @param upper
	 * @return
	 */
	public double[] optimize(final Function<? super double[], Double> func, final double[] lower,
			final double[] upper) {

		// initialization
		initialize(func, lower, upper);

		// main loop
		for (myGenr = 0; myGenr < Integer.MAX_VALUE; ++myGenr) {
			iterate();

			// check max number of evaluations
			if (myEvals >= myMaxEvals) {
				break;
			}

			// converge when distance in fitness between best and worst points
			// is below the given tolerance
			double bestFit = Double.POSITIVE_INFINITY;
			double worstFit = Double.NEGATIVE_INFINITY;
			for (int is = 0; is < mySwarmCount; ++is) {
				for (int ip = 0; ip < mySwarmSize; ++ip) {
					final double fit = myPersBestFit[is][ip];
					bestFit = Math.min(bestFit, fit);
					worstFit = Math.max(worstFit, fit);
				}
			}
			final double distY = Math.abs(bestFit - worstFit);
			final double avgY = 0.5 * (bestFit + worstFit);
			if (distY <= myTol + RELEPS * Math.abs(avgY)) {

				// compute standard deviation of swarm radiuses
				int count = 0;
				double mean = 0.0;
				double m2 = 0.0;
				for (final double[] pt : myPersBestPos) {
					final double x = BlasMath.denorm(myD, pt);
					++count;
					final double delta = x - mean;
					mean += delta / count;
					final double delta2 = x - mean;
					m2 += delta * delta2;
				}

				// test convergence in standard deviation
				if (m2 <= (mySwarmSize - 1) * mySigmaTol * mySigmaTol) {
					break;
				}
			}
		}
		return myBestPos;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void randomizeComponents() {

		// sample an s at random, the number of components per swarm
		myIs = RAND.nextInt(myS.length);
		myCompsPerSwarm = myS[myIs];
		mySwarmCount = myD / myCompsPerSwarm;

		// re-dimension all arrays
		myK = new int[mySwarmCount][myCompsPerSwarm];
		myPersBestFit = new double[mySwarmCount][mySwarmSize];
		mySampledCauchy = new boolean[mySwarmCount][mySwarmSize];

		// initialize the component indices for each swarm
		final int[] range = range(myD);
		shuffle(range);
		int i = 0;
		for (final int[] k : myK) {
			for (int j = 0; j < myCompsPerSwarm; ++j) {
				k[j] = range[i];
				++i;
			}
		}
	}

	private void randomizeSwarmPositions() {

		// initialize the particles in all swarms in range [lb, ub]
		for (int ip = 0; ip < mySwarmSize; ++ip) {
			for (int i = 0; i < myD; ++i) {
				final double c = RAND.nextDouble();
				myPos[ip][i] = myLower[i] + c * (myUpper[i] - myLower[i]);
			}
			System.arraycopy(myPos[ip], 0, myPersBestPos[ip], 0, myD);
		}

		// compute the fitness of all particles and get the global best particle
		myBestFit = Double.POSITIVE_INFINITY;
		int bestip = -1;
		for (int ip = 0; ip < mySwarmSize; ++ip) {
			final double fit = myFunc.apply(myPos[ip]);
			++myEvals;
			if (fit < myBestFit) {
				myBestFit = fit;
				bestip = ip;
			}
		}

		// set the swarm's best positions and global best position
		System.arraycopy(myPos[bestip], 0, mySwarmBestPos, 0, myD);
		System.arraycopy(myPos[bestip], 0, myBestPos, 0, myD);

	}

	private void updateSwarm(final int is) {

		// update particle personal bests
		double fPyhat = evaluate(is, mySwarmBestPos);
		for (int ip = 0; ip < mySwarmSize; ++ip) {

			// compute fitness
			final double fPx = evaluate(is, myPos[ip]);
			myPersBestFit[is][ip] = evaluate(is, myPersBestPos[ip]);

			// perform update of the personal best
			if (fPx < myPersBestFit[is][ip]) {
				for (final int i : myK[is]) {
					myPersBestPos[ip][i] = myPos[ip][i];
				}
				myPersBestFit[is][ip] = fPx;

				// update strategy probabilities for Cauchy/Gaussian sampling
				if (mySampledCauchy[is][ip]) {
					++myCSucc;
				} else {
					++myGSucc;
				}
			} else {

				// update strategy probabilities for Cauchy/Gaussian sampling
				if (mySampledCauchy[is][ip]) {
					++myCFail;
				} else {
					++myGFail;
				}
			}

			// perform update of the swarm best
			if (myPersBestFit[is][ip] < fPyhat) {
				for (final int i : myK[is]) {
					mySwarmBestPos[i] = myPersBestPos[ip][i];
				}
				fPyhat = myPersBestFit[is][ip];
			}
		}

		// update particle local best positions
		for (int ip = 0; ip < mySwarmSize; ++ip) {

			// get the fitness values in neighborhood of particle ip
			myTopIndices[0] = (ip - 1 + mySwarmSize) % mySwarmSize;
			myTopIndices[1] = ip;
			myTopIndices[2] = (ip + 1) % mySwarmSize;
			for (int i = 0; i < 3; ++i) {
				myTopFit[i] = myPersBestFit[is][myTopIndices[i]];
			}

			// get the best local particle among neighbors
			final int imin = argmin(3, myTopFit);
			myLocalBestPos[ip] = myPersBestPos[myTopIndices[imin]];
		}

		// update global best vector position and fitness
		if (fPyhat < myBestFit) {
			for (final int i : myK[is]) {
				myBestPos[i] = mySwarmBestPos[i];
			}
			myBestFit = myFunc.apply(myBestPos);
			++myEvals;
		}
	}

	private void updatePositions(final int is) {
		for (int ip = 0; ip < mySwarmSize; ++ip) {

			// decide whether the next sample will come from a Cauchy or Gaussian
			// distribution
			final double rand = RAND.nextDouble();
			final boolean cauchy = rand <= myF;

			// evolve the particle
			if (cauchy) {
				for (final int i : myK[is]) {
					final double c = cauchy();
					final double dist = myPersBestPos[ip][i] - myLocalBestPos[ip][i];
					myPos[ip][i] = myPersBestPos[ip][i] + c * Math.abs(dist);
				}
			} else {
				for (final int i : myK[is]) {
					final double c = RAND.nextGaussian();
					final double dist = myPersBestPos[ip][i] - myLocalBestPos[ip][i];
					myPos[ip][i] = myLocalBestPos[ip][i] + c * Math.abs(dist);
				}
			}
			mySampledCauchy[is][ip] = cauchy;

			// apply bounds constraint
			if (myApplyBoundsConstr) {
				for (final int i : myK[is]) {
					if (myPos[ip][i] < myLower[i] || myPos[ip][i] > myUpper[i]) {
						final double c = RAND.nextDouble();
						myPos[ip][i] = myLower[i] + c * (myUpper[i] - myLower[i]);
					}
				}
			}
		}
	}

	private double evaluate(final int is, final double[] z) {

		// cache the swarm best position currently for swarm is
		// then change the component values for swarm is to z
		for (final int i : myK[is]) {
			myWork[i] = mySwarmBestPos[i];
			mySwarmBestPos[i] = z[i];
		}

		// evaluate function at the modified vector
		final double fit = myFunc.apply(mySwarmBestPos);
		++myEvals;

		// restore the swarm best position
		for (final int i : myK[is]) {
			mySwarmBestPos[i] = myWork[i];
		}
		return fit;
	}

	private void updateParameters() {
		if (myGenr >= myUpdateFreq && myGenr % myUpdateFreq == 0) {

			// update F
			if (myCSucc + myCFail > 0 && myGSucc + myGFail > 0 && myCSucc > 0 && myGSucc > 0) {
				final double p1 = ((double) myCSucc) / (myCSucc + myCFail);
				final double p2 = ((double) myGSucc) / (myGSucc + myGFail);
				myF = p1 / (p1 + p2);
				myF = Math.max(0.05, Math.min(myF, 0.95));
			}

			// reset counters
			myCSucc = myCFail = myGSucc = myGFail = 0;
		}
	}

	private static double cauchy() {
		return Math.tan(Constants.PI * (RAND.nextDouble() - 0.5));
	}

	private static final void shuffle(final int... arr) {
		for (int i = arr.length - 1; i > 0; --i) {
			final int index = RAND.nextInt(i + 1);
			swap(arr, index, i);
		}
	}

	private static final void swap(final int[] data, final int i, final int j) {
		if (i == j) {
			return;
		}
		final int temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}

	private static final int[] range(final int end) {
		final int[] result = new int[end];
		for (int i = 0; i < end; ++i) {
			result[i] = i;
		}
		return result;
	}

	private static final int argmin(final int len, final double... data) {
		int k = 0;
		int imin = -1;
		double min = 0;
		for (final double t : data) {
			if (k >= len) {
				break;
			}
			if (k == 0 || t < min) {
				min = t;
				imin = k;
			}
			++k;
		}
		return imin;
	}
}
