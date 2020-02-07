package opt.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.unconstrained.order0.GradientFreeOptimizer;
import utils.BlasMath;

/**
 * Implemented according to "A Competitive Swarm Optimizer for Large Scale
 * Optimization" by Cheng and Jin, 2015
 *
 * Uses an optional local ring topology (e.g. see "Population Structure and
 * Particle Swarm Performance" by Kennedy and Mendes)
 *
 * @author Michael
 */
public final class CsoAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	private final class Particle {

		// local positional properties
		final double[] myPos;
		final double[] myVel;
		double[] myMean;
		double myFit;

		// references to neighbors in right topology
		Particle myLeft;
		Particle myRight;

		Particle(final double[] pos, final double[] vel) {
			myPos = pos;
			myVel = vel;
			myFit = myFunc.apply(myPos);
		}

		final void competeWith(final Particle other) {

			// find the loser
			final Particle loser;
			final Particle winner;
			if (myFit > other.myFit) {
				loser = this;
				winner = other;
			} else {
				loser = other;
				winner = this;
			}

			// update velocity and position of the loser: equations (6) and (7)
			for (int i = 0; i < myD; ++i) {

				// velocity update (6)
				final double r1 = RAND.nextDouble();
				final double r2 = RAND.nextDouble();
				final double r3 = RAND.nextDouble();
				loser.myVel[i] = r1 * loser.myVel[i] + r2 * (winner.myPos[i] - loser.myPos[i])
						+ myPhi * r3 * (loser.myMean[i] - loser.myPos[i]);

				// clip velocity
				final double range = myUpper[i] - myLower[i];
				final double maxv = 0.2 * range;
				loser.myVel[i] = Math.max(-maxv, Math.min(maxv, loser.myVel[i]));

				// position update: equation (7)
				loser.myPos[i] += loser.myVel[i];
			}

			// correct if out of box
			if (myCorrectInBox) {
				for (int i = 0; i < myD; ++i) {
					if (loser.myPos[i] < myLower[i]) {
						loser.myPos[i] = myLower[i];
					} else if (loser.myPos[i] > myUpper[i]) {
						loser.myPos[i] = myUpper[i];
					}
				}
			}

			// update the fitness of the loser
			loser.myFit = myFunc.apply(loser.myPos);
		}
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// model parameters
	private final boolean myUseRingTopology;
	private final boolean myCorrectInBox;
	private final double myPhi;
	private final double mySigmaTol;
	private final int mySize;
	private final int myMaxEvals;
	private final Particle[] mySwarm;

	// problem parameters
	private Function<? super double[], Double> myFunc;
	private int myD;
	private double[] myMean;
	private double[] myLower;
	private double[] myUpper;
	private Particle myBest;
	private Particle myWorst;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param swarmSize
	 * @param stdevTolerance
	 * @param tolerance
	 * @param phi
	 * @param maxEvaluations
	 * @param useRingTopology
	 * @param correctOutOfBounds
	 */
	public CsoAlgorithm(final double tolerance, final double stdevTolerance, final int swarmSize,
			final int maxEvaluations, final double phi, final boolean useRingTopology,
			final boolean correctOutOfBounds) {
		super(tolerance);
		mySigmaTol = stdevTolerance;
		mySize = ((swarmSize & 1) == 0) ? swarmSize : swarmSize + 1;
		mySwarm = new Particle[mySize];
		myPhi = phi;
		myMaxEvals = maxEvaluations;
		myUseRingTopology = useRingTopology;
		myCorrectInBox = correctOutOfBounds;
	}

	/**
	 *
	 * @param swarmSize
	 * @param stdevTolerance
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public CsoAlgorithm(final double tolerance, final double stdevTolerance, final int swarmSize,
			final int maxEvaluations) {
		this(tolerance, stdevTolerance, swarmSize, maxEvaluations, getPhi(swarmSize), false, false);
	}

	private static double getPhi(final int m) {

		// this is equation (25)
		if (m <= 100) {
			return 0.0;
		}

		// this is equations (25) and (26): take midpoint between hi and lo
		final double phimin;
		final double phimax;
		if (m <= 200) {
			phimin = 0.0;
			phimax = 0.1;
		} else if (m <= 400) {
			phimin = 0.1;
			phimax = 0.2;
		} else if (m <= 600) {
			phimin = 0.1;
			phimax = 0.2;
		} else {
			phimin = 0.1;
			phimax = 0.3;
		}
		return 0.5 * (phimin + phimax);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		initialize(func, lo, hi);
	}

	@Override
	public final double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, lo, hi);
	}

	@Override
	public final void iterate() {

		// split m particles in the swarm into pairs:
		// shuffle the swarm and assign element i to m/2 + i
		shuffle(mySwarm);

		// now go through each pairing and perform fitness selection
		final int halfm = mySize >>> 1;
		for (int i = 0; i < halfm; ++i) {
			final int j = i + halfm;
			mySwarm[i].competeWith(mySwarm[j]);
		}
		myEvals += halfm;

		// update means based on neighbors topology
		if (myUseRingTopology) {
			for (final Particle p : mySwarm) {
				for (int i = 0; i < myD; ++i) {
					p.myMean[i] = (p.myLeft.myPos[i] + p.myPos[i] + p.myRight.myPos[i]) / 3.0;
				}
			}
		} else {
			Arrays.fill(myMean, 0.0);
			for (final Particle p : mySwarm) {
				for (int i = 0; i < myD; ++i) {
					myMean[i] += p.myPos[i] / mySize;
				}
			}
		}

		// find the best and worst points
		myBest = mySwarm[0];
		myWorst = mySwarm[0];
		for (final Particle p : mySwarm) {
			if (p.myFit <= myBest.myFit) {
				myBest = p;
			}
			if (p.myFit >= myWorst.myFit) {
				myWorst = p;
			}
		}
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param func
	 * @param lb
	 * @param ub
	 */
	public final void initialize(final Function<? super double[], Double> func, final double[] lb, final double[] ub) {

		// initialize function
		myFunc = func;
		myD = lb.length;
		myLower = lb;
		myUpper = ub;
		myEvals = 0;

		// initialize swarm
		for (int i = 0; i < mySize; ++i) {
			final double[] x = new double[myD];
			final double[] v = new double[myD];
			for (int j = 0; j < myD; ++j) {

				// randomly initialize position within the search space
				final double r = RAND.nextDouble();
				x[j] = (myUpper[j] - myLower[j]) * r + myLower[j];

				// set velocity initially to zero to reduce the chance the
				// particle leaves the boundary in subsequent iterations
				v[j] = 0.0;
			}
			mySwarm[i] = new Particle(x, v);
		}
		myEvals += mySize;

		// initialize topology to ring or dense topology
		if (myUseRingTopology) {
			for (int i = 0; i < mySize; ++i) {
				final int il = i == 0 ? mySize - 1 : i - 1;
				final int ir = i == mySize - 1 ? 0 : i + 1;
				mySwarm[i].myLeft = mySwarm[il];
				mySwarm[i].myRight = mySwarm[ir];
				mySwarm[i].myMean = new double[myD];
				for (int j = 0; j < myD; ++j) {
					mySwarm[i].myMean[j] = (mySwarm[i].myLeft.myPos[j] + mySwarm[i].myPos[j]
							+ mySwarm[i].myRight.myPos[j]) / 3.0;
				}
			}
		} else {
			myMean = new double[myD];
			for (final Particle p : mySwarm) {
				for (int i = 0; i < myD; ++i) {
					myMean[i] += p.myPos[i] / mySize;
				}
				p.myMean = myMean;
			}
		}
	}

	/**
	 *
	 * @param func
	 * @param lb
	 * @param ub
	 * @return
	 */
	public final double[] optimize(final Function<? super double[], Double> func, final double[] lb,
			final double[] ub) {

		// initialize parameters
		initialize(func, lb, ub);

		// main iteration loop over generations
		while (myEvals < myMaxEvals) {

			// perform a single generation
			iterate();

			// converge when distance in fitness between best and worst points
			// is below the given tolerance
			final double distY = Math.abs(myBest.myFit - myWorst.myFit);
			final double avgY = 0.5 * (myBest.myFit + myWorst.myFit);
			if (distY <= myTol + RELEPS * Math.abs(avgY)) {

				// compute standard deviation of swarm radiuses
				final int D = lb.length;
				int count = 0;
				double mean = 0.0;
				double m2 = 0.0;
				for (final Particle pt : mySwarm) {
					final double x = BlasMath.denorm(D, pt.myPos);
					++count;
					final double delta = x - mean;
					mean += delta / count;
					final double delta2 = x - mean;
					m2 += delta * delta2;
				}

				// test convergence in standard deviation
				if (m2 <= (mySize - 1) * mySigmaTol * mySigmaTol) {
					break;
				}
			}
		}
		return myBest == null ? null : myBest.myPos;
	}

	@SafeVarargs
	private static final <T> void shuffle(final T... arr) {
		for (int i = arr.length - 1; i > 0; --i) {
			final int index = RAND.nextInt(i + 1);
			swap(arr, index, i);
		}
	}

	private static final <T> void swap(final T[] data, final int i, final int j) {
		if (i == j) {
			return;
		}
		final T temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}
}
