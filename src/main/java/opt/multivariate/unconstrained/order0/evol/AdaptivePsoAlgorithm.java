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

import opt.multivariate.unconstrained.order0.GradientFreeOptimizer;
import utils.BlasMath;
import utils.RealMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE
 * Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39,
 * Issue 6, 2009, Pages 1362-1381 (2009)
 */
public final class AdaptivePsoAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	private final class Particle {

		// local positional properties
		final double[] myPos;
		double myFit;
		final double[] myVel;
		final double[] myPBest;
		double myPBestFit;

		Particle(final double[] pos, final double[] vel) {
			myPos = pos;
			myVel = vel;
			myPBest = Arrays.copyOf(myPos, myD);
			myFit = myFunc.apply(myPos);
			myPBestFit = myFit;
		}

		final void update() {

			// update the velocity and position of this particle (1)-(2)
			for (int i = 0; i < myD; ++i) {
				final double r1 = RAND.nextDouble();
				final double r2 = RAND.nextDouble();
				myVel[i] = myVel[i] * myW + myC1 * r1 * (myPBest[i] - myPos[i]) + myC2 * r2 * (myGBest[i] - myPos[i]);
				myPos[i] += myVel[i];
			}

			// correct if out of box
			if (myCorrectInBox) {
				for (int i = 0; i < myD; ++i) {
					if (myPos[i] < myLower[i]) {
						myPos[i] = myLower[i];
					} else if (myPos[i] > myUpper[i]) {
						myPos[i] = myUpper[i];
					}
				}
			}

			// fitness re-evaluation and update best point so far
			myFit = myFunc.apply(myPos);
			++myEvals;
			if (myFit < myPBestFit) {
				System.arraycopy(myPos, 0, myPBest, 0, myD);
				myPBestFit = myFit;
			}
		}

		final void replace(final double pfit, final double[] p) {
			myFit = pfit;
			System.arraycopy(p, 0, myPos, 0, myD);
			if (myFit < myPBestFit) {
				System.arraycopy(myPos, 0, myPBest, 0, myD);
				myPBestFit = myFit;
			}
		}
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// algorithm parameters - specified by user or fixed
	private final double mySigmaMin = 0.1;
	private final double mySigmaMax = 1.0;
	private final double mySigmaTol;
	private final boolean myCorrectInBox;
	private final int myMaxEvals;
	private final int mySwarmSize;

	// algorithm parameters - adaptive
	private double myW, myC1, myC2;
	private int myIter;
	private int myState;
	private int myMaxIters;

	// swarm objects
	private Particle[] mySwarm;
	private double[] myGBest;
	private double myGBestFit;
	private int myIWorst;
	private double[] workp, works, workmu;

	// problem parameters
	private Function<? super double[], Double> myFunc;
	private double[] myLower;
	private double[] myUpper;
	private int myD;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param maxEvals
	 * @param swarmSize
	 * @param boundsCorrection
	 */
	public AdaptivePsoAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvals,
			final int swarmSize, final boolean boundsCorrection) {
		super(tolerance);
		mySigmaTol = stdevTolerance;
		mySwarmSize = swarmSize;
		myMaxEvals = maxEvals;
		myCorrectInBox = boundsCorrection;
	}

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param maxEvals
	 * @param swarmSize
	 */
	public AdaptivePsoAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvals,
			final int swarmSize) {
		this(tolerance, stdevTolerance, maxEvals, swarmSize, true);
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
	public double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, lo, hi);
	}

	@Override
	public void iterate() {

		// perform elitist learning
		updateGlobalBest(workp, myIter, myMaxIters);

		// compute the evolutionary factor
		final double f = getF(works);
		final int state = nextState(f, workmu, myState);

		// get the next state of exploration and update swarm parameters
		updateParams(f, state);

		// update swarm
		updateSwarm();

		// update counters
		myState = state;
		++myIter;
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
	public final void initialize(final Function<? super double[], Double> func, final double[] lower,
			final double[] upper) {

		// set problem
		myFunc = func;
		myD = lower.length;
		myLower = lower;
		myUpper = upper;

		// initialize parameters
		myW = 0.9;
		myC1 = myC2 = 2.0;
		myIter = myState = 0;
		myMaxIters = RealMath.roundInt(myMaxEvals / (1.0 + mySwarmSize));

		// initialize swarm
		mySwarm = new Particle[mySwarmSize];
		myGBestFit = Double.POSITIVE_INFINITY;
		int ibest = myIWorst = 0;
		for (int i = 0; i < mySwarm.length; ++i) {

			// create particle
			final double[] pos = new double[myD];
			final double[] vel = new double[myD];
			for (int j = 0; j < myD; ++j) {
				final double r1 = RAND.nextDouble();
				pos[j] = myLower[j] + (myUpper[j] - myLower[j]) * r1;
			}
			mySwarm[i] = new Particle(pos, vel);

			// update best and worst positions
			if (mySwarm[i].myFit < myGBestFit) {
				myGBestFit = mySwarm[i].myFit;
				ibest = i;
			}
			if (mySwarm[i].myFit >= mySwarm[myIWorst].myFit) {
				myIWorst = i;
			}
		}
		myGBest = Arrays.copyOf(mySwarm[ibest].myPos, myD);

		// initialize work arrays
		workp = new double[myD];
		works = new double[mySwarm.length];
		workmu = new double[4];
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
		while (myIter < myMaxIters) {

			// perform a single generation
			iterate();

			// converge when distance in fitness between best and worst points
			// is below the given tolerance
			final double distY = Math.abs(myGBestFit - mySwarm[myIWorst].myFit);
			final double avgY = 0.5 * (myGBestFit + mySwarm[myIWorst].myFit);
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
				if (m2 <= (mySwarmSize - 1) * mySigmaTol * mySigmaTol) {
					break;
				}
			}
		}
		return myGBest;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void updateGlobalBest(final double[] p, final int it, final int itmax) {

		// this subprocedure is based on Figure 7
		// set P = gbest;
		System.arraycopy(myGBest, 0, p, 0, myD);

		// perturb P(d)
		final int d = RAND.nextInt(myD);
		final double sigma = mySigmaMax - (mySigmaMax - mySigmaMin) * it / itmax;
		final double gaus = RAND.nextGaussian() * sigma;
		p[d] += (myUpper[d] - myLower[d]) * gaus;

		// make sure P is in the range
		if (myCorrectInBox) {
			if (p[d] < myLower[d]) {
				p[d] = myLower[d];
			} else if (p[d] > myUpper[d]) {
				p[d] = myUpper[d];
			}
		}

		// evaluate P and replace best or worst point if necessary
		final double nu = myFunc.apply(p);
		++myEvals;
		if (nu < myGBestFit) {
			System.arraycopy(p, 0, myGBest, 0, myD);
			myGBestFit = nu;
		} else {
			mySwarm[myIWorst].replace(nu, p);
		}
	}

	private void updateSwarm() {

		// update the swarm
		for (final Particle p : mySwarm) {
			p.update();
		}

		// compute the new global best and worst
		int ibest = -1;
		myIWorst = 0;
		myGBestFit = Double.POSITIVE_INFINITY;
		for (int i = 0; i < mySwarm.length; ++i) {
			if (mySwarm[i].myFit <= myGBestFit) {
				myGBestFit = mySwarm[i].myFit;
				ibest = i;
			}
			if (mySwarm[i].myFit >= mySwarm[myIWorst].myFit) {
				myIWorst = i;
			}
		}

		// if mutation resulted in improvement in best position record it
		if (ibest >= 0) {
			System.arraycopy(mySwarm[ibest].myPos, 0, myGBest, 0, myD);
		}
	}

	private void updateParams(final double f, final int state) {

		// update w in (10)
		myW = 1.0 / (1.0 + 1.5 * Math.exp(-2.6 * f));

		// update C1 and C2 in (11)-(12)
		final double delta1 = 0.05 * (1.0 + RAND.nextDouble());
		final double delta2 = 0.05 * (1.0 + RAND.nextDouble());
		switch (state) {
		case 1:
			myC1 += delta1;
			myC2 += delta2;
			break;
		case 2:
			myC1 += 0.5 * delta1;
			myC2 -= 0.5 * delta2;
			break;
		case 3:
			myC1 += 0.5 * delta1;
			myC2 += 0.5 * delta2;
			break;
		default:
			myC1 -= 0.5 * delta1;
			myC2 -= 0.5 * delta2;
			break;
		}
		myC1 = Math.max(1.5, Math.min(myC1, 2.5));
		myC2 = Math.max(1.5, Math.min(myC2, 2.5));
		if (myC1 + myC2 > 4.0) {
			final double sum = myC1 + myC2;
			myC1 *= 4.0 / sum;
			myC2 *= 4.0 / sum;
		}
	}

	private double getF(final double[] d) {

		// calculate the distances between the particles (7)
		final int n = mySwarm.length;
		double dmin = 0.0;
		double dmax = Double.POSITIVE_INFINITY;
		int ibest = 0;
		for (int i = 0; i < n; ++i) {
			d[i] = 0.0;
			for (int j = 0; j < n; ++j) {
				if (j != i) {
					double distij = 0.0;
					final double[] xi = mySwarm[i].myPos;
					final double[] xj = mySwarm[j].myPos;
					for (int k = 0; k < myD; ++k) {
						final double dist = xi[k] - xj[k];
						distij += dist * dist;
					}
					distij = Math.sqrt(distij);
					d[i] += distij;
				}
			}
			d[i] /= (n - 1.0);

			// update the least and greatest distance
			if (d[i] > dmax) {
				dmax = d[i];
			}
			if (d[i] < dmin) {
				dmin = d[i];
			}

			// search for the best point in swarm
			if (mySwarm[i].myFit < mySwarm[ibest].myFit) {
				ibest = i;
			}
		}

		// compute the evolutionary factory in (8)
		return (d[ibest] - dmin) / Math.max(dmax - dmin, 1e-8);
	}

	private static int nextState(final double f, final double[] mu, final int oldState) {

		// compute the decision regions for the next state
		final double m1 = mu1(f);
		final double m2 = mu2(f);
		final double m3 = mu3(f);
		final double m4 = mu4(f);
		mu[0] = m1;
		mu[1] = m2;
		mu[2] = m3;
		mu[3] = m4;
		switch (oldState) {
		case 0: {
			return argmax(mu) + 1;
		}
		case 1: {
			if (m1 > 0) {
				return 1;
			} else if (m2 > 0) {
				return 2;
			} else if (m4 > 0) {
				return 4;
			} else {
				return 3;
			}
		}
		case 2: {
			if (m2 > 0) {
				return 2;
			} else if (m3 > 0) {
				return 3;
			} else if (m1 > 0) {
				return 1;
			} else {
				return 4;
			}
		}
		case 3: {
			if (m3 > 0) {
				return 3;
			} else if (m4 > 0) {
				return 4;
			} else if (m2 > 0) {
				return 2;
			} else {
				return 1;
			}
		}
		default: {
			if (m4 > 0) {
				return 4;
			} else if (m1 > 0) {
				return 1;
			} else if (m2 > 0) {
				return 2;
			} else {
				return 3;
			}
		}
		}
	}

	private static double mu1(final double f) {
		if (f >= 0.0 && f <= 0.4) {
			return 0.0;
		} else if (f > 0.4 && f <= 0.6) {
			return 5.0 * f - 2.0;
		} else if (f > 0.6 && f <= 0.7) {
			return 1.0;
		} else if (f > 0.7 && f <= 0.8) {
			return -10.0 * f + 8.0;
		} else {
			return 0.0;
		}
	}

	private static double mu2(final double f) {
		if (f >= 0.0 && f <= 0.2) {
			return 0.0;
		} else if (f > 0.2 && f <= 0.3) {
			return 10.0 * f - 2.0;
		} else if (f > 0.3 && f <= 0.4) {
			return 1.0;
		} else if (f > 0.4 && f <= 0.6) {
			return -5.0 * f + 3.0;
		} else {
			return 0.0;
		}
	}

	private static double mu3(final double f) {
		if (f >= 0.0 && f <= 0.1) {
			return 1.0;
		} else if (f > 0.1 && f <= 0.3) {
			return -5.0 * f + 1.5;
		} else {
			return 0.0;
		}
	}

	private static double mu4(final double f) {
		if (f >= 0.0 && f <= 0.7) {
			return 0.0;
		} else if (f > 0.7 && f <= 0.9) {
			return 5.0 * f - 3.5;
		} else {
			return 1.0;
		}
	}

	private static final int argmax(final double... data) {
		int k = 0;
		int imax = -1;
		double max = 0.0;
		for (final double t : data) {
			if (k == 0 || t > max) {
				max = t;
				imax = k;
			}
			++k;
		}
		return imax;
	}
}
