/*
Copyright (c) 2012, Pinar Civicioglu
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, 
 this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the copyright notice,
 this list of conditions and the following disclaimer in the documentation 
 and/or other materials provided with the distribution
      
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
package opt.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.GradientFreeOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.BlasMath;
import utils.Sequences;

//%
//%
//% DIFFERENTIAL SEARCH ALGORITHM (DSA) (in MATLAB)
//% STANDARD VERSION of DSA (16.July.2013)
//%
//%
//% usage : > ds(method,fnc,mydata,popsize,dim,low,up,maxcycle)
//%
//% method
//%--------------------------------------------------------------
//% 1: Bijective DSA (B-DSA)
//% 2: Surjective DSA (S-DSA)
//% 3: Elitist DSA (strategy 1) (E1-DSA)
//% 4: Elitist DSA (strategy 2) (E2-DSA)
//% if method=[. . . ...],   Hybrid-DSA (H-DSA)
//%--------------------------------------------------------------
//% example : 
//% ds(1,'circlefit',mydata,10,3,-10,10,2000) ; % B-DSA
//% ds(2,'circlefit',mydata,10,3,-10,10,2000) ; % S-DSA
//% ds(3,'circlefit',mydata,10,3,-10,10,2000) ; % E1-DSA
//% ds(4,'circlefit',mydata,10,3,-10,10,2000) ; % E2-DSA
//% ds([1 2],'circlefit',mydata,10,3,-10,10,2000) ; % Hybrid-DSA, 
//in this case B-DSA and S-DSA are hybridized.
//%--------------------------------------------------------------
//% Please cite this article as;
//% P.Civicioglu, "Transforming geocentric cartesian coordinates to geodetic 
//  coordinates by using differential search algorithm",  Computers & Geosciences,
//  46 (2012), 229-247.
//% P.Civicioglu, "Understanding the nature of evolutionary search algorithms", 
//  Additional technical report for the project of 110Y309-Tubitak,2013.

/**
 *
 */
public final class DifferentialSearchAlgorithm extends GradientFreeOptimizer {

	// algorithm parameters
	private final int mySwarmSize, myMaxEvals;
	private final int[] myMethods;
	private final double mySigmaTol;

	// problem parameters
	private Function<? super double[], Double> myFunc;
	private int myD;
	private double[] myLower, myUpper;

	// storage and temporaries
	private Integer[] jind;
	private int[][] map;
	private double[][] superorganism, stopover, direction;
	private double[] fit_super, fit_stopover;
	private int myEvals;

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param maxEvaluations
	 * @param swarmSize
	 * @param methods
	 */
	public DifferentialSearchAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvaluations,
			final int swarmSize, final int[] methods) {
		super(tolerance);
		mySigmaTol = stdevTolerance;
		mySwarmSize = swarmSize;
		myMaxEvals = maxEvaluations;
		myMethods = methods;
	}

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param maxEvaluations
	 * @param swarmSize
	 */
	public DifferentialSearchAlgorithm(final double tolerance, final double stdevTolerance, final int maxEvaluations,
			final int swarmSize) {
		this(tolerance, stdevTolerance, maxEvaluations, swarmSize, new int[] { 1, 2, 3, 4 });
	}

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

		// SETTING OF ALGORITHMIC CONTROL PARAMETERS
		// Trial-pattern generation strategy for morphogenesis;
		// 'one-or-more morphogenesis'. (DEFAULT)
		final double p1 = 0.3 * RAND.nextDouble();
		final double p2 = 0.3 * RAND.nextDouble();

		// sample a method
		final int imethd = RAND.nextInt(myMethods.length);
		final int methd = myMethods[imethd];

		// search direction
		generate_dir(direction, methd, superorganism, mySwarmSize, fit_super, jind);

		// mutation strategy matrix
		generate_map(map, mySwarmSize, myD, p1, p2);

		// Recommended Methods for generation of Scale-Factor; R
		// R=4*randn; % brownian walk
		// R=4*randg; % brownian walk
		// R=lognrnd(rand,5*rand); % brownian walk
		// R=1/normrnd(0,5); % pseudo-stable walk
		// we use pseudo-stable walk
		final double R = 1.0 / (-2.0 * Math.log(RAND.nextDouble()));

		// bio-interaction (morphogenesis)
		for (int i = 0; i < mySwarmSize; ++i) {
			for (int j = 0; j < myD; ++j) {
				stopover[i][j] = superorganism[i][j] + R * map[i][j] * (direction[i][j] - superorganism[i][j]);
			}
		}

		// Boundary Control
		update(stopover, myLower, myUpper);

		// Selection-II
		for (int i = 0; i < mySwarmSize; ++i) {
			fit_stopover[i] = myFunc.apply(stopover[i]);
			if (fit_stopover[i] < fit_super[i]) {
				fit_super[i] = fit_stopover[i];
				System.arraycopy(stopover[i], 0, superorganism[i], 0, myD);
			}
		}
		myEvals += mySwarmSize;
	}

	@Override
	public MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func, final double[] guess) {
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

		// set functions
		myFunc = func;
		myD = lower.length;
		myLower = lower;
		myUpper = upper;
		myEvals = 0;

		// generate initial individuals, clans and superorganism.
		superorganism = generate_pop(mySwarmSize, myD, lower, upper);

		// success of clans/superorganism
		fit_super = new double[mySwarmSize];
		for (int i = 0; i < mySwarmSize; ++i) {
			fit_super[i] = myFunc.apply(superorganism[i]);
		}
		myEvals += mySwarmSize;

		// I have moved this here and implemented native copy operations
		// or internal modifications instead of the original implementation
		// I think this is much more efficient in Java
		stopover = new double[mySwarmSize][myD];
		map = new int[mySwarmSize][myD];
		direction = new double[mySwarmSize][myD];
		fit_stopover = new double[mySwarmSize];
		jind = new Integer[mySwarmSize];
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

		// initialization of swarm
		initialize(func, lower, upper);

		// main loop
		boolean converged = false;
		while (myEvals < myMaxEvals) {

			// perform iteration
			iterate();

			// converge when distance in fitness between best and worst points
			// is below the given tolerance
			final int imin = Sequences.argmin(mySwarmSize, fit_super);
			final int imax = Sequences.argmax(fit_super.length, fit_super);
			final double best = fit_super[imin];
			final double worst = fit_super[imax];
			final double distY = Math.abs(best - worst);
			final double avgY = 0.5 * (best + worst);
			if (distY <= myTol + RELEPS * Math.abs(avgY)) {

				// compute standard deviation of swarm radiuses
				final int D = lower.length;
				int count = 0;
				double mean = 0.0;
				double m2 = 0.0;
				for (final double[] pt : superorganism) {
					final double x = BlasMath.denorm(D, pt);
					++count;
					final double delta = x - mean;
					mean += delta / count;
					final double delta2 = x - mean;
					m2 += delta * delta2;
				}

				// test convergence in standard deviation
				if (m2 <= (mySwarmSize - 1) * mySigmaTol * mySigmaTol) {
					converged = true;
					break;
				}
			}
		}

		// update results
		final int imin = Sequences.argmin(mySwarmSize, fit_super);
		final double[] sol = superorganism[imin];
		return new MultivariateOptimizerSolution(sol, myEvals, 0, converged);
	}

	private static double[][] generate_dir(final double[][] direction, final int method, final double[][] superorganism,
			final int size, final double[] fit, final Integer[] jind) {
		switch (method) {
		case 1:

			// BIJECTIVE DSA (B-DSA) (i.e., go-to-rnd DSA)
			// philosophy: evolve the superorganism (i.e.,population)
			// towards to "permuted-superorganism (i.e., random directions)
			for (int i = 0; i < size; ++i) {
				jind[i] = i;
			}
			Sequences.shuffle(RAND, 0, jind.length - 1, jind);
			for (int i = 0; i < size; ++i) {
				final int j = jind[i];
				direction[i] = superorganism[j];
			}
			break;
		case 2:

			// SURJECTIVE DSA (S-DSA) (i.e., go-to-good DSA)
			// philosophy: evolve the superorganism (i.e.,population)
			// towards to "some of the random top-best" solutions
			for (int i = 0; i < size; ++i) {
				jind[i] = i;
			}
			Arrays.sort(jind, (i, j) -> Double.compare(fit[i], fit[j]));
			final int nums = (int) Math.ceil(RAND.nextDouble() * size);
			for (int i = 0; i < size; ++i) {
				final int j = RAND.nextInt(nums);
				direction[i] = superorganism[j];
			}
			break;
		case 3:

			// ELITIST DSA #1 (E1-DSA) (i.e., go-to-best DSA)
			// philosophy: evolve the superorganism (i.e.,population)
			// towards to "one of the random top-best" solution
			for (int i = 0; i < size; ++i) {
				jind[i] = i;
			}
			Arrays.sort(jind, (i, j) -> Double.compare(fit[i], fit[j]));
			final int nums1 = Math.min((int) Math.ceil(RAND.nextDouble() * size), size - 1);
			final int ibest = jind[nums1];
			for (int i = 0; i < size; ++i) {
				direction[i] = superorganism[ibest];
			}
			break;
		case 4:

			// ELITIST DSA #2 (E2-DSA) (i.e., go-to-best DSA)
			// philosophy: evolve the superorganism (i.e.,population)
			// towards to "the best" solution
			final int imin = Sequences.argmin(fit.length, fit);
			for (int i = 0; i < size; ++i) {
				direction[i] = superorganism[imin];
			}
			break;
		}
		return direction;
	}

	private static double[][] generate_pop(final int a, final int b, final double[] low, final double[] up) {
		final double[][] pop = new double[a][b];
		for (int i = 1; i <= a; ++i) {
			for (int j = 1; j <= b; ++j) {
				pop[i - 1][j - 1] = RAND.nextDouble() * (up[j - 1] - low[j - 1]) + low[j - 1];
			}
		}
		return pop;
	}

	private static void update(final double[][] p, final double[] low, final double[] up) {
		final int popsize = p.length;
		final int dim = p[0].length;
		for (int i = 1; i <= popsize; ++i) {
			for (int j = 1; j <= dim; ++j) {

				// first (standard)-method
				if (p[i - 1][j - 1] < low[j - 1]) {
					if (RAND.nextDouble() < RAND.nextDouble()) {
						p[i - 1][j - 1] = RAND.nextDouble() * (up[j - 1] - low[j - 1]) + low[j - 1];
					} else {
						p[i - 1][j - 1] = low[j - 1];
					}
				}
				if (p[i - 1][j - 1] > up[j - 1]) {
					if (RAND.nextDouble() < RAND.nextDouble()) {
						p[i - 1][j - 1] = RAND.nextDouble() * (up[j - 1] - low[j - 1]) + low[j - 1];
					} else {
						p[i - 1][j - 1] = up[j - 1];
					}
				}
			}
		}
	}

	private static int[][] generate_map(final int[][] map, final int size_super, final int size_clan, final double p1,
			final double p2) {

		// strategy-selection of active/passive individuals
		if (RAND.nextDouble() < RAND.nextDouble()) {
			if (RAND.nextDouble() < p1) {

				// Random-mutation #1 strategy
				for (int i = 0; i < size_super; ++i) {
					for (int j = 0; j < size_clan; ++j) {
						if (RAND.nextDouble() < RAND.nextDouble()) {
							map[i][j] = 1;
						} else {
							map[i][j] = 0;
						}
					}
				}
			} else {

				// Differential-mutation strategy
				for (int i = 0; i < size_super; ++i) {
					final int j = RAND.nextInt(size_clan);
					Arrays.fill(map[i], 0);
					map[i][j] = 1;
				}
			}
		} else {

			// Random-mutation #2 strategy
			final int mapmax = (int) Math.ceil(p2 * size_clan);
			for (int i = 0; i < size_super; ++i) {
				Arrays.fill(map[i], 0);
				for (int k = 0; k < mapmax; ++k) {
					final int j = RAND.nextInt(size_clan);
					map[i][j] = 1;
				}
			}
		}
		return map;
	}
}
