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

import opt.multivariate.GradientFreeOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import opt.multivariate.unconstrained.order0.cmaes.AbstractCmaesOptimizer.AbstractCmaesFactory;

/**
 * An object-oriented implementation of the adaptive two-population CMA-ES
 * algorithm and its variants (e.g. BIPOP-CMA-ES, NBIPOP-aCMA-ES, etc). The
 * underlying CMA-ES implementation can be specified by the user. Suited for
 * minimization of a (relatively smooth) non-linear function without
 * constraints.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009
 * function testbed." Proceedings of the 11th Annual Conference Companion on
 * Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM,
 * 2009.
 * 
 * [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 * Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 * BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 * (GECCO-2012), ACM Press : 269-276. July 2012.
 * 
 * [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 * restart strategies for CMA-ES." International Conference on Parallel Problem
 * Solving from Nature. Springer, Berlin, Heidelberg, 2012.
 */
public final class BiPopCmaesAlgorithm extends GradientFreeOptimizer {

	// algorithm parameters
	private final int myBudgetFactor = 2;
	private final int myMaxEvals, myMaxRuns;
	private final double myCmaesTol, mySigmaRef, mySigmaDec;
	private final boolean myPrint, myAdaptiveBudget;
	private final AbstractCmaesFactory myCmaesFactory;

	// algorithm memory and counters
	private int myLambdal, myLambdas, myLambda, myLambdaRef;
	private int myBudgetl, myBudgets;
	private int myIl, myIs, myIteration;
	private int myEvalsref, myLastRegime, myCurrentRegime, myLastBestRegime, myEvals;
	private double mySigmal, mySigma;
	private double myFx, myFxold, myFxBest;
	private double[] myX, myXGuess, myX0, myXBest;
	private AbstractCmaesOptimizer myCmaes;

	// domain
	private Function<? super double[], Double> myFunc;
	private int myD;

	/**
	 *
	 * @param tolerance
	 * @param cmaesTolerance;
	 * @param sigma0
	 * @param maxEvaluations
	 * @param maxLargePopulationRuns
	 * @param printProgress
	 * @param cmaesFactory
	 * @param sigmaDecayFactor
	 * @param useAdaptiveBudget
	 */
	public BiPopCmaesAlgorithm(final double tolerance, final double cmaesTolerance, final double sigma0,
			final int maxEvaluations, final int maxLargePopulationRuns, final boolean printProgress,
			final AbstractCmaesFactory cmaesFactory, final double sigmaDecayFactor, final boolean useAdaptiveBudget) {

		// implements generic BIPOP-CMA-ES
		super(tolerance);
		myCmaesTol = cmaesTolerance;
		mySigmaRef = sigma0;
		myMaxEvals = maxEvaluations;
		myMaxRuns = maxLargePopulationRuns;
		myPrint = printProgress;
		myCmaesFactory = cmaesFactory;
		mySigmaDec = sigmaDecayFactor;
		myAdaptiveBudget = useAdaptiveBudget;
	}

	/**
	 *
	 * @param tolerance
	 * @param cmaesTolerance
	 * @param sigma0
	 * @param maxEvaluations
	 * @param maxLargePopulationRuns
	 * @param printProgress
	 */
	public BiPopCmaesAlgorithm(final double tolerance, final double cmaesTolerance, final double sigma0,
			final int maxEvaluations, final int maxLargePopulationRuns, final boolean printProgress) {

		// implements NBIPOP-aCMA-ES
		this(tolerance, cmaesTolerance, sigma0, maxEvaluations, maxLargePopulationRuns, printProgress,
				new ActiveCmaesAlgorithm.ActiveCmaesFactory(), 1.6, true);
	}

	/**
	 *
	 * @param tolerance
	 * @param cmaesTolerance
	 * @param sigma0
	 * @param maxEvaluations
	 * @param printProgress
	 * @param cmaesFactory
	 */
	public BiPopCmaesAlgorithm(final double tolerance, final double cmaesTolerance, final double sigma0,
			final int maxEvaluations, final boolean printProgress) {

		// implements NBIPOP-aCMA-ES
		this(tolerance, cmaesTolerance, sigma0, maxEvaluations, 9, printProgress);
	}

	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {

		// initialize problem
		myFunc = func;
		myD = guess.length;
		myXGuess = guess;

		// initialize lambda
		myLambdaRef = 4 + (int) Math.floor(3.0 * Math.log(myD));
		myLambda = myLambdal = myLambdaRef;

		// initialize sigma
		mySigma = mySigmal = mySigmaRef;

		// initialize evaluations
		myEvalsref = (myD + 3) * (myD + 3);
		myEvalsref = (int) (100.0 + 50.0 * myEvalsref / Math.sqrt(myLambda)) * myLambda;
		myEvalsref = Math.min(myEvalsref, myMaxEvals);

		// set up the optimizer
		myCmaes = myCmaesFactory.createCmaStrategy(myCmaesTol, myLambda, mySigma, myEvalsref);
		myX0 = Arrays.copyOf(myXGuess, myD);

		// first default run with small population size
		final MultivariateOptimizerSolution sol = myCmaes.optimize(func, myX0);
		myX = sol.getOptimalPoint();
		myFx = func.apply(myX);

		// initialize counters - note we do first restart with first regime
		myEvals = sol.getFEvals() + 1;
		myBudgetl = myBudgets = 0;
		myIteration = myIl = myIs = 0;
		myLastRegime = myCurrentRegime = myLastBestRegime = 0;

		// initialize best points
		myXBest = myX;
		myFxBest = myFx;
		myFxold = Double.NaN;

		// print output
		if (myPrint) {
			System.out.println("Run\tMode\tRun1\tRun2\tBudget1\tBudget2\t" + "MaxBudget\tPop\tSigma\tF\tBestF");
			System.out.println(myIteration + "\t" + 0 + "\t" + myIl + "\t" + myIs + "\t" + myBudgetl + "\t" + myBudgets
					+ "\t" + myEvalsref + "\t" + myLambda + "\t" + mySigma + "\t" + myFx + "\t" + myFxBest);
		}
	}

	@Override
	public final void iterate() {

		// evolve the initial guess using D-dim random walk
		for (int i = 0; i < myD; ++i) {
			myX0[i] = myXGuess[i] + mySigmaRef * RAND.nextGaussian();
		}

		// decide which strategy to run
		if (myIteration == 0) {

			// first run is with the large population size
			myCurrentRegime = 0;
		} else {
			if (myAdaptiveBudget) {

				// in NBIPOP-aCMA-ES, use an adaptive strategy for each budget:
				// allocate twice the budget to the regime which yielded the best
				// solution in the recent runs
				if (myLastBestRegime == 0) {
					if (myBudgetl <= myBudgetFactor * myBudgets) {
						myCurrentRegime = 0;
					} else {
						myCurrentRegime = 1;
					}
				} else {
					if (myBudgets <= myBudgetFactor * myBudgetl) {
						myCurrentRegime = 1;
					} else {
						myCurrentRegime = 0;
					}
				}
			} else {

				// in BIPOP-CMA-ES we choose the regime with the lower budget
				if (myBudgetl <= myBudgets) {
					myCurrentRegime = 0;
				} else {
					myCurrentRegime = 1;
				}
			}
		}

		// apply the strategy with the lower budget
		if (myCurrentRegime == 0) {
			runFirstRegime();
		} else {
			runSecondRegime();
		}

		// update the best point so far
		if (myFx < myFxBest) {
			myXBest = myX;
			myFxBest = myFx;
			myLastBestRegime = myCurrentRegime;
		}

		// print output
		if (myPrint) {
			System.out.println(myIteration + "\t" + myCurrentRegime + "\t" + myIl + "\t" + myIs + "\t" + myBudgetl
					+ "\t" + myBudgets + "\t" + myEvalsref + "\t" + myLambda + "\t" + mySigma + "\t" + myFx + "\t"
					+ myFxBest);
		}
		++myIteration;
	}

	@Override
	public MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func, final double[] guess) {
		initialize(func, guess);
		boolean converged = false;
		while (true) {
			iterate();

			// check if reached max number of restarts or evals
			if (myIl >= myMaxRuns || myEvals >= myMaxEvals) {
				break;
			}

			// check for convergence
			if (myLastRegime == 0 && myFxold != myFx) {
				final double ftol = RELEPS * 0.5 * Math.abs(myFx + myFxold);
				if (Math.abs(myFx - myFxold) <= myTol + ftol) {
					converged = true;
					break;
				}
				myFxold = myFx;
			}
		}
		return new MultivariateOptimizerSolution(myXBest, myEvals, 0, converged);
	}

	/**
	 * 
	 */
	public final void runFirstRegime() {

		// compute the new lambda
		myLambdal <<= 1;
		myLambda = myLambdal;

		// adjust the sigma based on Loshchilov et al. (2012)
		mySigmal /= mySigmaDec;
		mySigmal = Math.max(0.01 * mySigmaRef, mySigmal);
		mySigma = mySigmal;

		// number of function evaluations
		myEvalsref = (myD + 3) * (myD + 3);
		myEvalsref = (int) (100.0 + 50.0 * myEvalsref / Math.sqrt(myLambda)) * myLambda;
		myEvalsref = Math.min(myEvalsref, myMaxEvals - myEvals);

		// set up the optimizer
		myCmaes = myCmaesFactory.createCmaStrategy(myCmaesTol, myLambdal, mySigma, myEvalsref);

		// run the CMAES with increasing population size
		final MultivariateOptimizerSolution sol = myCmaes.optimize(myFunc, myX0);
		myX = sol.getOptimalPoint();
		myFx = myFunc.apply(myX);

		// increment counters and adjust budget
		myEvals += sol.getFEvals() + 1;
		myBudgetl += sol.getFEvals() + 1;
		++myIl;
		myLastRegime = 0;
	}

	/**
	 * 
	 */
	public final void runSecondRegime() {

		// compute new lambda
		final double u = RAND.nextDouble();
		myLambdas = (int) (myLambdaRef * Math.pow(0.5 * myLambdal / myLambdaRef, u * u));
		myLambda = myLambdas;

		// compute new sigma
		mySigma = mySigmaRef * Math.pow(10.0, -2.0 * RAND.nextDouble());

		// number of function evaluations
		if (myLastRegime == 0) {
			myEvalsref = myBudgetl >>> 1;
		} else {
			myEvalsref = (myD + 3) * (myD + 3);
			myEvalsref = (int) (100.0 + 50.0 * myEvalsref / Math.sqrt(myLambda)) * myLambda;
		}
		myEvalsref = Math.min(myEvalsref, myMaxEvals - myEvals);

		// set up the optimizer
		myCmaes = myCmaesFactory.createCmaStrategy(myCmaesTol, myLambda, mySigma, myEvalsref);

		// run the CMAES with small population size
		final MultivariateOptimizerSolution sol = myCmaes.optimize(myFunc, myX0);
		myX = sol.getOptimalPoint();
		myFx = myFunc.apply(myX);

		// increment counters and adjust budget
		myEvals += sol.getFEvals() + 1;
		myBudgets += sol.getFEvals() + 1;
		++myIs;
		myLastRegime = 1;
	}
}
