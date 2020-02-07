package min4j.multivariate.unconstrained.order0.cmaes;

import java.util.Arrays;
import java.util.function.Function;

import min4j.multivariate.unconstrained.order0.GradientFreeOptimizer;
import min4j.multivariate.unconstrained.order0.cmaes.AbstractCmaesOptimizer.AbstractCmaesFactory;

/**
 * Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009
 * function testbed." Proceedings of the 11th Annual Conference Companion on
 * Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM,
 * 2009.
 * 
 * Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box Optimization
 * Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the BBOB-2012 Noiseless
 * Testbed." Genetic and Evolutionary Computation Conference (GECCO-2012), ACM
 * Press : 269-276. July 2012.
 * 
 * Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative restart
 * strategies for CMA-ES." International Conference on Parallel Problem Solving
 * from Nature. Springer, Berlin, Heidelberg, 2012.
 * 
 * @author Michael
 */
public final class BiPopCmaesAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
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
	private int myEvalsref, myLastRegime, myCurrentRegime, myLastBestRegime;
	private double mySigmal, mySigma;
	private double myFx, myFxold, myFxBest;
	private double[] myX, myXGuess, myX0, myXBest;
	private AbstractCmaesOptimizer myCmaes;

	// domain
	private Function<? super double[], Double> myFunc;
	private int myD;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
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

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
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
		myX = myCmaes.optimize(func, myX0);
		myFx = func.apply(myX);

		// initialize counters - note we do first restart with first regime
		myEvals = myCmaes.countEvaluations() + 1;
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
	public double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		initialize(func, guess);
		while (true) {
			iterate();

			// check if reached max number of restarts or evals
			if (myIl >= myMaxRuns || myEvals >= myMaxEvals) {
				return myXBest;
			}

			// check for convergence
			if (myLastRegime == 0 && myFxold != myFx) {
				final double ftol = RELEPS * 0.5 * Math.abs(myFx + myFxold);
				if (Math.abs(myFx - myFxold) <= myTol + ftol) {
					return myXBest;
				}
				myFxold = myFx;
			}
		}
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
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
		myX = myCmaes.optimize(myFunc, myX0);
		myFx = myFunc.apply(myX);

		// increment counters and adjust budget
		myEvals += myCmaes.countEvaluations() + 1;
		myBudgetl += myCmaes.countEvaluations() + 1;
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
		myX = myCmaes.optimize(myFunc, myX0);
		myFx = myFunc.apply(myX);

		// increment counters and adjust budget
		myEvals += myCmaes.countEvaluations() + 1;
		myBudgets += myCmaes.countEvaluations() + 1;
		++myIs;
		myLastRegime = 1;
	}
}
