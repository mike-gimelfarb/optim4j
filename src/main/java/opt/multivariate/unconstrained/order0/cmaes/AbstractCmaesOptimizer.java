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
import utils.BlasMath;

/**
 * An abstract class to represent all covariance matrix adaptation strategy
 * based algorithm (CMA-ES).
 */
public abstract class AbstractCmaesOptimizer extends GradientFreeOptimizer {

	/**
	 * 
	 * @author Michael
	 *
	 */
	public static interface AbstractCmaesFactory {

		/**
		 * 
		 * @param tolerance
		 * @param populationSize
		 * @param initialSigma
		 * @param maxEvaluations
		 * @return
		 */
		public AbstractCmaesOptimizer createCmaStrategy(final double tolerance, final int populationSize,
				final double initialSigma, final int maxEvaluations);
	}

	// A structure to hold a pair of integer double values.
	final class IntDoublePair implements Comparable<IntDoublePair> {

		int index;
		double value;

		@Override
		public final int compareTo(final IntDoublePair o) {
			return Double.compare(value, o.value);
		}
	}

	// A structure implemented as a circular buffer to store the history of fitness
	// values.
	final class FitnessHistory {

		final int capacity;
		int buffer;
		int length;
		double[] values;

		FitnessHistory(final int len) {
			capacity = len;
			values = new double[capacity];
			buffer = -1;
			length = 0;
		}

		final void add(final double value) {
			buffer = (buffer + 1) % capacity;
			values[buffer] = value;
			if (length < capacity) {
				++length;
			}
		}

		final double get(final int i) {
			final int idx = (capacity + buffer - i) % capacity;
			return values[idx];
		}
	}

	// domain properties
	protected Function<? super double[], Double> myFunc;
	protected int D;

	// algorithm properties
	protected final boolean myAdaptivePop, myAdaptiveIters;
	protected final double mySigma0;

	// algorithm mutable memory and parameters
	protected int myMaxEvals, myMaxIters;
	protected int myLambda, myMu;
	protected int myIteration;
	protected IntDoublePair[] arfitness;
	protected int[] ibw;
	protected double[] ybw;
	protected double[][] arx;
	protected double mueff, cc, cs, c1, cmu, damps, chi, sigma;
	protected double[] xmean, xold, weights, artmp, pc, ps;

	// history
	protected int myHistoryLength, ik, myEvals;
	protected double myHistoryBestFit, myHistoryWorstFit;
	protected FitnessHistory myHistoryBest, myHistoryKth;

	/**
	 *
	 * @param tolerance
	 * @param populationSize
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public AbstractCmaesOptimizer(final double tolerance, final int populationSize, final double initialSigma,
			final int maxEvaluations) {
		super(tolerance);
		myLambda = populationSize;
		mySigma0 = initialSigma;
		myMaxEvals = maxEvaluations;
		myAdaptivePop = myAdaptiveIters = false;
	}

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param initialSigma
	 * @param maxEvaluations
	 */
	public AbstractCmaesOptimizer(final double tolerance, final double initialSigma, final int maxEvaluations) {
		super(tolerance);
		mySigma0 = initialSigma;
		myMaxEvals = maxEvaluations;
		myAdaptivePop = true;
		myAdaptiveIters = false;
	}

	/**
	 *
	 * @param tolerance
	 * @param initialSigma
	 */
	public AbstractCmaesOptimizer(final double tolerance, final double initialSigma) {
		super(tolerance);
		mySigma0 = initialSigma;
		myAdaptivePop = myAdaptiveIters = true;
	}

	/**
	 * 
	 */
	public abstract void samplePopulation();

	/**
	 * 
	 */
	public abstract void updateDistribution();

	/**
	 * 
	 * @return
	 */
	public abstract boolean converged();

	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {

		// initialize domain
		myFunc = func;
		D = guess.length;
		myEvals = 0;

		// adaptive initialization of population size
		if (myAdaptivePop) {
			myLambda = 4 + (int) Math.floor(3.0 * Math.log(D));
		}
		myMu = (int) Math.floor(myLambda / 2.0);

		// adaptive initialization of maximum evaluations
		if (myAdaptiveIters) {
			myMaxIters = (int) (100.0 + 50.0 * (D + 3) * (D + 3) / Math.sqrt(myLambda));
			myMaxEvals = myMaxIters * myLambda;
		} else {
			myMaxIters = myMaxEvals / myLambda;
		}

		// initialization of population and ranking memory
		arx = new double[myLambda][D];
		ibw = new int[4];
		ybw = new double[4];
		arfitness = new IntDoublePair[myLambda];
		for (int i = 0; i < myLambda; ++i) {
			arfitness[i] = new IntDoublePair();
		}

		// initialize array for weighted recombination
		weights = new double[myMu];
		double sum = 0.0;
		for (int i = 0; i < myMu; ++i) {
			weights[i] = Math.log(0.5 * (myLambda + 1)) - Math.log(i + 1);
			sum += weights[i];
		}
		BlasMath.dscalm(myMu, 1.0 / sum, weights, 1);

		// initialize variance-effectiveness of sum w_i x_i
		final double lenw = BlasMath.denorm(myMu, weights);
		mueff = 1.0 / (lenw * lenw);

		// initialize strategy parameter settings
		chi = Math.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D * D));
		sigma = mySigma0;
		cc = (4.0 + mueff / D) / (D + 4.0 + 2.0 * mueff / D);
		cs = (mueff + 2.0) / (5.0 + D + mueff);
		c1 = 2.0 / ((1.3 + D) * (1.3 + D) + mueff);
		cmu = Math.min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((2.0 + D) * (2.0 + D) + mueff));
		damps = 1.0 + cs + 2.0 * Math.max(0.0, Math.sqrt((mueff - 1.0) / (D + 1.0)) - 1.0);

		// initialize other memories
		pc = new double[D];
		ps = new double[D];
		artmp = new double[D];
		xold = new double[D];
		xmean = Arrays.copyOf(guess, D);
		myIteration = myEvals = 0;

		// initialize history and convergence parameters
		myHistoryLength = 10 + (int) Math.ceil(30.0 * D / myLambda);
		ik = (int) Math.ceil(0.1 + myLambda / 4.0);
		myHistoryBest = new FitnessHistory(myHistoryLength);
		myHistoryKth = new FitnessHistory(myHistoryLength);
		myHistoryBestFit = Double.NEGATIVE_INFINITY;
		myHistoryWorstFit = Double.POSITIVE_INFINITY;
	}

	@Override
	public void iterate() {
		samplePopulation();
		evaluateAndSortPopulation();
		updateDistribution();
		updateHistory();
		++myIteration;
	}

	@Override
	public MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func, final double[] guess) {
		initialize(func, guess);
		boolean converged = false;
		while (myEvals < myMaxEvals) {
			iterate();
			if (converged()) {
				converged = true;
				break;
			}
		}
		return new MultivariateOptimizerSolution(getBestSolution(), myEvals, 0, converged);
	}

	/**
	 * 
	 */
	public void updateSigma() {

		// basic sigma update
		final double pslen = BlasMath.denorm(D, ps);
		sigma *= Math.exp(Math.min(1.0, (cs / damps) * (pslen / chi - 1.0)));

		// Adjust step size in case of equal function values (flat fitness)
		if (arfitness[0].value == arfitness[ik].value) {
			sigma *= Math.exp(0.2 + cs / damps);
		}
		if (myIteration >= myHistoryLength && myHistoryWorstFit - myHistoryBestFit == 0.0) {
			sigma *= Math.exp(0.2 + cs / damps);
		}
	}

	/**
	 * 
	 */
	public void updateHistory() {
		if (myIteration >= myMaxIters) {
			return;
		}

		// append new observation
		myHistoryBest.add(arfitness[0].value);
		myHistoryKth.add(arfitness[ik].value);

		// update running recent worst and best fitness values
		if (myHistoryBest.length == myHistoryBest.capacity) {
			myHistoryBestFit = Double.POSITIVE_INFINITY;
			myHistoryWorstFit = Double.NEGATIVE_INFINITY;
			for (final double fx : myHistoryBest.values) {
				myHistoryBestFit = Math.min(fx, myHistoryBestFit);
				myHistoryWorstFit = Math.max(fx, myHistoryWorstFit);
			}
		}
	}

	/**
	 * 
	 */
	public void evaluateAndSortPopulation() {

		// Sort by fitness
		for (int i = 0; i < myLambda; ++i) {
			arfitness[i].index = i;
			arfitness[i].value = myFunc.apply(arx[i]);
		}
		myEvals += myLambda;

		// get the best and worst elements
		Arrays.sort(arfitness);
		ibw[0] = arfitness[0].index;
		ibw[1] = arfitness[1].index;
		ibw[2] = arfitness[myLambda - 2].index;
		ibw[3] = arfitness[myLambda - 1].index;
		ybw[0] = arfitness[0].value;
		ybw[1] = arfitness[1].value;
		ybw[2] = arfitness[myLambda - 2].value;
		ybw[3] = arfitness[myLambda - 1].value;
	}

	/**
	 *
	 * @return
	 */
	public double[] getBestSolution() {
		if (myIteration <= 0) {
			return xmean;
		} else {
			return arx[ibw[0]];
		}
	}

	/**
	 * 
	 * @return
	 */
	public final int countIterations() {
		return myIteration;
	}
}
