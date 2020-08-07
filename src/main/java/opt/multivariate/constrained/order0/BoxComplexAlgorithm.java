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
package opt.multivariate.constrained.order0;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Predicate;

import opt.OptimizerSolution;
import utils.Sequences;

/**
 * An algorithm for minimization of a general function subject to general
 * nonlinear constraints that define a convex set. without derivative
 * information, introduced by Box (1965). This implementation also considers
 * modifications suggested in Guin (1968).
 * 
 * 
 * REFERENCES:
 * 
 * [1] Box, M. J. "A new method of constrained optimization and a comparison
 * with other methods." The Computer Journal 8.1 (1965): 42-52.
 * 
 * [2] Guin, J. A. "Modification of the complex method of constrained
 * optimization." The Computer Journal 10.4 (1968): 416-417.
 */
public final class BoxComplexAlgorithm {

	private static final Random RAND = new Random();

	// algorithm parameters
	private final double myTol;
	private final int myMaxEvals;
	private final Function<Integer, Integer> myBoxSize;
	private final boolean myMoveToBest;
	private final boolean myAdaptiveAlpha;

	// problem parameters
	private Function<? super double[], Double> myObj;
	private Predicate<? super double[]> myConstr;
	private double[] myLower;
	private double[] myUpper;
	private int myN;
	private int myK;

	// memory for iterations
	private double myAlpha;
	private int myFEvals;
	private int myGEvals;
	private double[][] myPts;
	private double[] myValue;
	private double[] myCenter;
	private double[] myCenter0;
	private double[] myXReflect;

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 * @param alphaParam
	 * @param boxSizeFunc
	 * @param moveToBest
	 */
	public BoxComplexAlgorithm(final double tolerance, final int maxEvaluations, final double alphaParam,
			final boolean moveToBest, final Function<Integer, Integer> boxSizeFunc) {
		myAlpha = alphaParam;
		myTol = tolerance;
		myMaxEvals = maxEvaluations;
		myBoxSize = boxSizeFunc;
		myMoveToBest = moveToBest;
		myAdaptiveAlpha = false;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 * @param alphaParam
	 * @param moveToBest
	 */
	public BoxComplexAlgorithm(final double tolerance, final int maxEvaluations, final double alphaParam,
			final boolean moveToBest) {
		this(tolerance, maxEvaluations, alphaParam, moveToBest, d -> 2 * d);
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 * @param moveToBest
	 */
	public BoxComplexAlgorithm(final double tolerance, final int maxEvaluations, final boolean moveToBest) {
		myTol = tolerance;
		myMaxEvals = maxEvaluations;
		myBoxSize = d -> 2 * d;
		myMoveToBest = moveToBest;
		myAdaptiveAlpha = true;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxEvaluations
	 */
	public BoxComplexAlgorithm(final double tolerance, final int maxEvaluations) {
		this(tolerance, maxEvaluations, false);
	}

	/**
	 *
	 * @param objective
	 * @param feasibleRegion
	 * @param lowerBound
	 * @param upperBound
	 * @param guess
	 * @return
	 */
	public final OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> objective,
			final Predicate<? super double[]> feasibleRegion, final double[] lowerBound, final double[] upperBound,
			final double[] guess) {

		// initialize complex
		initialize(objective, feasibleRegion, lowerBound, upperBound, guess);

		// main loop of optimization
		boolean converged = false;
		while (true) {

			// check budget
			if (myFEvals >= myMaxEvals || myGEvals >= myMaxEvals) {
				break;
			}

			// perform update of complex
			iterate();

			// check convergence
			if (isConverged()) {
				converged = true;
				break;
			}
		}
		final int imin = Sequences.argmin(myN, myValue);
		return new OptimizerSolution<>(myPts[imin], myFEvals, myGEvals, converged);
	}

	/**
	 *
	 * @param objective
	 * @param feasibleRegion
	 * @param lowerBound
	 * @param upperBound
	 * @param guess
	 */
	public final void initialize(final Function<? super double[], Double> objective,
			final Predicate<? super double[]> feasibleRegion, final double[] lowerBound, final double[] upperBound,
			final double[] guess) {

		// initialize parameters and memory
		myObj = objective;
		myConstr = feasibleRegion;
		myLower = lowerBound;
		myUpper = upperBound;
		myN = guess.length;
		myK = myBoxSize.apply(myN);
		myFEvals = myGEvals = 0;
		myPts = new double[myK][myN];
		myValue = new double[myK];
		myCenter0 = new double[myN];
		myXReflect = new double[myN];

		// adaptive mode - this is still experimental
		if (myAdaptiveAlpha) {
			myAlpha = 1.0 + 1.0 / myN;
		}

		// perform a monte carlo search for a feasible point
		// if the feasible region occupies proportion p of the area in
		// the hypercube [lower, upper], then on average we will need
		// 1 / p constraint evaluations to find a feasible point, e.g.
		// if p = 0.05 then we need around 20 evaluations
		final double[] start = Arrays.copyOf(guess, myN);
		while (myGEvals < 100) {
			++myGEvals;
			if (myConstr.test(start)) {
				break;
			}
			final double r = RAND.nextDouble();
			for (int j = 0; j < myN; ++j) {
				start[j] = myLower[j] + r * (myUpper[j] - myLower[j]);
			}
		}
		if (!myConstr.test(start)) {
			throw new IllegalArgumentException(
					"Initial point is not feasible " + "- feasible region might be " + "too small.");
		}

		// set initial point as guess
		myPts[0] = Arrays.copyOf(start, myN);
		myCenter = Arrays.copyOf(start, myN);
		myValue[0] = myObj.apply(start);
		++myFEvals;

		// add the remaining points
		for (int i = 1; i < myK; ++i) {

			// generate the initial point
			final double[] ptsi = myPts[i];
			for (int j = 0; j < myN; ++j) {
				final double r = RAND.nextDouble();
				ptsi[j] = myLower[j] + r * (myUpper[j] - myLower[j]);
			}

			// perform bisection until the point is feasible
			while (!myConstr.test(ptsi)) {
				++myGEvals;
				for (int j = 0; j < myN; ++j) {
					ptsi[j] = 0.5 * (ptsi[j] + myCenter[j]);
				}
			}

			// update the center
			for (int j = 0; j < myN; ++j) {
				myCenter[j] += (ptsi[j] - myCenter[j]) / (i + 1);
			}

			// update the cached function values
			myValue[i] = myObj.apply(ptsi);
			++myFEvals;
		}
	}

	/**
	 *
	 */
	public final void iterate() {

		// find the point with the highest value and the center of the remaining
		// points in the complex
		final int imax = Sequences.argmax(myValue.length, myValue);
		final double[] xhigh = myPts[imax];
		for (int j = 0; j < myN; ++j) {
			myCenter0[j] = myCenter[j] + (myCenter[j] - xhigh[j]) / (myK - 1);
		}

		// find the reflection of the highest point
		for (int j = 0; j < myN; ++j) {
			myXReflect[j] = myCenter0[j] + myAlpha * (myCenter0[j] - xhigh[j]);
		}

		// enforce the bound constraints for the new point
		for (int j = 0; j < myN; ++j) {
			if (myXReflect[j] < myLower[j]) {
				myXReflect[j] = myLower[j] + 1.0e-6;
			}
			if (myXReflect[j] > myUpper[j]) {
				myXReflect[j] = myUpper[j] - 1.0e-6;
			}
		}

		// while the new point is not feasible, move the new point closer to the
		// center
		while (!myConstr.test(myXReflect)) {
			++myGEvals;
			for (int j = 0; j < myN; ++j) {
				myXReflect[j] = 0.5 * (myXReflect[j] + myCenter0[j]);
			}
			if (myGEvals >= myMaxEvals) {
				return;
			}
		}

		// while the fitness of the new point is worse than the worst in the
		// box, move the new point closer to the center
		final int imin = Sequences.argmin(myValue.length, myValue);
		final double[] xlow = myPts[imin];
		final double fhigh = myValue[imax];
		double freflect = myObj.apply(myXReflect);
		int kf = 0;
		++myFEvals;
		while (freflect > fhigh) {
			++kf;
			if (myMoveToBest) {
				final double a = 1.0 - Math.exp(-kf / 4.0);
				for (int j = 0; j < myN; ++j) {
					final double temp = a * xlow[j] + (1.0 - a) * myCenter0[j];
					myXReflect[j] = 0.5 * (temp + myXReflect[j]);
				}
			} else {
				for (int j = 0; j < myN; ++j) {
					myXReflect[j] = 0.5 * (myCenter0[j] + myXReflect[j]);
				}
			}
			freflect = myObj.apply(myXReflect);
			++myFEvals;
			if (myFEvals >= myMaxEvals) {
				return;
			}
		}

		// replace worst point in complex by the new point
		for (int j = 0; j < myN; ++j) {
			myCenter[j] += (myXReflect[j] - xhigh[j]) / myK;
		}
		System.arraycopy(myXReflect, 0, xhigh, 0, myN);
		myValue[imax] = freflect;
	}

	private boolean isConverged() {
		int count = 0;
		double mean = 0.0;
		double m2 = 0.0;
		for (final double value : myValue) {
			++count;
			final double delta = value - mean;
			mean += delta / count;
			final double delta2 = value - mean;
			m2 += delta * delta2;
		}
		return m2 <= (count - 1.0) * myTol * myTol;
	}
}
