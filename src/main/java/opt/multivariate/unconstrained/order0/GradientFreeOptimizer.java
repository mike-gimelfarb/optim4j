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
package opt.multivariate.unconstrained.order0;

import java.util.Random;
import java.util.function.Function;

import opt.Optimizer;
import utils.Constants;

/**
 * 
 */
public abstract class GradientFreeOptimizer extends Optimizer<double[], Double, Function<? super double[], Double>> {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	protected static final Random RAND = new Random();
	protected static final double RELEPS = Constants.EPSILON;

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final double myTol;
	protected int myEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 */
	public GradientFreeOptimizer(final double tolerance) {
		myTol = tolerance;
	}

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	public abstract void initialize(Function<? super double[], Double> func, double[] guess);

	public abstract void iterate();

	public abstract double[] optimize(Function<? super double[], Double> func, double[] guess);

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @return
	 */
	public final int countEvaluations() {
		return myEvals;
	}

	/*
	
	 */
	public final void resetCounter() {
		myEvals = 0;
	}
}
