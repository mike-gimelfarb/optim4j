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
package opt.multivariate;

import java.util.function.Function;

import opt.Optimizer;
import opt.OptimizerSolution;

/**
 *
 */
public abstract class GradientOptimizer extends Optimizer<double[], Double, Function<? super double[], Double>> {

	protected final double myTol;

	/**
	 *
	 * @param tolerance
	 */
	public GradientOptimizer(final double tolerance) {
		myTol = tolerance;
	}

	/**
	 * 
	 * @param f
	 * @param df
	 * @param guess
	 * @return
	 */
	public abstract OptimizerSolution<double[], Double> optimize(Function<? super double[], Double> f,
			Function<? super double[], double[]> df, double[] guess);

	/**
	 *
	 * @param f
	 * @param guess
	 * @return
	 */
	public OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> f,
			final double[] guess) {
		throw new IllegalArgumentException("f' not provided; no numerical diff. method exists yet!");
	}
}
