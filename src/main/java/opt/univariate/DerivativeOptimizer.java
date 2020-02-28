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
package opt.univariate;

import java.util.function.Function;

import opt.OptimizerSolution;
import utils.Constants;

/**
 *
 */
public abstract class DerivativeOptimizer extends UnivariateOptimizer {

	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public DerivativeOptimizer(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		super(absoluteTolerance, relativeTolerance, maxEvaluations);
	}

	public abstract OptimizerSolution<Double, Double> optimize(Function<? super Double, Double> f,
			Function<? super Double, Double> df, double a, double b);

	@Override
	public OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f, final Double guess) {
		throw new IllegalArgumentException("f' not provided; no numerical diff. method exists yet!");
	}

	public OptimizerSolution<Double, Double> optimize(final Function<? super Double, Double> f,
			final Function<? super Double, Double> df, final Double guess) {

		// first use guess to compute a bracket [a, b] that contains a min
		final int[] fev = new int[1];
		final double[] brackt = bracket(f, guess, Constants.GOLDEN, myMaxEvals, fev);
		if (brackt == null) {
			return new OptimizerSolution<>(Double.NaN, fev[0], 0, false);
		}
		final double a = brackt[0];
		final double b = brackt[1];

		// perform optimization using the bracketed routine
		final OptimizerSolution<Double, Double> result = optimize(f, df, a, b);
		return new OptimizerSolution<>(result.getOptimalPoint(), result.getFEvals() + fev[0], result.getDFEvals(),
				result.converged());
	}
}
