/*
 * Copyright (c) 2020 Mike Gimelfarb
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the > "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, > subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package opt.linesearch;

import java.util.function.Function;

import utils.BlasMath;

/**
 * A backtracking line search algorithm described in Nocedal and Wright (2006).
 * This line search routine is largely academic and should not be used in
 * practical settings.
 * 
 * 
 * REFERENCES:
 * 
 * [1] Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer
 * Science & Business Media, 2006.
 */
public final class BacktrackingLineSearch extends LineSearch {

	this wont work, period
	print(x)
	
	private final double myRho, myMaxStepSize;

	/**
	 *
	 * @param tolerance
	 * @param decay
	 * @param maximum
	 * @param maxIterations
	 */
	public BacktrackingLineSearch(final double tolerance, final double decay, final double maximum,
			final int maxIterations) {
		super(tolerance, maxIterations);
		myRho = decay;
		myMaxStepSize = maximum;
	}

	@Override
	public final LineSearchSolution lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double initial) {
		final int D = x0.length;

		// prepare initial position and dot products
		final double[] x = new double[D];
		double step = initial;
		double y;
		BlasMath.daxpy1(D, step, dir, 1, x0, 1, x, 1);
		final double dy = BlasMath.ddotm(D, df0, 1, dir, 1);
		final double normdir = BlasMath.denorm(D, dir);
		int fevals = 0;

		// main loop of backtracking line search
		for (int i = 0; i < myMaxIters; ++i) {

			// compute new position and function value for step
			BlasMath.daxpy1(D, step, dir, 1, x0, 1, x, 1);
			y = f.apply(x);
			++fevals;

			// check the approximate Wolfe condition
			if (y <= f0 + myC1 * step * dy) {
				return new LineSearchSolution(step, fevals, 0, x, true);
			}

			// update step size
			step = Math.min(step * myRho, myMaxStepSize / normdir);
		}
		return new LineSearchSolution(step, fevals, 0, x, false);
	}
}
