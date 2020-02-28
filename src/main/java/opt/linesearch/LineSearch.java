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

import opt.Optimizer;

/**
 * An abstract algorithm for performing a line search.
 */
public abstract class LineSearch extends Optimizer<Double, Double, LineSearchProblem> {

	protected final double myTol;
	protected final int myMaxIters;
	protected final double myC1;

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 * @param c1
	 */
	public LineSearch(final double tolerance, final int maxIterations, final double c1) {
		myTol = tolerance;
		myMaxIters = maxIterations;
		myC1 = c1;
	}

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 */
	public LineSearch(final double tolerance, final int maxIterations) {
		this(tolerance, maxIterations, 1e-4);
	}

	/**
	 *
	 * @param f
	 * @param df
	 * @param x0
	 * @param dir
	 * @param df0
	 * @param f0
	 * @param initial
	 * @return
	 */
	public abstract LineSearchSolution lineSearch(Function<? super double[], Double> f,
			Function<? super double[], double[]> df, double[] x0, double[] dir, double[] df0, double f0,
			double initial);

	@Override
	public LineSearchSolution optimize(final LineSearchProblem problem, final Double guess) {
		final Function<double[], Double> f = problem.myFunc;
		final Function<double[], double[]> df = problem.myDFunc;
		final double[] x0 = problem.myX0;
		final double[] dir = problem.myD;
		final double[] df0 = df.apply(x0);
		final double f0 = f.apply(x0);
		return lineSearch(f, df, x0, dir, df0, f0, guess);
	}
}
