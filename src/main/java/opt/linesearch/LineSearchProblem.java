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
 * A class that represents a univariate function that is defined by evaluating a
 * multivariate function along a ray. In other words, given a vector x0 and a
 * vector v, and an arbitrary differentiable function f, this class represents a
 * function g of the form
 * 
 * g(a) = f(x0 + a*v)
 * 
 * where a is an element of the real number.
 */
public final class LineSearchProblem implements Function<Double, Double> {

	protected final Function<double[], Double> myFunc;
	protected final Function<double[], double[]> myDFunc;
	protected final double[] myX0, myD;
	private final double[] myTemp;
	private final int myN;

	/**
	 *
	 * @param func
	 * @param dfunc
	 * @param x0
	 * @param dir
	 */
	public LineSearchProblem(final Function<double[], Double> func, final Function<double[], double[]> dfunc,
			final double[] x0, final double[] dir) {
		myFunc = func;
		myDFunc = dfunc;
		myX0 = x0;
		myD = dir;
		myN = myX0.length;
		myTemp = new double[myN];
	}

	@Override
	public final Double apply(final Double t) {
		BlasMath.daxpy1(myN, t, myD, 1, myX0, 1, myTemp, 1);
		return myFunc.apply(myTemp);
	}
}
