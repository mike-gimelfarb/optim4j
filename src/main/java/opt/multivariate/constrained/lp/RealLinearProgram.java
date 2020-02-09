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
package opt.multivariate.constrained.lp;

import java.util.function.Function;

import utils.BlasMath;

/**
 * This class represents a general linear programming formulation. This consists
 * of a pair(c, P), where: c is a cost vector P is a Polyhedron object.
 */
public final class RealLinearProgram implements Function<double[], Double> {

	protected final Polyhedron mySimplex;
	protected double[] myCostVec;

	/**
	 *
	 * @param polyhedron
	 * @param costVector
	 */
	public RealLinearProgram(final Polyhedron polyhedron, final double... costVector) {
		mySimplex = polyhedron;
		myCostVec = costVector;
	}

	@Override
	public final Double apply(final double[] x) {
		return BlasMath.ddotm(mySimplex.myD, myCostVec, 1, x, 1);
	}
}
