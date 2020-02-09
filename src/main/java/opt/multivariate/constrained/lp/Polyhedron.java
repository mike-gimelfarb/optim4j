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

/**
 * A polyhedron defined by a set of constraints of the form >= and <=.
 */
public class Polyhedron {

	private final double[][] myA;
	private final double[] myB;
	public final int myD, myNumLe, myNumGe;

	/**
	 *
	 * @param amat
	 * @param bvec
	 * @param numlesseq
	 * @param numgreeq
	 */
	public Polyhedron(final double[][] amat, final double[] bvec, final int numlesseq, final int numgreeq) {
		myA = amat;
		myB = bvec;
		myNumLe = numlesseq;
		myNumGe = numgreeq;
		myD = amat[0].length;
	}

	/**
	 *
	 * @return
	 */
	public final double[][] getA() {
		return myA;
	}

	/**
	 *
	 * @return
	 */
	public final double[] getB() {
		return myB;
	}
}
