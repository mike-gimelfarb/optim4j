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
package opt;

/**
 * 
 *
 * @param <X>
 * @param <Y>
 */
public class OptimizerSolution<X, Y> {

	protected final X mySol;
	protected int myFEvals;
	protected int myDEvals;
	protected final boolean myConverged;

	/**
	 * 
	 * @param sol
	 * @param value
	 * @param fevals
	 * @param dfevals
	 * @param converged
	 */
	public OptimizerSolution(final X sol, final int fevals, final int dfevals, final boolean converged) {
		mySol = sol;
		myFEvals = fevals;
		myDEvals = dfevals;
		myConverged = converged;
	}

	/**
	 * 
	 * @return
	 */
	public X getOptimalPoint() {
		return mySol;
	}

	/**
	 * 
	 * @return
	 */
	public int getFEvals() {
		return myFEvals;
	}

	/**
	 * 
	 * @return
	 */
	public int getDFEvals() {
		return myDEvals;
	}

	/**
	 * 
	 * @return
	 */
	public boolean converged() {
		return myConverged;
	}
}
