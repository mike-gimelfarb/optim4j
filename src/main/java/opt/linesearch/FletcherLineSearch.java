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
import utils.Pair;

/**
 * An inexact line search method introduced in Fletcher (1987) and described in
 * Antoniou et al. (2007).
 * 
 * 
 * REFERENCES
 * 
 * [1] Practical Methods of Optimization (Roger Fletcher), John Wiley & Sons,
 * 1987.
 * 
 * [2] Antoniou, Andreas, and Wu-Sheng Lu. Practical optimization: algorithms
 * and engineering applications. Springer Science & Business Media, 2007.
 */
public final class FletcherLineSearch extends LineSearch {

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 */
	public FletcherLineSearch(final double tolerance, final int maxIterations) {
		super(tolerance, maxIterations);
	}

	@Override
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			double f0, final double initial) {
		final double rhoscal = 0.1;
		final double sig = 0.7;
		final double tau = 0.1;
		final double chi = 0.75;
		final double eps2 = myTol;
		final int mhat = myMaxIters;
		final int n = x0.length;
		final double[] wa = new double[n];

		// INITIALIZE TEMPORARY ALGORITHM VARIABLES
		double alfa0 = initial;
		double alfal = 0.0;
		double alfau = 1e99;
		double fl = f0;
		double flp = BlasMath.ddotm(n, df0, 1, dir, 1);
		int mit = 0;
		int dmit = 0;

		// MAIN LOOP OF THE LINE SEARCH PROCEDURE STARTS HERE
		while (true) {

			// UPDATE DELTA AND COMPUTE F(X+DELTA)
			BlasMath.daxpy1(n, alfa0, dir, 1, x0, 1, wa, 1);
			f0 = f.apply(wa);
			++mit;

			if (f0 > fl + rhoscal * (alfa0 - alfal) * flp && Math.abs(fl - f0) > eps2 && mit < mhat) {

				// PERFORM INTERPOLATION
				if (alfa0 < alfau) {
					alfau = alfa0;
				}
				final double num = (alfa0 - alfal) * (alfa0 - alfal) * flp;
				final double den = 2.0 * (fl - f0 + (alfa0 - alfal) * flp);
				double alfac0 = alfal + (num / den);
				final double alfac0l = alfal + tau * (alfau - alfal);
				if (alfac0 < alfac0l) {
					alfac0 = alfac0l;
				}
				final double alfac0u = alfau - tau * (alfau - alfal);
				if (alfac0 > alfac0u) {
					alfac0 = alfac0u;
				}
				alfa0 = alfac0;
			} else {

				// COMPUTE DF(X+ALPHA_0*D)^T*D
				BlasMath.daxpy1(n, alfa0, dir, 1, x0, 1, wa, 1);
				final double f0p = BlasMath.ddotm(n, df.apply(wa), 1, dir, 1);
				++dmit;

				// PERFORM EXTRAPOLATION
				if (f0p < sig * flp && Math.abs(fl - f0) > eps2 && mit < mhat) {
					double dalfa0 = (alfa0 - alfal) * f0p / (flp - f0p);
					if (dalfa0 < tau * (alfa0 - alfal)) {
						dalfa0 = tau * (alfa0 - alfal);
					}
					if (dalfa0 > chi * (alfa0 - alfal)) {
						dalfa0 = chi * (alfa0 - alfal);
					}
					final double alfac0 = alfa0 + dalfa0;
					alfal = alfa0;
					alfa0 = alfac0;
					fl = f0;
					flp = f0p;
				} else {
					break;
				}
			}
		}

		// RETURN RESULT
		myEvals += mit;
		myDEvals += dmit;
		return new Pair<>(alfa0, wa);
	}
}
