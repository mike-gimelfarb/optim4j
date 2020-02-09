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
package opt.linesearch;

import java.util.function.Function;

import utils.BlasMath;
import utils.Pair;

/**
 * This is an independent implementation of the line search based on the Strong
 * Wolfe conditions, as outlined in Nocedal and Wright (2006).
 * 
 * 
 * REFERENCES:
 * 
 * [1] Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer
 * Science & Business Media, 2006.
 */
public final class StrongWolfeLineSearch extends LineSearch {

	private final double myC2, myMaxStep, myRho;

	/**
	 *
	 * @param tolerance
	 * @param c2
	 * @param maximum
	 * @param scale
	 * @param maxIterations
	 */
	public StrongWolfeLineSearch(final double tolerance, final double c2, final double scale, final double maximum,
			final int maxIterations) {
		super(tolerance, maxIterations);
		myC2 = c2;
		myRho = scale;
		myMaxStep = maximum;
	}

	@Override
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double initial) {
		final int D = x0.length;
		final double[] wa = new double[D];

		// compute dphi(0)/da
		double dphi0 = BlasMath.ddotm(D, df0, 1, dir, 1);
		double a0 = 0.0;
		double a1 = initial;
		double y0 = f0;
		double dy0 = dphi0;
		double step = a0;
		boolean first = true;

		// main loop of line search
		while (true) {

			// compute position at a1, function and gradient values
			BlasMath.daxpy1(D, a1, dir, 1, x0, 1, wa, 1);
			final double[] df1 = df.apply(wa);
			final double y1 = f.apply(wa);
			final double dy1 = BlasMath.ddotm(D, df1, 1, dir, 1);
			++myEvals;
			++myDEvals;

			// check sufficient condition
			if (y1 > f0 + C1 * a1 * dphi0 || (y1 >= y0 && !first)) {
				step = zoom(f, df, a0, a1, y0, y1, dy0, f0, dphi0, wa, x0, dir, D);
				break;
			}

			// make sure we are not too close
			if (Math.abs(dy1) <= -myC2 * dphi0) {
				step = a1;
				break;
			}

			// check if we passed the minimum
			if (dy1 >= 0.0) {
				step = zoom(f, df, a1, a0, y1, y0, dy1, f0, dphi0, wa, x0, dir, D);
				break;
			}

			// update vectors and step size
			a0 = a1;
			a1 = Math.min(myMaxStep, myRho * a1);
			y0 = y1;
			if (first) {
				first = false;
			}
		}

		// compute the final posititon
		BlasMath.daxpy1(D, step, dir, 1, x0, 1, wa, 1);
		return new Pair<>(step, wa);
	}

	private double zoom(final Function<? super double[], Double> f, final Function<? super double[], double[]> df,
			double alo, double ahi, double ylo, double yhi, double dylo, final double phi0, final double dphi0,
			final double[] wa, final double[] x0, final double[] dir, final int D) {

		// initialize
		double atry = 0.0;
		double ar = 0.0;
		double yr = phi0;

		// updating portion of zoom algorithm in Nocedal and Wright (3.3)
		for (int i = 0; i < myMaxIters; ++i) {

			// interpolation step
			if (i == 0) {

				// quadratic interpolation
				atry = alo - 0.5 * dylo / ((yhi - ylo - dylo * (ahi - alo)) / ((ahi - alo) * (ahi - alo)));
			} else {

				// cubic interpolation
				final double dab = ahi - alo;
				final double dac = ar - alo;
				final double denom = dab * dab * dac * dac * (dab - dac);
				final double a00 = dac * dac;
				final double a01 = -dab * dab;
				final double a10 = -dac * dac * dac;
				final double a11 = dab * dab * dab;
				final double b0 = yhi - ylo - dylo * dab;
				final double b1 = yr - ylo - dylo * dac;
				final double aa = (a00 * b0 + a01 * b1) / denom;
				final double bb = (a10 * b0 + a11 * b1) / denom;
				final double disc = bb * bb - 3.0 * aa * dylo;
				atry = alo + (Math.sqrt(disc) - bb) / (3.0 * aa);
			}

			// if the interpolation step failed then bisect
			if (atry != atry || atry <= Math.min(alo, ahi) || atry >= Math.max(alo, ahi)) {
				atry = 0.5 * (alo + ahi);
			}

			// compute new point
			BlasMath.daxpy1(D, atry, dir, 1, x0, 1, wa, 1);
			final double ytry = f.apply(wa);
			++myEvals;

			// check wolfe conditions
			if (ytry > phi0 + C1 * atry * dphi0 || ytry >= ylo) {
				ar = ahi;
				yr = yhi;
				ahi = atry;
				yhi = ytry;
			} else {

				// check descent condition
				final double dytry = BlasMath.ddotm(D, df.apply(wa), 1, dir, 1);
				++myDEvals;
				if (Math.abs(dytry) <= -myC2 * dphi0) {
					break;
				} else {
					if (dytry * (ahi - alo) >= 0.0) {
						ar = ahi;
						yr = yhi;
						ahi = alo;
						yhi = ylo;
					} else {
						ar = alo;
						yr = ylo;
					}
					alo = atry;
					ylo = ytry;
					dylo = dytry;
				}
			}
		}
		return atry;
	}
}
