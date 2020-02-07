package opt.univariate;

import java.util.function.Function;

import opt.Optimizer;

/**
 *
 * @author Michael
 */
public abstract class UnivariateOptimizer extends Optimizer<Double, Double, Function<? super Double, Double>> {

	/**
	 *
	 * @param f
	 * @param guess
	 * @param factor
	 * @param fev
	 * @return
	 */
	public static final double[] bracket(final Function<? super Double, Double> f, final double guess,
			final double factor, final int maxfev, final int[] fev) {
		fev[0] = 0;
		double a = guess;
		double fa = f.apply(a);
		++fev[0];
		double b = a + 1.0;
		double fb = f.apply(b);
		++fev[0];
		double c, fc;
		if (fa < fb) {
			c = a;
			fc = fa;
			a = b;
			b = c;
			fb = fc;
		}
		c = b + factor * (b - a);
		fc = f.apply(c);
		++fev[0];
		if (fc <= fb) {
			while (true) {
				final double d = c + factor * (c - b);
				if (Math.abs(d) >= 1e100 || fev[0] >= maxfev) {
					return null;
				}
				final double fd = f.apply(d);
				++fev[0];
				a = b;
				b = c;
				fb = fc;
				c = d;
				fc = fd;
				if (fc > fb) {
					break;
				}
			}
		}
		return new double[] { Math.min(a, c), Math.max(a, c) };
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected int myEvals;
	protected int myMaxEvals;
	protected double myTol, myRelTol;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param absoluteTolerance
	 * @param relativeTolerance
	 * @param maxEvaluations
	 */
	public UnivariateOptimizer(final double absoluteTolerance, final double relativeTolerance,
			final int maxEvaluations) {
		myTol = absoluteTolerance;
		myRelTol = relativeTolerance;
		myMaxEvals = maxEvaluations;
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @return
	 */
	public final int countEvaluations() {
		return myEvals;
	}

	/**
	 *
	 */
	public void resetCounter() {
		myEvals = 0;
	}

	/**
	 *
	 * @param newTolerance
	 */
	public final void setTolerance(final double newTolerance) {
		myTol = newTolerance;
	}
}
