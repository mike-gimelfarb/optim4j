package opt.multivariate.unconstrained.order1;

import java.util.function.Function;

import opt.Optimizer;
import utils.BlasMath;
import utils.Constants;

/**
 *
 * @author Michael
 */
public abstract class GradientOptimizer extends Optimizer<double[], Double, Function<? super double[], Double>> {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	public static final int MAX_ITERS = (int) 1e6;

	protected static final double TINY = 1e-60;

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final double myTol;
	protected int myEvals, myGEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 */
	public GradientOptimizer(final double tolerance) {
		myTol = tolerance;
	}

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	/**
	 * 
	 * @param f
	 * @param df
	 * @param guess
	 * @return
	 */
	public abstract double[] optimize(Function<? super double[], Double> f, Function<? super double[], double[]> df,
			double[] guess);

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	/**
	 *
	 * @param f
	 * @param guess
	 * @return
	 */
	public double[] optimize(final Function<? super double[], Double> f, final double[] guess) {

		// use forward differencing of function
		final int[] fevs = new int[1];
		final Function<double[], double[]> df = (x) -> {
			final int n = x.length;
			final double fvec = f.apply(x);
			final double[] g = new double[n];
			dfdjcmod(f, n, x, fvec, g, Constants.EPSILON, new double[1]);
			fevs[0] += (n + 1);
			return g;
		};
		final double[] result = optimize(f, df, guess);
		myEvals += fevs[0];
		return result;
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
	 * @return
	 */
	public final int countGradientEvaluations() {
		return myGEvals;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static void dfdjcmod(final Function<? super double[], Double> fcn, final int n, final double[] x,
			final double fvec, final double[] fjac, final double epsfcn, final double[] wa) {
		final double epsmch = BlasMath.D1MACH[4 - 1];
		final double eps = Math.sqrt(Math.max(epsfcn, epsmch));
		for (int j = 1; j <= n; ++j) {
			final double temp = x[j - 1];
			double h = eps * Math.abs(temp);
			if (h == 0.0) {
				h = eps;
			}
			x[j - 1] = temp + h;
			wa[0] = fcn.apply(x);
			x[j - 1] = temp;
			fjac[j - 1] = (wa[0] - fvec) / h;
		}
	}
}
