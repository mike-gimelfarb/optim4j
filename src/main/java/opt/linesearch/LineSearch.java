package opt.linesearch;

import java.util.function.Function;

import opt.Optimizer;
import utils.Pair;

/**
 *
 * @author Michael
 */
public abstract class LineSearch extends Optimizer<Double, Double, LineSearchProblem> {

	protected static final double C1 = 1e-4;

	protected final double myTol;
	protected final int myMaxIters;
	protected int myEvals, myDEvals;

	/**
	 *
	 * @param tolerance
	 * @param maxIterations
	 */
	public LineSearch(final double tolerance, final int maxIterations) {
		myTol = tolerance;
		myMaxIters = maxIterations;
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
	public abstract Pair<Double, double[]> lineSearch(Function<? super double[], Double> f,
			Function<? super double[], double[]> df, double[] x0, double[] dir, double[] df0, double f0,
			double initial);

	@Override
	public Double optimize(final LineSearchProblem problem, final Double guess) {
		final Function<double[], Double> f = problem.myFunc;
		final Function<double[], double[]> df = problem.myDFunc;
		final double[] x0 = problem.myX0;
		final double[] dir = problem.myD;
		final double[] df0 = df.apply(x0);
		final double f0 = f.apply(x0);
		++myEvals;
		++myDEvals;
		return lineSearch(f, df, x0, dir, df0, f0, guess).first();
	}

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
		return myDEvals;
	}

	/**
	 *
	 */
	public final void resetCounter() {
		myEvals = myDEvals = 0;
	}
}
