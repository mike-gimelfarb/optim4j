package opt.linesearch;

import java.util.function.Function;

import utils.BlasMath;
import utils.Pair;

/**
 * A line search method that uses a constant user-defined step size. This method
 * is provided for completeness and should never be used in practical settings,
 * unless adaptive line search is not required or for benchmarking other line
 * search methods.
 * 
 * @author Michael
 */
public final class ConstantStepSizeSearch extends LineSearch {

	private final double myStepSize;

	/**
	 *
	 * @param stepSize
	 */
	public ConstantStepSizeSearch(final double stepSize) {
		super(0.0, 1);
		myStepSize = stepSize;
	}
	
	@Override
	public final Pair<Double, double[]> lineSearch(final Function<? super double[], Double> f,
			final Function<? super double[], double[]> df, final double[] x0, final double[] dir, final double[] df0,
			final double f0, final double initial) {
		final int D = x0.length;
		final double[] x = new double[D];
		BlasMath.daxpy1(D, myStepSize, dir, 1, x0, 1, x, 1);
		return new Pair<>(myStepSize, x);
	}
}
