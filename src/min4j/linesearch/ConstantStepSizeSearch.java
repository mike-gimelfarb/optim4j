package min4j.linesearch;

import java.util.function.Function;

import min4j.utils.BlasMath;
import min4j.utils.Pair;

/**
 *
 * @author Michael
 */
public final class ConstantStepSizeSearch extends LineSearch {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double myStepSize;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param stepSize
	 */
	public ConstantStepSizeSearch(final double stepSize) {
		super(0.0, 1);
		myStepSize = stepSize;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
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
