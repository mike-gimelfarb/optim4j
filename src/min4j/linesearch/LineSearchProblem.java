package min4j.linesearch;

import java.util.function.Function;

import min4j.utils.BlasMath;

/**
 *
 * @author michael
 */
public final class LineSearchProblem implements Function<Double, Double> {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final Function<double[], Double> myFunc;
	protected final Function<double[], double[]> myDFunc;
	protected final double[] myX0, myD;
	private final int myN;
	private final double[] myTemp;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param func
	 * @param dfunc
	 * @param x0
	 * @param dir
	 */
	public LineSearchProblem(final Function<double[], Double> func, final Function<double[], double[]> dfunc,
			final double[] x0, final double[] dir) {
		myFunc = func;
		myDFunc = dfunc;
		myX0 = x0;
		myD = dir;
		myN = myX0.length;
		myTemp = new double[myN];
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final Double apply(final Double t) {
		BlasMath.daxpy1(myN, t, myD, 1, myX0, 1, myTemp, 1);
		return myFunc.apply(myTemp);
	}
}
