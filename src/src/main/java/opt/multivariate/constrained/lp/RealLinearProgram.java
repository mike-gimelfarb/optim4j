package opt.multivariate.constrained.lp;

import java.util.function.Function;

import utils.BlasMath;

/**
 *
 * @author michael
 */
public final class RealLinearProgram implements Function<double[], Double> {

	protected final Polyhedron mySimplex;
	protected double[] myCostVec;

	/**
	 *
	 * @param polyhedron
	 * @param costVector
	 */
	public RealLinearProgram(final Polyhedron polyhedron, final double... costVector) {
		mySimplex = polyhedron;
		myCostVec = costVector;
	}

	@Override
	public final Double apply(final double[] x) {
		return BlasMath.ddotm(mySimplex.myD, myCostVec, 1, x, 1);
	}
}
