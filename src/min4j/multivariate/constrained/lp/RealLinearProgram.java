package min4j.multivariate.constrained.lp;

import java.util.function.Function;

import min4j.utils.BlasMath;

/**
 *
 * @author michael
 */
public final class RealLinearProgram implements Function<double[], Double> {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final Polyhedron mySimplex;
	protected double[] myCostVec;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param polyhedron
	 * @param costVector
	 */
	public RealLinearProgram(final Polyhedron polyhedron, final double... costVector) {
		mySimplex = polyhedron;
		myCostVec = costVector;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final Double apply(final double[] x) {
		return BlasMath.ddotm(mySimplex.myD, myCostVec, 1, x, 1);
	}
}
