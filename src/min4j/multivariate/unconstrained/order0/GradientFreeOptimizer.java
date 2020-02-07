package min4j.multivariate.unconstrained.order0;

import java.util.Random;
import java.util.function.Function;

import min4j.Optimizer;
import min4j.utils.Constants;

/**
 *
 * @author Michael
 */
public abstract class GradientFreeOptimizer extends Optimizer<double[], Double, Function<? super double[], Double>> {

	// ==========================================================================
	// STATIC FIELDS
	// ==========================================================================
	protected static final Random RAND = new Random();
	protected static final double RELEPS = Constants.EPSILON;

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	protected final double myTol;
	protected int myEvals;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 */
	public GradientFreeOptimizer(final double tolerance) {
		myTol = tolerance;
	}

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	public abstract void initialize(Function<? super double[], Double> func, double[] guess);

	public abstract void iterate();

	public abstract double[] optimize(Function<? super double[], Double> func, double[] guess);

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

	/*
	
	 */
	public final void resetCounter() {
		myEvals = 0;
	}
}
