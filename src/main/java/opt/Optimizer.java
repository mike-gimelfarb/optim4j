package opt;

import java.util.function.Function;

public abstract class Optimizer<X, Y, F extends Function<? super X, ? extends Y>> {

	// ==========================================================================
	// ABSTRACT METHODS
	// ==========================================================================
	public abstract X optimize(F function, X guess);

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param function
	 * @param guess
	 * @return
	 */
	public Y optimumValue(final F function, final X guess) {
		final X optimum = optimize(function, guess);
		if (optimum == null) {
			return null;
		} else {
			return function.apply(optimum);
		}
	}
}
