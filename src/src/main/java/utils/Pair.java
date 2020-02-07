package utils;

import java.util.Objects;

/**
 *
 * @author Michael
 * @param <X>
 * @param <Y>
 */
public class Pair<X, Y> {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private X myX;
	private Y myY;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param coord1
	 * @param coord2
	 */
	public Pair(final X coord1, final Y coord2) {
		myX = coord1;
		myY = coord2;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final int hashCode() {
		return Objects.hash(myX, myY);
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) {
			return true;
		} else if (obj == null) {
			return false;
		} else if (getClass() != obj.getClass()) {
			return false;
		}
		final Pair<?, ?> other = (Pair<?, ?>) obj;
		if (!Objects.equals(this.myX, other.myX)) {
			return false;
		} else {
			return Objects.equals(this.myY, other.myY);
		}
	}

	@Override
	public final String toString() {
		return "(" + Objects.toString(myX) + ", \t" + Objects.toString(myY) + ")";
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @return
	 */
	public final X first() {
		return myX;
	}

	/**
	 *
	 * @return
	 */
	public final Y second() {
		return myY;
	}

	/**
	 *
	 * @param value
	 */
	public final void setFirst(final X value) {
		myX = value;
	}

	/**
	 *
	 * @param value
	 */
	public final void setSecond(final Y value) {
		myY = value;
	}
}
