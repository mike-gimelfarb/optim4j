/*
Copyright (c) 2020 Mike Gimelfarb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the > "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, > subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package utils;

import java.util.Objects;

/**
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
