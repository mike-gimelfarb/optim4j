/*
This code is a translation of public domain software.


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
package opt.multivariate.unconstrained.order0.direct;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.unconstrained.order0.GradientFreeOptimizer;
import utils.BlasMath;
import utils.Constants;
import utils.RealMath;

/**
 *
 * REFERENCES:
 * 
 * [1] Brent, Richard P. Algorithms for minimization without derivatives.
 * Courier Corporation, 2013.
 */
public final class PraxisAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final double H0;

	// GLOBAL
	private double fx, ldt, dmin;
	private int nl, nf;

	// Q
	private double[][] v;
	private double[] q0, q1, tmp;
	private double qa, qb, qc, qd0, qd1, qf1;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param maxStepSize
	 */
	public PraxisAlgorithm(final double tolerance, final double maxStepSize) {
		super(tolerance);
		H0 = maxStepSize;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {
		// nothing to do here
	}

	@Override
	public final void iterate() {
		// nothing to do here
	}

	@Override
	public final double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		// prepare variables
		final int n = guess.length;
		final double[] x = Arrays.copyOf(guess, n);

		// call main subroutine
		praxis(myTol, Constants.EPSILON, H0, n, x, func);
		myEvals = nf;
		return x;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private double praxis(final double t0, final double machep, final double h0, final int n, final double[] x,
			final Function<? super double[], Double> f) {

		// IF N>20 OR IF N<20 AND YOU NEED MORE SPACE, CHANGE '20' TO THE
		// LARGEST VALUE OF N IN THE NEXT CARD, IN THE CARD 'IDIM=20', AND
		// IN THE DIMENSION STATEMENTS IN SUBROUTINES MINFIT,MIN,FLIN,QUAD
		v = new double[n][n];
		q0 = new double[n];
		q1 = new double[n];
		tmp = new double[n];

		boolean illc, skipto80;
		int i, ii, im1, j, k, k2, kl, klmk, km1, kt, ktm;
		double dn, dni, df, f1, h, ldfac, scbd, sf, sl, small, t, t2, vsmall, large, vlarge, m2, m4;
		final double[] d = new double[n], dk2 = new double[1], lds = new double[1], s = new double[1],
				value = new double[1], y = new double[n], z = new double[n], zeroes = new double[n];

		// INITIALIZATION.....MACHINE DEPENDENT NUMBERS:
		small = machep * machep;
		vsmall = machep * machep;
		large = 1.0 / small;
		vlarge = 1.0 / vsmall;
		m2 = Math.sqrt(machep);
		m4 = Math.sqrt(m2);

		// HEURISTIC NUMBERS:
		// IF THE AXES MAY BE BADLY SCALED (WHICH IS TO BE AVOIDED IF
		// POSSIBLE), THEN SET SCBD=10. OTHERWISE SET SCBD=1.
		// IF THE PROBLEM IS KNOWN TO BE ILL-CONDITIONED, SET ILLC=TRUE.
		// OTHERWISE SET ILLC=FALSE.
		// KTM IS THE NUMBER OF ITERATIONS WITHOUT IMPROVEMENT BEFORE THE
		// ALGORITHM TERMINATES. KTM=4 IS VERY CAUTIOUS; USUALLY KTM=1
		// IS SATISFACTORY
		scbd = 1.0;
		illc = false;
		ktm = 1;
		ldfac = 0.01;
		if (illc) {
			ldfac = 0.1;
		}
		kt = nl = 0;
		nf = 1;
		fx = f.apply(x);
		qf1 = fx;
		t = small + Math.abs(t0);
		t2 = t;
		dmin = small;
		h = h0;
		if (h < 100.0 * t) {
			h = 100.0 * t;
		}
		ldt = h;

		// THE FIRST SET OF SEARCH DIRECTIONS V IS THE IDENTITY MATRIX.....
		for (i = 1; i <= n; ++i) {
			System.arraycopy(zeroes, 0, v[i - 1], 0, n);
			v[i - 1][i - 1] = 1.0;
		}
		d[1 - 1] = qd0 = 0.0;
		System.arraycopy(x, 0, q0, 0, n);
		System.arraycopy(x, 0, q1, 0, n);

		// THE MAIN LOOP STARTS HERE.....
		while (true) {

			sf = d[1 - 1];
			d[1 - 1] = s[0] = 0.0;

			// MINIMIZE ALONG THE FIRST DIRECTION V(*,1).
			// FX MUST BE PASSED TO MIN BY VALUE
			value[0] = fx;
			min(n, 1, 2, d, s, value, false, f, x, t, machep, h);
			if (s[0] <= 0.0) {
				for (i = 1; i <= n; ++i) {
					v[i - 1][1 - 1] = -v[i - 1][1 - 1];
				}
			}
			if (sf <= 0.9 * d[1 - 1] || 0.9 * sf >= d[1 - 1]) {
				System.arraycopy(zeroes, 0, d, 1, n - 1);
			}

			// THE INNER LOOP STARTS HERE.....
			skipto80 = false;
			for (k = 2; k <= n; ++k) {

				if (!skipto80) {
					System.arraycopy(x, 0, y, 0, n);
					sf = fx;
					if (kt > 0) {
						illc = true;
					}
				}
				kl = k;
				df = 0.0;

				// A RANDOM STEP FOLLOWS (TO AVOID RESOLUTION VALLEYS).
				// PRAXIS ASSUMES THAT RANDOM RETURNS A RANDOM NUMBER UNIFORMLY
				// DISTRIBUTED IN (0,1)
				if (illc) {
					for (i = 1; i <= n; ++i) {
						s[0] = (0.1 * ldt + t2 * RealMath.pow(10.0, kt)) * (RAND.nextDouble() - 0.5);
						z[i - 1] = s[0];
						for (j = 1; j <= n; ++j) {
							x[j - 1] += s[0] * v[j - 1][i - 1];
						}
					}
					fx = f.apply(x);
					++nf;
				}

				// MINIMIZE ALONG THE "NON-CONJUGATE" DIRECTIONS V(*,K),...,V(*,N)
				for (k2 = k; k2 <= n; ++k2) {
					sl = fx;
					s[0] = 0.0;
					value[0] = fx;
					dk2[0] = d[k2 - 1];
					min(n, k2, 2, dk2, s, value, false, f, x, t, machep, h);
					d[k2 - 1] = dk2[0];
					if (illc) {
						s[0] = d[k2 - 1] * (s[0] + z[k2 - 1]) * (s[0] + z[k2 - 1]);
					} else {
						s[0] = sl - fx;
					}
					if (df <= s[0]) {
						df = s[0];
						kl = k2;
					}
				}

				if (!illc && !(df >= Math.abs((100.0 * machep) * fx))) {

					// IF THERE WAS NOT MUCH IMPROVEMENT ON THE FIRST TRY, SET
					// ILLC=TRUE AND START THE INNER LOOP AGAIN.....
					illc = true;
					skipto80 = true;
					continue;
				}

				// MINIMIZE ALONG THE "CONJUGATE" DIRECTIONS V(*,1),...,V(*,K-1)
				km1 = k - 1;
				for (k2 = 1; k2 <= km1; ++k2) {
					s[0] = 0.0;
					value[0] = fx;
					dk2[0] = d[k2 - 1];
					min(n, k2, 2, dk2, s, value, false, f, x, t, machep, h);
					d[k2 - 1] = dk2[0];
				}
				f1 = fx;
				fx = sf;
				lds[0] = 0.0;
				for (i = 1; i <= n; ++i) {
					sl = x[i - 1];
					x[i - 1] = y[i - 1];
					sl -= y[i - 1];
					y[i - 1] = sl;
					lds[0] += (sl * sl);
				}
				lds[0] = Math.sqrt(lds[0]);

				if (lds[0] > small) {

					// DISCARD DIRECTION V(*,KL).
					// IF NO RANDOM STEP WAS TAKEN, V(*,KL) IS THE "NON-CONJUGATE"
					// DIRECTION ALONG WHICH THE GREATEST IMPROVEMENT WAS MADE.....
					klmk = kl - k;
					if (klmk >= 1) {
						for (ii = 1; ii <= klmk; ++ii) {
							i = kl - ii;
							for (j = 1; j <= n; ++j) {
								v[j - 1][i + 1 - 1] = v[j - 1][i - 1];
							}
							d[i + 1 - 1] = d[i - 1];
						}
					}
					d[k - 1] = 0.0;
					for (i = 1; i <= n; ++i) {
						v[i - 1][k - 1] = y[i - 1] / lds[0];
					}

					// MINIMIZE ALONG THE NEW "CONJUGATE" DIRECTION V(*,K), WHICH IS
					// THE NORMALIZED VECTOR: (NEW X) - (0LD X).....
					value[0] = f1;
					dk2[0] = d[k - 1];
					min(n, k, 4, dk2, lds, value, true, f, x, t, machep, h);
					d[k - 1] = dk2[0];
					if (lds[0] <= 0.0) {
						lds[0] = -lds[0];
						for (i = 1; i <= n; ++i) {
							v[i - 1][k - 1] = -v[i - 1][k - 1];
						}
					}
				}
				ldt *= ldfac;
				if (ldt < lds[0]) {
					ldt = lds[0];
				}
				t2 = BlasMath.ddotm(n, x, 1, x, 1);
				t2 = m2 * Math.sqrt(t2) + t;

				// SEE WHETHER THE LENGTH OF THE STEP TAKEN SINCE STARTING THE
				// INNER LOOP EXCEEDS HALF THE TOLERANCE.....
				if (ldt > (0.5 * t2)) {
					kt = -1;
				}
				++kt;
				if (kt > ktm) {

					// RETURN.....
					return fx;
				}
				skipto80 = false;
			}
			// THE INNER LOOP ENDS HERE

			// TRY QUADRATIC EXTRAPOLATION IN CASE WE ARE IN A CURVED VALLEY
			quad(n, f, x, t, machep, h);
			dn = 0.0;
			for (i = 1; i <= n; ++i) {
				d[i - 1] = 1.0 / Math.sqrt(d[i - 1]);
				if (dn < d[i - 1]) {
					dn = d[i - 1];
				}
			}
			for (j = 1; j <= n; ++j) {
				s[0] = d[j - 1] / dn;
				for (i = 1; i <= n; ++i) {
					v[i - 1][j - 1] *= s[0];
				}
			}

			// SCALE THE AXES TO TRY TO REDUCE THE CONDITION NUMBER.....
			if (scbd > 1.0) {
				s[0] = vlarge;
				for (i = 1; i <= n; ++i) {
					sl = BlasMath.ddotm(n, v[i - 1], 1, v[i - 1], 1);
					z[i - 1] = Math.sqrt(sl);
					if (z[i - 1] < m4) {
						z[i - 1] = m4;
					}
					if (s[0] > z[i - 1]) {
						s[0] = z[i - 1];
					}
				}
				for (i = 1; i <= n; ++i) {
					sl = s[0] / z[i - 1];
					z[i - 1] = 1.0 / sl;
					if (z[i - 1] > scbd) {
						sl = 1.0 / scbd;
						z[i - 1] = scbd;
					}
					BlasMath.dscalm(n, sl, v[i - 1], 1);
				}
			}

			// CALCULATE A NEW SET OF ORTHOGONAL DIRECTIONS BEFORE REPEATING
			// THE MAIN LOOP FIRST TRANSPOSE V FOR MINFIT:
			for (i = 2; i <= n; ++i) {
				im1 = i - 1;
				for (j = 1; j <= im1; ++j) {
					s[0] = v[i - 1][j - 1];
					v[i - 1][j - 1] = v[j - 1][i - 1];
					v[j - 1][i - 1] = s[0];
				}
			}

			// CALL MINFIT TO FIND THE SINGULAR VALUE DECOMPOSITION OF V.
			// THIS GIVES THE PRINCIPAL VALUES AND PRINCIPAL DIRECTIONS OF THE
			// APPROXIMATING QUADRATIC FORM WITHOUT SQUARING THE CONDITION
			// NUMBER.....
			minfit(n, n, machep, vsmall, v, d, tmp);

			// UNSCALE THE AXES.....
			if (scbd > 1.0) {
				for (i = 1; i <= n; ++i) {
					s[0] = z[i - 1];
					BlasMath.dscalm(n, s[0], v[i - 1], 1);
				}
				for (i = 1; i <= n; ++i) {
					s[0] = 0.0;
					for (j = 1; j <= n; ++j) {
						s[0] += (v[j - 1][i - 1] * v[j - 1][i - 1]);
					}
					s[0] = Math.sqrt(s[0]);
					d[i - 1] *= s[0];
					s[0] = 1.0 / s[0];
					for (j = 1; j <= n; ++j) {
						v[j - 1][i - 1] *= s[0];
					}
				}
			}
			for (i = 1; i <= n; ++i) {
				dni = dn * d[i - 1];
				if (dni > large) {
					d[i - 1] = vsmall;
				} else if (dni < small) {
					d[i - 1] = vlarge;
				} else {
					d[i - 1] = 1.0 / (dni * dni);
				}
			}

			// SORT THE EIGENVALUES AND EIGENVECTORS.....
			sort(n, n, d, v);
			dmin = d[n - 1];
			if (dmin < small) {
				dmin = small;
			}
			illc = false;
			if (m2 * d[1 - 1] > dmin) {
				illc = true;
			}
		}
		// THE MAIN LOOP ENDS HERE.....
	}

	private void quad(final int n, final Function<? super double[], Double> f, final double[] x, final double t,
			final double machep, final double h) {
		int i;
		final double[] l = new double[1], s = new double[1], value = new double[1];
		s[0] = fx;
		fx = qf1;
		qf1 = s[0];
		qd1 = 0.0;
		for (i = 1; i <= n; ++i) {
			s[0] = x[i - 1];
			l[0] = q1[i - 1];
			x[i - 1] = l[0];
			q1[i - 1] = s[0];
			qd1 += (s[0] - l[0]) * (s[0] - l[0]);
		}
		qd1 = Math.sqrt(qd1);
		l[0] = qd1;
		s[0] = 0.0;
		if (qd0 <= 0.0 || qd1 <= 0.0 || nl < 3 * n * n) {
			fx = qf1;
			qa = 0.0;
			qb = qa;
			qc = 1.0;
		} else {
			value[0] = qf1;
			min(n, 0, 2, s, l, value, true, f, x, t, machep, h);
			qa = (l[0] * (l[0] - qd1)) / (qd0 * (qd0 + qd1));
			qb = ((l[0] + qd0) * (qd1 - l[0])) / (qd0 * qd1);
			qc = (l[0] * (l[0] + qd0)) / (qd1 * (qd0 + qd1));
		}
		qd0 = qd1;
		for (i = 1; i <= n; ++i) {
			s[0] = q0[i - 1];
			q0[i - 1] = x[i - 1];
			x[i - 1] = (qa * s[0] + qb * x[i - 1]) + qc * q1[i - 1];
		}
	}

	private void min(final int n, final int j, final int nits, final double[] d2, final double[] x1, final double[] f1,
			final boolean fk, final Function<? super double[], Double> f, final double[] x, final double t,
			final double machep, final double h) {
		boolean dz;
		int i, k;
		double d1, fm, f0, f2, m2, m4, small, s, sf1, sx1, temp, t2, x2, xm;

		small = machep * machep;
		m2 = Math.sqrt(machep);
		m4 = Math.sqrt(m2);
		sf1 = f1[0];
		sx1 = x1[0];
		k = 0;
		xm = 0.0;
		fm = f0 = fx;
		dz = d2[0] < machep;

		// FIND THE STEP SIZE...
		s = BlasMath.ddotm(n, x, 1, x, 1);
		s = Math.sqrt(s);
		temp = d2[0];
		if (dz) {
			temp = dmin;
		}
		t2 = m4 * Math.sqrt(Math.abs(fx) / temp + s * ldt) + m2 * ldt;
		s = m4 * s + t;
		if (dz && t2 > s) {
			t2 = s;
		}
		t2 = Math.max(t2, small);
		t2 = Math.min(t2, 0.01 * h);
		if (fk && f1[0] <= fm) {
			xm = x1[0];
			fm = f1[0];
		}
		if (!fk || Math.abs(x1[0]) < t2) {
			temp = 1.0;
			if (x1[0] < 0.0) {
				temp = -1.0;
			}
			x1[0] = temp * t2;
			f1[0] = flin(n, j, x1[0], f, x);
		}
		if (f1[0] <= fm) {
			xm = x1[0];
			fm = f1[0];
		}

		while (true) {

			if (dz) {

				// EVALUATE FLIN AT ANOTHER POINT AND ESTIMATE THE SECOND
				// DERIVATIVE...
				x2 = -x1[0];
				if (f0 >= f1[0]) {
					x2 = 2.0 * x1[0];
				}
				f2 = flin(n, j, x2, f, x);
				if (f2 <= fm) {
					xm = x2;
					fm = f2;
				}
				d2[0] = (x2 * (f1[0] - f0) - x1[0] * (f2 - f0)) / ((x1[0] * x2) * (x1[0] - x2));
			}

			// ESTIMATE THE FIRST DERIVATIVE AT 0...
			d1 = (f1[0] - f0) / x1[0] - x1[0] * d2[0];
			dz = true;

			// PREDICT THE MINIMUM...
			if (d2[0] > small) {
				x2 = (-0.5 * d1) / d2[0];
			} else {
				x2 = h;
				if (d1 >= 0.0) {
					x2 = -x2;
				}
			}
			if (Math.abs(x2) > h) {
				x2 = (x2 <= 0.0) ? -h : h;
			}

			boolean goto4;
			while (true) {

				// EVALUATE F AT THE PREDICTED MINIMUM...
				f2 = flin(n, j, x2, f, x);
				if (k >= nits || f2 <= f0) {
					goto4 = false;
					break;
				} else {

					// NO SUCCESS, SO TRY AGAIN...
					++k;
					if (f0 < f1[0] && (x1[0] * x2) > 0.0) {
						goto4 = true;
						break;
					} else {
						x2 *= 0.5;
					}
				}
			}
			if (!goto4) {
				break;
			}
		}

		// INCREMENT THE ONE-DIMENSIONAL SEARCH COUNTER...
		++nl;
		if (f2 <= fm) {
			fm = f2;
		} else {
			x2 = xm;
		}

		// GET A NEW ESTIMATE OF THE SECOND DERIVATIVE...
		if (Math.abs(x2 * (x2 - x1[0])) <= small) {
			if (k > 0) {
				d2[0] = 0.0;
			}
		} else {
			d2[0] = (x2 * (f1[0] - f0) - x1[0] * (fm - f0)) / ((x1[0] * x2) * (x1[0] - x2));
		}
		if (d2[0] <= small) {
			d2[0] = small;
		}
		x1[0] = x2;
		fx = fm;
		if (sf1 < fx) {
			fx = sf1;
			x1[0] = sx1;
		}

		// UPDATE X FOR LINEAR BUT NOT PARABOLIC SEARCH...
		if (j == 0) {
			return;
		}
		for (i = 1; i <= n; ++i) {
			x[i - 1] += x1[0] * v[i - 1][j - 1];
		}
	}

	private double flin(final int n, final int j, final double l, final Function<? super double[], Double> f,
			final double[] x) {
		int i;
		if (j != 0) {

			// THE SEARCH IS LINEAR...
			for (i = 1; i <= n; ++i) {
				tmp[i - 1] = x[i - 1] + l * v[i - 1][j - 1];
			}
		} else {

			// THE SEARCH IS ALONG A PARABOLIC SPACE CURVE...
			qa = (l * (l - qd1)) / (qd0 * (qd0 + qd1));
			qb = ((l + qd0) * (qd1 - l)) / (qd0 * qd1);
			qc = (l * (l + qd0)) / (qd1 * (qd0 + qd1));
			for (i = 1; i <= n; ++i) {
				tmp[i - 1] = (qa * q0[i - 1] + qb * x[i - 1]) + qc * q1[i - 1];
			}
		}

		// THE FUNCTION EVALUATION COUNTER NF IS INCREMENTED...
		++nf;
		return f.apply(tmp);
	}

	private static void sort(final int m, final int n, final double[] d, final double[][] v) {
		double s;
		int i, ip1, j, k, nm1;
		if (n == 1) {
			return;
		}
		nm1 = n - 1;
		for (i = 1; i <= nm1; ++i) {
			k = i;
			s = d[i - 1];
			ip1 = i + 1;
			for (j = ip1; j <= n; ++j) {
				if (d[j - 1] > s) {
					k = j;
					s = d[j - 1];
				}
			}
			if (k > i) {
				d[k - 1] = d[i - 1];
				d[i - 1] = s;
				for (j = 1; j <= n; ++j) {
					s = v[j - 1][i - 1];
					v[j - 1][i - 1] = v[j - 1][k - 1];
					v[j - 1][k - 1] = s;
				}
			}
		}
	}

	private static void minfit(final int m, final int n, final double machep, final double tol, final double[][] ab,
			final double[] q, final double[] e) {
		boolean skipto101, skip110;
		int i, ii, j, k = 0, kk, kt = 0, l, l2, ll2, lp1;
		double c, eps, f = 0.0, g, h, s, temp, x, y, z;

		// HOUSEHOLDER'S REDUCTION TO BIDIAGONAL FORM...
		if (n == 1) {
			q[1 - 1] = ab[1 - 1][1 - 1];
			ab[1 - 1][1 - 1] = 1.0;
			return;
		}

		eps = machep;
		g = x = 0.0;
		for (i = 1; i <= n; ++i) {
			e[i - 1] = g;
			s = 0.0;
			l = i + 1;
			for (j = i; j <= n; ++j) {
				s += (ab[j - 1][i - 1] * ab[j - 1][i - 1]);
			}
			g = 0.0;
			if (s >= tol) {
				f = ab[i - 1][i - 1];
				g = Math.sqrt(s);
				if (f >= 0.0) {
					g = -g;
				}
				h = f * g - s;
				ab[i - 1][i - 1] = f - g;
				if (l <= n) {
					for (j = l; j <= n; ++j) {
						f = 0.0;
						for (k = i; k <= n; ++k) {
							f += ab[k - 1][i - 1] * ab[k - 1][j - 1];
						}
						f /= h;
						for (k = i; k <= n; ++k) {
							ab[k - 1][j - 1] += f * ab[k - 1][i - 1];
						}
					}
				}
			}
			q[i - 1] = g;
			s = 0.0;
			if (i != n) {
				s += BlasMath.ddotm(n - l + 1, ab[i - 1], l, ab[i - 1], l);
			}
			g = 0.0;
			if (s >= tol) {
				if (i != n) {
					f = ab[i - 1][i + 1 - 1];
				}
				g = Math.sqrt(g);
				if (f >= 0.0) {
					g = -g;
				}
				h = f * g - s;
				if (i != n) {
					ab[i - 1][i + 1 - 1] = f - g;
					for (j = l; j <= n; ++j) {
						e[j - 1] = ab[i - 1][j - 1] / h;
					}
					for (j = l; j <= n; ++j) {
						s = BlasMath.ddotm(n - l + 1, ab[j - 1], l, ab[i - 1], l);
						BlasMath.daxpym(n - l + 1, s, e, l, ab[j - 1], l);
					}
				}
			}
			y = Math.abs(q[i - 1]) + Math.abs(e[i - 1]);
			if (y > x) {
				x = y;
			}
		}

		// ACCUMULATION OF RIGHT-HAND TRANSFORMATIONS...
		ab[n - 1][n - 1] = 1.0;
		g = e[n - 1];
		l = n;
		for (ii = 2; ii <= n; ++ii) {
			i = n - ii + 1;
			if (g != 0.0) {
				h = ab[i - 1][i + 1 - 1] * g;
				for (j = l; j <= n; ++j) {
					ab[j - 1][i - 1] = ab[i - 1][j - 1] / h;
				}
				for (j = l; j <= n; ++j) {
					s = 0.0;
					for (k = l; k <= n; ++k) {
						s += ab[i - 1][k - 1] * ab[k - 1][j - 1];
					}
					for (k = l; k <= n; ++k) {
						ab[k - 1][j - 1] += s * ab[k - 1][i - 1];
					}
				}
			}
			for (j = l; j <= n; ++j) {
				ab[i - 1][j - 1] = ab[j - 1][i - 1] = 0.0;
			}
			ab[i - 1][i - 1] = 1.0;
			g = e[i - 1];
			l = i;
		}

		// DIAGONALIZATION OF THE BIDIAGONAL FORM...
		eps *= x;
		skipto101 = false;
		for (kk = 1; kk <= n; ++kk) {

			if (!skipto101) {
				k = n - kk + 1;
				kt = 0;
			}
			++kt;
			if (kt > 30) {
				e[k - 1] = 0.0;
			}
			skip110 = false;
			for (ll2 = 1; ll2 <= k; ++ll2) {
				l2 = k - ll2 + 1;
				l = l2;
				if (Math.abs(e[l - 1]) <= eps) {
					skip110 = true;
					break;
				}
				if (l == 1) {
				} else if (Math.abs(q[l - 1 - 1]) <= eps) {
					skip110 = false;
					break;
				}
			}

			if (!skip110) {

				// CANCELLATION OF E(L) IF L>1...
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; ++i) {
					f = s * e[i - 1];
					e[i - 1] *= c;
					if (Math.abs(f) <= eps) {
						break;
					}
					g = q[i - 1];

					// Q(I) = H = DSQRT(G*G + F*F)...
					if (Math.abs(f) < Math.abs(g)) {
						h = Math.abs(g) * RealMath.hypot(1.0, f / g);
					} else if (f < 0.0 || f > 0.0) {
						h = Math.abs(f) * RealMath.hypot(1.0, g / f);
					} else {
						h = 0.0;
					}
					q[i - 1] = h;
					if (h == 0.0) {
						g = h = 1.0;
					}
					c = g / h;
					s = -f / h;
				}
			}

			// TEST FOR CONVERGENCE...
			z = q[k - 1];
			if (l == k) {

				// CONVERGENCE: Q(K) IS MADE NON-NEGATIVE...
				if (z < 0.0) {
					q[k - 1] = -z;
					for (j = 1; j <= n; ++j) {
						ab[j - 1][k - 1] = -ab[j - 1][k - 1];
					}
				}
				skipto101 = false;
				continue;
			}

			// SHIFT FROM BOTTOM 2*2 MINOR...
			x = q[l - 1];
			y = q[k - 1 - 1];
			g = e[k - 1 - 1];
			h = e[k - 1];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = RealMath.hypot(f, 1.0);
			temp = f - g;
			if (f >= 0.0) {
				temp = f + g;
			}
			f = ((x - z) * (x + z) + h * (y / temp - h)) / x;

			// NEXT QR TRANSFORMATION...
			c = s = 1.0;
			lp1 = l + 1;
			if (lp1 <= k) {
				for (i = lp1; i <= k; ++i) {
					g = e[i - 1];
					y = q[i - 1];
					h = s * g;
					g *= c;
					if (Math.abs(f) < Math.abs(h)) {
						z = Math.abs(h) * RealMath.hypot(1.0, f / h);
					} else if (f < 0.0 || f > 0.0) {
						z = Math.abs(f) * RealMath.hypot(1.0, h / f);
					} else {
						z = 0.0;
					}
					e[i - 1 - 1] = z;
					if (z == 0.0) {
						f = z = 1.0;
					}
					c = f / z;
					s = h / z;
					f = x * c + g * s;
					g = -x * s + g * c;
					h = y * s;
					y *= c;
					for (j = 1; j <= n; ++j) {
						x = ab[j - 1][i - 1 - 1];
						z = ab[j - 1][i - 1];
						ab[j - 1][i - 1 - 1] = x * c + z * s;
						ab[j - 1][i - 1] = -x * s + z * c;
					}
					if (Math.abs(f) < Math.abs(h)) {
						z = Math.abs(h) * RealMath.hypot(1.0, f / h);
					} else if (f < 0.0 || f > 0.0) {
						z = Math.abs(f) * RealMath.hypot(1.0, h / f);
					} else {
						z = 0.0;
					}
					q[i - 1 - 1] = z;
					if (z == 0.0) {
						f = z = 1.0;
					}
					c = f / z;
					s = h / z;
					f = c * g + s * y;
					x = -s * g + c * y;
				}
			}
			e[l - 1] = 0.0;
			e[k - 1] = f;
			q[k - 1] = x;
			skipto101 = true;
		}
	}
}
