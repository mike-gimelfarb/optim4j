package min4j.utils;

/**
 *
 * @author Michael
 */
public final class BlasMath {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	/**
	 *
	 */
	public static final double[] D1MACH = { Double.MIN_VALUE, Double.MAX_VALUE, RealMath.pow(2.0, -52),
			RealMath.pow(2.0, -51), Math.log(2.0) / Math.log(10.0) };

	// ==========================================================================
	// STATIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param n
	 * @param x
	 * @return
	 */
	public static final double denorm(final int n, final double[] x) {
		final double rdwarf = 3.834e-20, rgiant = 1.304e19, floatn = n, agiant = rgiant / floatn;
		double s1, s2, s3, x1max, x3max, xabs;
		s1 = s2 = s3 = x1max = x3max = 0.0;

		for (int i = 1; i <= n; ++i) {
			xabs = Math.abs(x[i - 1]);
			if (xabs <= rdwarf || xabs >= agiant) {
				if (xabs > rdwarf) {

					// SUM FOR LARGE COMPONENTS
					if (xabs > x1max) {
						s1 = 1.0 + s1 * (x1max / xabs) * (x1max / xabs);
						x1max = xabs;
					} else {
						s1 += (xabs / x1max) * (xabs / x1max);
					}

					// SUM FOR SMALL COMPONENTS
				} else if (xabs > x3max) {
					s3 = 1.0 + s3 * (x3max / xabs) * (x3max / xabs);
					x3max = xabs;
				} else if (xabs != 0.0) {
					s3 += (xabs / x3max) * (xabs / x3max);
				}
				continue;
			}

			// SUM FOR INTERMEDIATE COMPONENTS
			s2 += xabs * xabs;
		}

		// CALCULATION OF NORM
		if (s1 != 0.0) {
			return x1max * Math.sqrt(s1 + (s2 / x1max) / x1max);
		} else if (s2 != 0.0) {
			if (s2 >= x3max) {
				return Math.sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)));
			} else {
				return Math.sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
			}
		} else {
			return x3max * Math.sqrt(s3);
		}
	}

	/**
	 *
	 * @param m
	 * @param n
	 * @param a
	 * @param lda
	 * @param pivot
	 * @param ipvt
	 * @param lipvt
	 * @param sigma
	 * @param acnorm
	 * @param wa
	 */
	public static final void dqrfac(final int m, final int n, final double[][] a, final int lda, final boolean pivot,
			final int[] ipvt, final int lipvt, final double[] sigma, final double[] acnorm, final double[] wa) {
		int i, j, jp1, k, kmax, minmn, l;
		double ajnorm, epsmch, p05 = 5.0e-2, sum, temp;
		final double[] cola = new double[m];

		// epsmch is the machine precision
		epsmch = D1MACH[4 - 1];

		// COMPUTE THE INITIAL COLUMN NORMS AND INITIALIZE SEVERAL ARRAYS
		for (j = 1; j <= n; ++j) {
			for (l = 1; l <= m; ++l) {
				cola[l - 1] = a[l - 1][j - 1];
			}
			acnorm[j - 1] = denorm(m, cola);
			sigma[j - 1] = acnorm[j - 1];
			wa[j - 1] = sigma[j - 1];
			if (pivot) {
				ipvt[j - 1] = j;
			}
		}

		// REDUCE A TO R WITH HOUSEHOLDER TRANSFORMATIONS
		minmn = Math.min(m, n);
		for (j = 1; j <= minmn; ++j) {

			if (pivot) {

				// BRING THE COLUMN OF LARGEST NORM INTO THE PIVOT POSITION
				kmax = j;
				for (k = j; k <= n; ++k) {
					if (sigma[k - 1] > sigma[kmax - 1]) {
						kmax = k;
					}
				}
				if (kmax != j) {
					for (i = 1; i <= m; ++i) {
						temp = a[i - 1][j - 1];
						a[i - 1][j - 1] = a[i - 1][kmax - 1];
						a[i - 1][kmax - 1] = temp;
					}
					sigma[kmax - 1] = sigma[j - 1];
					wa[kmax - 1] = wa[j - 1];
					k = ipvt[j - 1];
					ipvt[j - 1] = ipvt[kmax - 1];
					ipvt[kmax - 1] = k;
				}
			}

			// COMPUTE THE HOUSEHOLDER TRANSFORMATION TO REDUCE THE
			// J-TH COLUMN OF A TO A MULTIPLE OF THE J-TH UNIT VECTOR
			for (l = j; l <= m; ++l) {
				cola[l - j] = a[l - 1][j - 1];
			}
			ajnorm = denorm(m - j + 1, cola);
			if (ajnorm == 0.0) {
				sigma[j - 1] = -ajnorm;
				continue;
			}
			if (a[j - 1][j - 1] < 0.0) {
				ajnorm = -ajnorm;
			}
			for (i = j; i <= m; ++i) {
				a[i - 1][j - 1] /= ajnorm;
			}
			a[j - 1][j - 1] += 1.0;

			// APPLY THE TRANSFORMATION TO THE REMAINING COLUMNS
			// AND UPDATE THE NORMS
			jp1 = j + 1;
			if (n >= jp1) {
				for (k = jp1; k <= n; ++k) {
					sum = 0.0;
					for (i = j; i <= m; ++i) {
						sum += a[i - 1][j - 1] * a[i - 1][k - 1];
					}
					temp = sum / a[j - 1][j - 1];
					for (i = j; i <= m; ++i) {
						a[i - 1][k - 1] -= temp * a[i - 1][j - 1];
					}
					if (!pivot || sigma[k - 1] == 0.0) {
						continue;
					}
					temp = a[j - 1][k - 1] / sigma[k - 1];
					sigma[k - 1] *= Math.sqrt(Math.max(0.0, 1.0 - temp * temp));
					if (p05 * (sigma[k - 1] / wa[k - 1]) * (sigma[k - 1] / wa[k - 1]) > epsmch) {
						continue;
					}
					for (l = jp1; l <= m; ++l) {
						cola[l - jp1] = a[l - 1][k - 1];
					}
					sigma[k - 1] = denorm(m - j, cola);
					wa[k - 1] = sigma[k - 1];
				}
			}
			sigma[j - 1] = -ajnorm;
		}
	}

	/**
	 *
	 * @param n
	 * @param dx
	 * @param idx
	 * @param dy
	 * @param idy
	 * @return
	 */
	public static final double ddotm(final int n, final double[] dx, final int idx, final double[] dy, final int idy) {
		if (n <= 0) {
			return 0.0;
		}

		// CODE FOR BOTH INCREMENTS EQUAL TO 1
		double dtemp = 0.0;
		final int m = n % 5;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dtemp += (dx[i + idx - 1] * dy[i + idy - 1]);
			}
			if (n < 5) {
				return dtemp;
			}
		}
		for (int i = m; i < n; i += 5) {
			dtemp += (dx[i + 0 + idx - 1] * dy[i + 0 + idy - 1] + dx[i + 1 + idx - 1] * dy[i + 1 + idy - 1]
					+ dx[i + 2 + idx - 1] * dy[i + 2 + idy - 1] + dx[i + 3 + idx - 1] * dy[i + 3 + idy - 1]
					+ dx[i + 4 + idx - 1] * dy[i + 4 + idy - 1]);
		}
		return dtemp;
	}

	/**
	 *
	 * @param n
	 * @param da
	 * @param dx
	 * @param idx
	 */
	public static final void dscalm(final int n, final double da, final double[] dx, final int idx) {
		if (n <= 0) {
			return;
		}

		// code for increment equal to 1
		final int m = n % 5;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dx[i + idx - 1] *= da;
			}
			if (n < 5) {
				return;
			}
		}
		for (int i = m; i < n; i += 5) {
			dx[i + 0 + idx - 1] *= da;
			dx[i + 1 + idx - 1] *= da;
			dx[i + 2 + idx - 1] *= da;
			dx[i + 3 + idx - 1] *= da;
			dx[i + 4 + idx - 1] *= da;
		}
	}

	/**
	 *
	 * @param n
	 * @param da
	 * @param dx
	 * @param idx
	 * @param dy
	 * @param idy
	 */
	public static final void dscal1(final int n, final double da, final double[] dx, final int idx, final double[] dy,
			final int idy) {
		if (n <= 0) {
			return;
		}

		// code for increment equal to 1
		final int m = n % 5;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dy[i + idy - 1] = dx[i + idx - 1] * da;
			}
			if (n < 5) {
				return;
			}
		}
		for (int i = m; i < n; i += 5) {
			dy[i + 0 + idy - 1] = dx[i + 0 + idx - 1] * da;
			dy[i + 1 + idy - 1] = dx[i + 1 + idx - 1] * da;
			dy[i + 2 + idy - 1] = dx[i + 2 + idx - 1] * da;
			dy[i + 3 + idy - 1] = dx[i + 3 + idx - 1] * da;
			dy[i + 4 + idy - 1] = dx[i + 4 + idx - 1] * da;
		}
	}

	/**
	 *
	 * @param n
	 * @param da
	 * @param dx
	 * @param idx
	 * @param dy
	 * @param idy
	 */
	public static final void daxpym(final int n, final double da, final double[] dx, final int idx, final double[] dy,
			final int idy) {
		if (n <= 0 || da == 0.0) {
			return;
		}

		// CODE FOR BOTH INCREMENTS EQUAL TO 1
		final int m = n % 4;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dy[i + idy - 1] += da * dx[i + idx - 1];
			}
			if (n < 4) {
				return;
			}
		}
		for (int i = m; i < n; i += 4) {
			dy[i + 0 + idy - 1] += da * dx[i + 0 + idx - 1];
			dy[i + 1 + idy - 1] += da * dx[i + 1 + idx - 1];
			dy[i + 2 + idy - 1] += da * dx[i + 2 + idx - 1];
			dy[i + 3 + idy - 1] += da * dx[i + 3 + idx - 1];
		}
	}

	/**
	 *
	 * @param n
	 * @param da
	 * @param dx
	 * @param idx
	 * @param dy
	 * @param idy
	 * @param dz
	 * @param idz
	 */
	public static final void daxpy1(final int n, final double da, final double[] dx, final int idx, final double[] dy,
			final int idy, final double[] dz, final int idz) {
		if (n <= 0) {
			return;
		}
		if (da == 0.0) {

			// COPY Y INTO Z
			System.arraycopy(dy, idx - 1, dz, idz - 1, n);
			return;
		}

		// CODE FOR BOTH INCREMENTS EQUAL TO 1
		final int m = n % 4;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dz[i + idz - 1] = dy[i + idy - 1] + da * dx[i + idx - 1];
			}
			if (n < 4) {
				return;
			}
		}
		for (int i = m; i < n; i += 4) {
			dz[i + 0 + idz - 1] = dy[i + 0 + idy - 1] + da * dx[i + 0 + idx - 1];
			dz[i + 1 + idz - 1] = dy[i + 1 + idy - 1] + da * dx[i + 1 + idx - 1];
			dz[i + 2 + idz - 1] = dy[i + 2 + idy - 1] + da * dx[i + 2 + idx - 1];
			dz[i + 3 + idz - 1] = dy[i + 3 + idy - 1] + da * dx[i + 3 + idx - 1];
		}
	}

	/**
	 *
	 * @param n
	 * @param dx
	 * @param idx
	 * @param incx
	 * @return
	 */
	public static final double dnrm2(final int n, final double[] dx, final int idx, final int incx) {
		if (n <= 0) {
			return 0.0;
		}

		double cutlo = 8.23181e-11, cuthi = 1.30438e19, hitest, sum = 0.0, xmax = 0.0;
		int i = 1, j, next = 30, nn = n * incx;

		// BEGIN MAIN LOOP
		while (true) {

			// main switch
			int gotoflag;
			if (next == 30) {
				if (Math.abs(dx[i - 1 + idx - 1]) > cutlo) {
					gotoflag = 85;
				} else {
					next = 50;
					xmax = 0.0;
					gotoflag = 50;
				}
			} else if (next == 50) {
				gotoflag = 50;
			} else if (next == 70) {

				// PHASE 2. SUM IS SMALL. SCALE TO AVOID DESTRUCTIVE UNDERFLOW
				if (Math.abs(dx[i - 1 + idx - 1]) <= cutlo) {
					gotoflag = 110;
				} else {

					// PREPARE FOR PHASE 3
					sum = (sum * xmax) * xmax;
					gotoflag = 85;
				}
			} else {
				gotoflag = 110;
			}

			// main logic
			if (gotoflag <= 50) {

				// PHASE 1. SUM IS ZERO
				if (dx[i - 1 + idx - 1] == 0.0) {
					i += incx;
					if (i <= nn) {
						continue;
					} else {
						return xmax * Math.sqrt(sum);
					}
				}
				if (Math.abs(dx[i - 1 + idx - 1]) > cutlo) {
					gotoflag = 85;
				} else {

					// PREPARE FOR PHASE 2
					next = 70;
					xmax = Math.abs(dx[i - 1 + idx - 1]);
					sum += (dx[i - 1 + idx - 1] / xmax) * (dx[i - 1 + idx - 1] / xmax);
					i += incx;
					if (i <= nn) {
						continue;
					} else {
						return xmax * Math.sqrt(sum);
					}
				}
			}

			if (gotoflag <= 85) {
				hitest = cuthi / n;

				// PHASE 3. SUM IS MID-RANGE. NO SCALING
				boolean skip = false;
				for (j = i; j <= nn; j += incx) {
					if (Math.abs(dx[j - 1 + idx - 1]) >= hitest) {

						// PREPARE FOR PHASE 4
						i = j;
						next = 110;
						sum = (sum / dx[i - 1 + idx - 1]) / dx[i - 1 + idx - 1];
						xmax = Math.abs(dx[i - 1 + idx - 1]);
						sum += (dx[i - 1 + idx - 1] / xmax) * (dx[i - 1 + idx - 1] / xmax);
						i += incx;
						if (i <= nn) {
							skip = true;
							break;
						} else {
							return xmax * Math.sqrt(sum);
						}
					}
					sum += (dx[j - 1 + idx - 1] * dx[j - 1 + idx - 1]);
				}
				if (skip) {
					continue;
				} else {
					return Math.sqrt(sum);
				}
			}

			if (gotoflag <= 110) {

				// COMMON CODE FOR PHASES 2 AND 4.
				// IN PHASE 4 SUM IS LARGE. SCALE TO AVOID OVERFLOW
				if (Math.abs(dx[i - 1 + idx - 1]) <= xmax) {
					sum += (dx[i - 1 + idx - 1] / xmax) * (dx[i - 1 + idx - 1] / xmax);
				} else {
					sum = 1.0 + sum * (xmax / dx[i - 1 + idx - 1]) * (xmax / dx[i - 1 + idx - 1]);
					xmax = Math.abs(dx[i - 1 + idx - 1]);
				}
				i += incx;
				if (i > nn) {
					return xmax * Math.sqrt(sum);
				}
			}
		}
	}

	/**
	 *
	 * @param n
	 * @param dx
	 * @param idx
	 * @param dy
	 * @param idy
	 */
	public static final void dxpym(final int n, final double[] dx, final int idx, final double[] dy, final int idy) {
		if (n <= 0) {
			return;
		}

		// CODE FOR BOTH INCREMENTS EQUAL TO 1
		final int m = n % 4;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dy[i + idy - 1] += dx[i + idx - 1];
			}
			if (n < 4) {
				return;
			}
		}
		for (int i = m; i < n; i += 4) {
			dy[i + idy - 1] += dx[i + idx - 1];
			dy[i + 1 + idy - 1] += dx[i + 1 + idx - 1];
			dy[i + 2 + idy - 1] += dx[i + 2 + idx - 1];
			dy[i + 3 + idy - 1] += dx[i + 3 + idx - 1];
		}
	}

	/**
	 *
	 * @param n
	 * @param dx
	 * @param idx
	 * @param dy
	 * @param idy
	 * @param dz
	 * @param idz
	 */
	public static final void dxpy1(final int n, final double[] dx, final int idx, final double[] dy, final int idy,
			final double[] dz, final int idz) {
		if (n <= 0) {
			return;
		}

		// CODE FOR BOTH INCREMENTS EQUAL TO 1
		final int m = n % 4;
		if (m != 0) {
			for (int i = 0; i < m; ++i) {
				dz[i + idz - 1] = dy[i + idy - 1] + dx[i + idx - 1];
			}
			if (n < 4) {
				return;
			}
		}
		for (int i = m; i < n; i += 4) {
			dz[i + idz - 1] = dy[i + idy - 1] + dx[i + idx - 1];
			dz[i + 1 + idy - 1] = dy[i + 1 + idy - 1] + dx[i + 1 + idx - 1];
			dz[i + 2 + idy - 1] = dy[i + 2 + idy - 1] + dx[i + 2 + idx - 1];
			dz[i + 3 + idy - 1] = dy[i + 3 + idy - 1] + dx[i + 3 + idx - 1];
		}
	}

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	private BlasMath() {
	}
}
