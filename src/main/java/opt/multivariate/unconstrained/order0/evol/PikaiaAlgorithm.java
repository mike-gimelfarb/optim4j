/*
 * The original version of this PIKAIA software is public domain software
 * written by the High Altitude Observatory and available here:
 * https://www.hao.ucar.edu/modeling/pikaia/pikaia.php#sec4
 * 
 * 
 * Copyright (c) 2020 Mike Gimelfarb
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the > "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, > subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package opt.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.GradientFreeOptimizer;
import opt.multivariate.MultivariateOptimizerSolution;
import utils.RealMath;

/**
 * 
 * REFERENCES:
 * 
 * [1] https://www.hao.ucar.edu/modeling/pikaia/pikaia.php#sec4
 */
public class PikaiaAlgorithm extends GradientFreeOptimizer {

	@FunctionalInterface
	private interface FF {

		double ff(int n, double[] x);
	}

	private static final double[] DFAULT = { 100, 500, 5, 0.85, 2, 0.005, 0.0005, 0.25, 1, 1, 1, 0 };
	private static final int NMAX = 32, PMAX = 128, DMAX = 6;

	// Local variables
	private final int[] np = new int[1], nd = new int[1], ngen = new int[1], imut = new int[1], irep = new int[1],
			ielite = new int[1], ivrb = new int[1], ip1 = new int[1], ip2 = new int[1], nnew = new int[1],
			newtot = new int[1], status = new int[1];
	private int k, ip, ig;
	private final double[] pcross = new double[1], pmut = new double[1], pmutmn = new double[1], pmutmx = new double[1],
			fdif = new double[1];
	private final double[][] ph = new double[2][NMAX], oldph = new double[PMAX][NMAX], newph = new double[PMAX][NMAX];
	private final double[] fitns = new double[PMAX];
	private final int[] gn1 = new int[NMAX * DMAX], gn2 = new int[NMAX * DMAX], ifit = new int[PMAX],
			jfit = new int[PMAX];

	private FF ff;
	private int n;
	private double[] ctrl;
	private int myEvals = 0;

	/**
	 *
	 * @param popSize
	 * @param maxGens
	 */
	public PikaiaAlgorithm(final int popSize, final int maxGens) {
		this(popSize, maxGens, DFAULT[4 - 1], DFAULT[6 - 1], DFAULT[7 - 1], DFAULT[8 - 1]);
	}

	/**
	 *
	 * @param popSize
	 * @param maxGens
	 * @param crossRate
	 * @param initMutateRate
	 * @param minMutateRate
	 * @param maxMutateRate
	 */
	public PikaiaAlgorithm(final int popSize, final int maxGens, final double crossRate, final double initMutateRate,
			final double minMutateRate, final double maxMutateRate) {
		this(popSize, maxGens, crossRate, (int) DFAULT[5 - 1], initMutateRate, minMutateRate, maxMutateRate,
				(int) DFAULT[10 - 1], (int) DFAULT[11 - 1]);
	}

	/**
	 *
	 * @param popSize
	 * @param maxGens
	 * @param crossRate
	 * @param mutationMode
	 * @param initMutateRate
	 * @param minMutateRate
	 * @param maxMutateRate
	 * @param reproductionPlan
	 * @param elitismFlag
	 */
	public PikaiaAlgorithm(final int popSize, final int maxGens, final double crossRate, final int mutationMode,
			final double initMutateRate, final double minMutateRate, final double maxMutateRate,
			final int reproductionPlan, final int elitismFlag) {
		super(1e-6);
		ctrl = new double[12];
		ctrl[1 - 1] = popSize;
		ctrl[2 - 1] = maxGens;
		ctrl[3 - 1] = 6;
		ctrl[4 - 1] = crossRate;
		ctrl[5 - 1] = mutationMode;
		ctrl[6 - 1] = initMutateRate;
		ctrl[7 - 1] = minMutateRate;
		ctrl[8 - 1] = maxMutateRate;
		ctrl[9 - 1] = 1.0;
		ctrl[10 - 1] = reproductionPlan;
		ctrl[11 - 1] = elitismFlag;
		ctrl[12 - 1] = 0;
	}

	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {

		n = guess.length;
		ff = (pn, px) -> -func.apply(px);

		// Set control variables from input and defaults
		setctl(ctrl, n, np, ngen, nd, pcross, pmutmn, pmutmx, pmut, imut, fdif, irep, ielite, ivrb, status);
		if (status[0] != 0) {
			return;
		}

		// Make sure locally-dimensioned arrays are big enough
		if (n > NMAX || np[0] > PMAX || nd[0] > DMAX) {
			status[0] = -1;
			return;
		}

		// Compute initial (random but bounded) phenotypes
		for (ip = 1; ip <= np[0]; ++ip) {
			for (k = 1; k <= n; ++k) {
				oldph[ip - 1][k - 1] = RAND.nextDouble();
			}
			fitns[ip - 1] = ff.ff(n, oldph[ip - 1]);
		}
		myEvals = np[0];

		// Rank initial population by fitness order
		rnkpop(np[0], fitns, ifit, jfit);
	}

	@Override
	public final void iterate() {

		// Main Population Loop
		newtot[0] = 0;
		for (ip = 1; ip <= np[0] / 2; ++ip) {

			// 1. pick two parents
			select(np[0], jfit, fdif[0], ip1);
			do {
				select(np[0], jfit, fdif[0], ip2);
			} while (ip1[0] == ip2[0]);

			// 2. encode parent phenotypes
			encode(n, nd[0], oldph[ip1[0] - 1], gn1);
			encode(n, nd[0], oldph[ip2[0] - 1], gn2);

			// 3. breed
			cross(n, nd[0], pcross[0], gn1, gn2);
			mutate(n, nd[0], pmut[0], gn1, imut[0]);
			mutate(n, nd[0], pmut[0], gn2, imut[0]);

			// 4. decode offspring genotypes
			decode(n, nd[0], gn1, ph[1 - 1]);
			decode(n, nd[0], gn2, ph[2 - 1]);

			// 5. insert into population
			if (irep[0] == 1) {
				genrep(NMAX, n, np[0], ip, ph, newph);
			} else {
				final int[] fev = new int[1];
				stdrep(ff, NMAX, n, np[0], irep[0], ielite[0], ph, oldph, fitns, ifit, jfit, nnew, fev);
				myEvals += fev[0];
				newtot[0] += nnew[0];
			}

			// End of Main Population Loop
		}

		// if running full generational replacement: swap populations
		if (irep[0] == 1) {
			final int[] fev = new int[1];
			newpop(ff, ielite[0], NMAX, n, np[0], oldph, newph, ifit, jfit, fitns, newtot, fev);
			myEvals += fev[0];
		}

		// adjust mutation rate?
		if (imut[0] == 2 || imut[0] == 3 || imut[0] == 5 || imut[0] == 6) {
			adjmut(NMAX, n, np[0], oldph, fitns, ifit, pmutmn[0], pmutmx[0], pmut, imut[0]);
		}
	}

	@Override
	public MultivariateOptimizerSolution optimize(final Function<? super double[], Double> func, final double[] guess) {
		initialize(func, guess);

		// Main Generation Loop
		for (ig = 1; ig <= ngen[0]; ++ig) {
			iterate();
			if (status[0] != 0) {
				break;
			}
		}

		// Return best phenotype and its fitness
		// TODO: check convergence
		return new MultivariateOptimizerSolution(Arrays.copyOf(oldph[ifit[np[0] - 1] - 1], n), myEvals, 0, false);
	}

	private static void setctl(final double[] ctrl, final int n, final int[] np, final int[] ngen, final int[] nd,
			final double[] pcross, final double[] pmutmn, final double[] pmutmx, final double[] pmut, final int[] imut,
			final double[] fdif, final int[] irep, final int[] ielite, final int[] ivrb, final int[] status) {
		int i;
		for (i = 1; i <= 12; ++i) {
			if (ctrl[i - 1] < 0.0) {
				ctrl[i - 1] = DFAULT[i - 1];
			}
		}

		np[0] = (int) ctrl[1 - 1];
		ngen[0] = (int) ctrl[2 - 1];
		nd[0] = (int) ctrl[3 - 1];
		pcross[0] = ctrl[4 - 1];
		imut[0] = (int) ctrl[5 - 1];
		pmut[0] = ctrl[6 - 1];
		pmutmn[0] = ctrl[7 - 1];
		pmutmx[0] = ctrl[8 - 1];
		fdif[0] = ctrl[9 - 1];
		irep[0] = (int) ctrl[10 - 1];
		ielite[0] = (int) ctrl[11 - 1];
		ivrb[0] = (int) ctrl[12 - 1];
		status[0] = 0;

		// Check some control values
		if (imut[0] != 1 && imut[0] != 2 && imut[0] != 3 && imut[0] != 4 && imut[0] != 5 && imut[0] != 6) {
			status[0] = 5;
		}
		if (fdif[0] > 1.0) {
			status[0] = 9;
		}
		if (irep[0] != 1 && irep[0] != 2 && irep[0] != 3) {
			status[0] = 10;
		}
		if (pcross[0] > 1.0 || pcross[0] < 0.0) {
			status[0] = 4;
		}
		if (ielite[0] != 0 && ielite[0] != 1) {
			status[0] = 11;
		}
		if (np[0] % 2 > 0) {
			--np[0];
		}
	}

	// c**********************************************************************
	// c GENETICS MODULE
	// c**********************************************************************
	// c
	// c ENCODE: encodes phenotype into genotype
	// c called by: PIKAIA
	// c
	// c DECODE: decodes genotype into phenotype
	// c called by: PIKAIA
	// c
	// c CROSS: Breeds two offspring from two parents
	// c called by: PIKAIA
	// c
	// c MUTATE: Introduces random mutation in a genotype
	// c called by: PIKAIA
	// c
	// c ADJMUT: Implements variable mutation rate
	// c called by: PIKAIA
	// c
	private static void encode(final int n, final int nd, final double[] ph, final int[] gn) {
		double z = RealMath.pow(10.0, nd);
		int ii = 0;
		for (int i = 1; i <= n; ++i) {
			int ip = (int) (ph[i - 1] * z);
			for (int j = nd; j >= 1; --j) {
				gn[ii + j - 1] = ip % 10;
				ip /= 10;
			}
			ii += nd;
		}
	}

	private static void decode(final int n, final int nd, final int[] gn, final double[] ph) {
		double z = RealMath.pow(10.0, -nd);
		int ii = 0;
		for (int i = 1; i <= n; ++i) {
			int ip = 0;
			for (int j = 1; j <= nd; ++j) {
				ip = 10 * ip + gn[ii + j - 1];
			}
			ph[i - 1] = ip * z;
			ii += nd;
		}
	}

	private static void cross(final int n, final int nd, final double pcross, final int[] gn1, final int[] gn2) {

		// Local:
		int i, ispl, ispl2, itmp, t;

		// Use crossover probability to decide whether a crossover occurs
		if (RAND.nextDouble() < pcross) {

			// Compute first crossover point
			ispl = (int) (RAND.nextDouble() * n * nd) + 1;

			// Now choose between one-point and two-point crossover
			if (RAND.nextDouble() < 0.5) {
				ispl2 = n * nd;
			} else {
				ispl2 = (int) (RAND.nextDouble() * n * nd) + 1;

				// Un-comment following line to enforce one-point crossover
				// ispl2=n*nd;
				if (ispl2 < ispl) {
					itmp = ispl2;
					ispl2 = ispl;
					ispl = itmp;
				}
			}

			// Swap genes from ispl to ispl2
			for (i = ispl; i <= ispl2; ++i) {
				t = gn2[i - 1];
				gn2[i - 1] = gn1[i - 1];
				gn1[i - 1] = t;
			}
		}
	}

	private static void mutate(final int n, final int nd, final double pmut, final int[] gn, final int imut) {

		// Local:
		int i, j, k, l, ist, inc, loc;

		// Decide which type of mutation is to occur
		if (imut >= 4 && RAND.nextDouble() <= 0.5) {

			// CREEP MUTATION OPERATOR
			// Subject each locus to random +/- 1 increment at the rate pmut
			for (i = 1; i <= n; ++i) {
				for (j = 1; j <= nd; ++j) {
					if (RAND.nextDouble() < pmut) {

						// Construct integer
						loc = (i - 1) * nd + j;
						inc = ((int) Math.round(RAND.nextDouble())) * 2 - 1;
						ist = (i - 1) * nd + 1;
						gn[loc - 1] += inc;

						// This is where we carry over the one (up to two digits)
						// first take care of decrement below 0 case
						if (inc < 0 && gn[loc - 1] < 0) {
							if (j == 1) {
								gn[loc - 1] = 0;
							} else {

								boolean skipto4 = false;
								for (k = loc; k >= ist + 1; --k) {
									gn[k - 1] = 9;
									--gn[k - 1 - 1];
									if (gn[k - 1 - 1] >= 0) {
										skipto4 = true;
										break;
									}
								}

								if (!skipto4) {

									// we popped under 0.00000 lower bound; fix it up
									if (gn[ist - 1] < 0) {
										for (l = ist; l <= loc; ++l) {
											gn[l - 1] = 0;
										}
									}
								}
							}
						}
						if (inc > 0 && gn[loc - 1] > 9) {
							if (j == 1) {
								gn[loc - 1] = 9;
							} else {

								boolean skipto7 = false;
								for (k = loc; k >= ist + 1; --k) {
									gn[k - 1] = 0;
									++gn[k - 1 - 1];
									if (gn[k - 1 - 1] <= 9) {
										skipto7 = true;
										break;
									}
								}

								if (!skipto7) {

									// we popped over 9.99999 upper bound; fix it up
									if (gn[ist - 1] > 9) {
										for (l = ist; l <= loc; ++l) {
											gn[l - 1] = 9;
										}
									}
								}
							}
						}
					}
				}
			}
		} else {

			// UNIFORM MUTATION OPERATOR
			// Subject each locus to random mutation at the rate pmut
			for (i = 1; i <= n * nd; ++i) {
				if (RAND.nextDouble() < pmut) {
					gn[i - 1] = (int) (RAND.nextDouble() * 10.0);
				}
			}
		}
	}

	private static void adjmut(final int ndim, final int n, final int np, final double[][] oldph, final double[] fitns,
			final int[] ifit, final double pmutmn, final double pmutmx, final double[] pmut, final int imut) {

		// Local:
		int i;
		double rdif = 0.0, rdiflo = 0.05, rdifhi = 0.25, delta = 1.5;

		if (imut == 2 || imut == 5) {

			// Adjustment based on fitness differential
			rdif = Math.abs(fitns[ifit[np - 1] - 1] - fitns[ifit[np / 2 - 1] - 1])
					/ (fitns[ifit[np - 1] - 1] + fitns[ifit[np / 2 - 1] - 1]);
		} else if (imut == 3 || imut == 6) {

			// Adjustment based on normalized metric distance
			rdif = 0.0;
			for (i = 1; i <= n; ++i) {
				rdif += RealMath.pow(oldph[ifit[np - 1] - 1][i - 1] - oldph[ifit[np / 2 - 1] - 1][i - 1], 2);
			}
			rdif = Math.sqrt(rdif) / n;
		}

		if (rdif <= rdiflo) {
			pmut[0] = Math.min(pmutmx, pmut[0] * delta);
		} else if (rdif >= rdifhi) {
			pmut[0] = Math.max(pmutmn, pmut[0] / delta);
		}
	}

	// c**********************************************************************
	// c REPRODUCTION MODULE
	// c**********************************************************************
	// c
	// c SELECT: Parent selection by roulette wheel algorithm
	// c called by: PIKAIA
	// c
	// c RNKPOP: Ranks initial population
	// c called by: PIKAIA, NEWPOP
	// c
	// c GENREP: Inserts offspring into population, for full
	// c generational replacement
	// c called by: PIKAIA
	// c
	// c STDREP: Inserts offspring into population, for steady-state
	// c reproduction
	// c called by: PIKAIA
	// c calls: FF
	// c
	// c NEWPOP: Replaces old generation with new generation
	// c called by: PIKAIA
	// c calls: FF, RNKPOP
	// c
	private static void select(final int np, final int[] jfit, final double fdif, final int[] idad) {

		// Local:
		int np1, i;
		double dice, rtfit;

		np1 = np + 1;
		dice = RAND.nextDouble() * np * np1;
		rtfit = 0;
		for (i = 1; i <= np; ++i) {
			rtfit += (np1 + fdif * (np1 - 2 * jfit[i - 1]));
			if (rtfit >= dice) {
				idad[0] = i;
				return;
			}
		}
		// Assert: loop will never exit by falling through
	}

	private static void rnkpop(final int n, final double[] arrin, final int[] indx, final int[] rank) {

		// Compute the key index
		rqsort(n, arrin, indx);

		// ...and the rank order
		for (int i = 1; i <= n; ++i) {
			rank[indx[i - 1] - 1] = n - i + 1;
		}
	}

	private static void genrep(final int ndim, final int n, final int np, final int ip, final double[][] ph,
			final double[][] newph) {

		// Insert one offspring pair into new population
		final int i1 = 2 * ip - 1;
		final int i2 = i1 + 1;
		System.arraycopy(ph[1 - 1], 0, newph[i1 - 1], 0, n);
		System.arraycopy(ph[2 - 1], 0, newph[i2 - 1], 0, n);
	}

	private static void stdrep(final FF ff, final int ndim, final int n, final int np, final int irep, final int ielite,
			final double[][] ph, final double[][] oldph, final double[] fitns, final int[] ifit, final int[] jfit,
			final int[] nnew, final int[] fev) {

		// Local:
		int i, j, k, i1, if1;
		double fit;

		nnew[0] = 0;
		for (j = 1; j <= 2; ++j) {

			// 1. compute offspring fitness (with caller's fitness function)
			fit = ff.ff(n, ph[j - 1]);
			++fev[0];

			// 2. if fit enough, insert in population
			for (i = np; i >= 1; --i) {
				if (fit > fitns[ifit[i - 1] - 1]) {

					// make sure the phenotype is not already in the population
					if (i < np) {
						boolean goto1 = true;
						for (k = 1; k <= n; ++k) {
							final double oldphi = oldph[ifit[i + 1 - 1] - 1][k - 1];
							if (oldphi != ph[j - 1][k - 1]) {
								goto1 = false;
								break;
							}
						}
						if (goto1) {
							break;
						}
					}

					// offspring is fit enough for insertion, and is unique
					// (i) insert phenotype at appropriate place in population
					if (irep == 3) {
						i1 = 1;
					} else if (ielite == 0 || i == np) {
						i1 = (int) (RAND.nextDouble() * np) + 1;
					} else {
						i1 = (int) (RAND.nextDouble() * (np - 1)) + 1;
					}
					if1 = ifit[i1 - 1];
					fitns[if1 - 1] = fit;
					System.arraycopy(ph[j - 1], 0, oldph[if1 - 1], 0, n);

					// (ii) shift and update ranking arrays
					if (i < i1) {

						// shift up
						jfit[if1 - 1] = np - i;
						for (k = i1 - 1; k >= i + 1; --k) {
							--jfit[ifit[k - 1] - 1];
							ifit[k + 1 - 1] = ifit[k - 1];
						}
						ifit[i + 1 - 1] = if1;
					} else {

						// shift down
						jfit[if1 - 1] = np - i + 1;
						for (k = i1 + 1; k <= i; ++k) {
							++jfit[ifit[k - 1] - 1];
							ifit[k - 1 - 1] = ifit[k - 1];
						}
						ifit[i - 1] = if1;
					}
					++nnew[0];
					break;
				}
			}
		}
	}

	private static void newpop(final FF ff, final int ielite, final int ndim, final int n, final int np,
			final double[][] oldph, final double[][] newph, final int[] ifit, final int[] jfit, final double[] fitns,
			final int[] nnew, final int[] fev) {
		nnew[0] = np;

		// if using elitism, introduce in new population fittest of old
		// population (if greater than fitness of the individual it is
		// to replace)
		if (ielite == 1 && ff.ff(n, newph[1 - 1]) < fitns[ifit[np - 1] - 1]) {
			System.arraycopy(oldph[ifit[np - 1] - 1], 0, newph[1 - 1], 0, n);
			--nnew[0];
		}
		++fev[0];

		// replace population
		for (int i = 1; i <= np; ++i) {
			System.arraycopy(newph[i - 1], 0, oldph[i - 1], 0, n);

			// get fitness using caller's fitness function
			fitns[i - 1] = ff.ff(n, oldph[i - 1]);
			++fev[0];
		}

		// compute new population fitness rank order
		rnkpop(np, fitns, ifit, jfit);
	}

	private static void rqsort(final int n, final double[] a, final int[] p) {

		final int lgn = 32, q = 11;
		final int[] stackl = new int[lgn], stackr = new int[lgn];
		double x;
		int s, t, l, m, r, i, j;

		// Initialize the stack
		stackl[1 - 1] = 1;
		stackr[1 - 1] = n;
		s = 1;

		// Initialize the pointer array
		for (i = 1; i <= n; ++i) {
			p[i - 1] = i;
		}

		while (s > 0) {
			l = stackl[s - 1];
			r = stackr[s - 1];
			--s;

			while (true) {
				if ((r - l) < q) {

					// Use straight insertion
					for (i = l + 1; i <= r; ++i) {
						t = p[i - 1];
						x = a[t - 1];
						boolean skipto5 = false;
						for (j = i - 1; j >= l; --j) {
							if (a[p[j - 1] - 1] <= x) {
								skipto5 = true;
								break;
							}
							p[j + 1 - 1] = p[j - 1];
						}
						if (!skipto5) {
							j = l - 1;
						}
						p[j + 1 - 1] = t;
					}
					break;
				} else {

					// Use quicksort, with pivot as median of a(l), a(m), a(r)
					m = (l + r) / 2;
					t = p[m - 1];
					if (a[t - 1] < a[p[l - 1] - 1]) {
						p[m - 1] = p[l - 1];
						p[l - 1] = t;
						t = p[m - 1];
					}
					if (a[t - 1] > a[p[r - 1] - 1]) {
						p[m - 1] = p[r - 1];
						p[r - 1] = t;
						t = p[m - 1];
						if (a[t - 1] < a[p[l - 1] - 1]) {
							p[m - 1] = p[l - 1];
							p[l - 1] = t;
							t = p[m - 1];
						}
					}

					// Partition
					x = a[t - 1];
					i = l + 1;
					j = r - 1;
					while (i <= j) {
						while (a[p[i - 1] - 1] < x) {
							++i;
						}
						while (x < a[p[j - 1] - 1]) {
							--j;
						}
						if (i <= j) {
							t = p[i - 1];
							p[i - 1] = p[j - 1];
							p[j - 1] = t;
							++i;
							--j;
						}
					}

					// Stack the larger subfile
					++s;
					if ((j - l) > (r - i)) {
						stackl[s - 1] = l;
						stackr[s - 1] = j;
						l = i;
					} else {
						stackl[s - 1] = i;
						stackr[s - 1] = r;
						r = j;
					}
				}
			}
		}
	}
}
