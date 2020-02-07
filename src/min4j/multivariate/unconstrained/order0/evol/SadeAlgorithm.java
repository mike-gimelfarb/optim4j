package min4j.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.function.Function;

import min4j.multivariate.unconstrained.order0.GradientFreeOptimizer;
import min4j.utils.BlasMath;
import min4j.utils.Constants;

/**
 * Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential
 * evolution algorithm with strategy adaptation for global numerical
 * optimization." IEEE transactions on Evolutionary Computation 13.2 (2009):
 * 398-417.
 * 
 * Huang, Vicky Ling, A. Kai Qin, and Ponnuthurai N. Suganthan. "Self-adaptive
 * differential evolution algorithm for constrained real-parameter
 * optimization." Evolutionary Computation, 2006. CEC 2006. IEEE Congress on.
 * IEEE, 2006. Yang, Zhenyu, Ke Tang, and Xin Yao.
 * 
 * "Self-adaptive differential evolution with neighborhood search." Evolutionary
 * Computation, 2008. CEC 2008.(IEEE World Congress on Computational
 * Intelligence). IEEE Congress on. IEEE, 2008.
 * 
 * @author Michael
 */
public final class SadeAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	private final int myNp; // population size, >= 4
	private final int myLp; // learning period
	private final int myCp; // crossover refresh period
	private final int myK = 5; // number of strategies [1,# available]
	private final int myMaxEvals;
	private final double mySigmaTol; // tolerance in standard dev. of swarm
	private final double mySigmaCr = 0.1;
	private final double mySigmaF = 0.3;
	private final double myMu = 0.5;

	private Function<? super double[], Double> myFunc;
	private double CRm, Fp;
	private double[] p, y, CR, CRrec, dfit, lower, upper, xtrii;
	private double[][] pool;
	private int genr, ihist, Fns0, Fnf0, Fns1, Fnf1;
	private int[] ns, nf, ibw;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param populationSize
	 * @param learningPeriod
	 * @param crossoverRefreshPeriod
	 * @param maxEvaluations
	 */
	public SadeAlgorithm(final double tolerance, final double stdevTolerance, final int populationSize,
			final int learningPeriod, final int crossoverRefreshPeriod, final int maxEvaluations) {
		super(tolerance);
		mySigmaTol = stdevTolerance;
		myNp = populationSize;
		myLp = learningPeriod;
		myCp = crossoverRefreshPeriod;
		myMaxEvals = maxEvaluations;
	}

	/**
	 *
	 * @param tolerance
	 * @param stdevTolerance
	 * @param populationSize
	 * @param maxEvaluations
	 */
	public SadeAlgorithm(final double tolerance, final double stdevTolerance, final int populationSize,
			final int maxEvaluations) {
		this(tolerance, stdevTolerance, populationSize, 25, 5, maxEvaluations);
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public final void initialize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		initialize(func, lo, hi);
	}

	@Override
	public double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, lo, hi);
	}

	@Override
	public final void iterate() {
		final int d = lower.length;

		// learning update
		if (genr >= myLp && genr % myLp == 0) {

			// update strategy probabilities
			boolean updateP = true;
			for (int k = 0; k < myK; ++k) {
				if (ns[k] == 0) {
					updateP = false;
					break;
				}
			}
			for (int k = 0; k < myK; ++k) {
				if (updateP) {
					p[k] = (double) ns[k] / (ns[k] + nf[k]);
				}
				ns[k] = nf[k] = 0;
			}

			// update crossover CRm parameter
			double denom = 0.0;
			double numer = 0.0;
			for (int i = 0; i < ihist; ++i) {
				numer += CRrec[i] * dfit[i];
				denom += dfit[i];
				dfit[i] = CRrec[i] = 0.0;
			}
			if (denom > 0.0) {
				CRm = numer / denom;
			}
			ihist = 0;

			// update local search F parameter Fp
			denom = Fns1 * (Fns0 + Fnf0) + Fns0 * (Fns1 + Fnf1);
			if (denom > 0.0) {
				Fp = Fns0 * (Fns1 + Fnf1) / denom;
			}
			Fns0 = Fnf0 = Fns1 = Fnf1 = 0;
		}

		// update population
		for (int i = 0; i < myNp; ++i) {

			// compute crossover constant F
			final double u = RAND.nextDouble();
			final boolean usegauss = u < Fp;
			double F = 0.0;
			while (F <= 0.0) {
				if (usegauss) {
					F = RAND.nextGaussian() * mySigmaF + myMu;
				} else {
					F = Math.tan(Constants.PI * (RAND.nextDouble() - 0.5));
				}
			}

			// compute CR constant if needed
			if (genr > 0 && genr % myCp == 0) {
				double CRi = RAND.nextGaussian() * mySigmaCr + CRm;
				while (CRi <= 0.0 || CRi >= 1.0) {
					CRi = RAND.nextGaussian() * mySigmaCr + CRm;
				}
				CR[i] = CRi;
			}

			// generate a strategy to use for the current member
			final int ki = rouletteSample(p);

			// trial vector generation and fitness
			trial(d, i, ki, F, CR[i], ibw[0], xtrii);
			final double newy = myFunc.apply(xtrii);

			// update all counters and data for learning
			if (newy < y[i]) {
				CRrec[ihist] = CR[i];
				dfit[ihist] = y[i] - newy;
				++ihist;
				++ns[ki];
				if (usegauss) {
					++Fns0;
				} else {
					++Fns1;
				}

				// update population if the new vector is improvement
				System.arraycopy(xtrii, 0, pool[i], 0, d);
				y[i] = newy;
			} else {
				++nf[ki];
				if (usegauss) {
					++Fnf0;
				} else {
					++Fnf1;
				}
			}
		}

		// update indices
		myEvals += myNp;
		++genr;

		// find the new ranking
		rank();
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 *
	 * @param func
	 * @param lb
	 * @param ub
	 */
	public final void initialize(final Function<? super double[], Double> func, final double[] lb, final double[] ub) {
		myFunc = func;
		lower = lb;
		upper = ub;
		genr = myEvals = ihist = 0;

		// initialize members
		final int D = lower.length;
		pool = new double[myNp][D];
		y = new double[myNp];
		xtrii = new double[D];
		for (int i = 0; i < myNp; ++i) {
			final double[] xi = pool[i];
			for (int j = 0; j < D; ++j) {
				xi[j] = (upper[j] - lower[j]) * RAND.nextDouble() + lower[j];
			}
			pool[i] = xi;
			y[i] = myFunc.apply(xi);
		}
		myEvals += myNp;

		// compute the rankings
		ibw = new int[4];
		rank();

		// initialize learning parameters
		CRm = 0.5;
		ns = new int[myK];
		nf = new int[myK];
		Fns0 = Fnf0 = Fns1 = Fnf1 = 0;
		p = new double[myK];
		Fp = 0.5;
		Arrays.fill(p, 1.0 / myK);
		CR = new double[myNp];
		CRrec = new double[myNp * myLp];
		dfit = new double[myNp * myLp];
	}

	/**
	 *
	 * @param function
	 * @param lb
	 * @param ub
	 * @return
	 */
	public final double[] optimize(final Function<? super double[], Double> function, final double[] lb,
			final double[] ub) {

		// initialize parameters
		initialize(function, lb, ub);

		// main loop of SADE
		while (myEvals < myMaxEvals) {

			// learning and solution update
			iterate();

			// test convergence in function values
			final double y0 = y[ibw[0]];
			final double y3 = y[ibw[3]];
			final double toly = 0.5 * RELEPS * Math.abs(y0 + y3);
			if (Math.abs(y0 - y3) <= myTol + toly) {

				// compute standard deviation of swarm radiuses
				final int D = lb.length;
				int count = 0;
				double mean = 0.0;
				double m2 = 0.0;
				for (final double[] pt : pool) {
					final double x = BlasMath.denorm(D, pt);
					++count;
					final double delta = x - mean;
					mean += delta / count;
					final double delta2 = x - mean;
					m2 += delta * delta2;
				}

				// test convergence in standard deviation
				if (m2 <= (myNp - 1) * mySigmaTol * mySigmaTol) {
					break;
				}
			}
		}
		return pool[ibw[0]];
	}

	/**
	 *
	 * @return
	 */
	public final double[][] pool() {
		return pool;
	}

	/**
	 *
	 * @return
	 */
	public final double[] fitnesses() {
		return y;
	}

	/**
	 *
	 * @return
	 */
	public final int bestIndex() {
		return ibw[0];
	}

	/**
	 *
	 * @return
	 */
	public final int worstIndex() {
		return ibw[3];
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private void trial(final int D, final int i, final int ki, final double F, final double CR, final int ib,
			final double[] out) {

		// randomly select five distinct agents from population
		int a, b, c, d, e;
		do {
			a = RAND.nextInt(myNp);
		} while (a == i);
		do {
			b = RAND.nextInt(myNp);
		} while (b == i || b == a);
		do {
			c = RAND.nextInt(myNp);
		} while (c == i || c == a || c == b);
		do {
			d = RAND.nextInt(myNp);
		} while (d == i || d == a || d == b || d == c);
		do {
			e = RAND.nextInt(myNp);
		} while (e == i || e == a || e == b || e == c || e == d);

		// retrieve their data
		final double[] p1 = pool[a];
		final double[] p2 = pool[b];
		final double[] p3 = pool[c];
		final double[] p4 = pool[d];
		final double[] p5 = pool[e];
		final double[] x = pool[i];
		final double[] bb = pool[ib];

		// use them to generate a mutated vector from the original
		final int jrnd = RAND.nextInt(D);
		switch (ki) {
		case 0:

			// DE/rand/1
			for (int j = 0; j < D; ++j) {
				if (j == jrnd || RAND.nextDouble() <= CR) {
					out[j] = p1[j] + F * (p2[j] - p3[j]);
				} else {
					out[j] = x[j];
				}
			}
			break;
		case 1:

			// DE/best/1
			for (int j = 0; j < D; ++j) {
				if (j == jrnd || RAND.nextDouble() <= CR) {
					out[j] = bb[j] + F * (p1[j] - p2[j]);
				} else {
					out[j] = x[j];
				}
			}
			break;
		case 2:

			// DE/current-to-best/1
			for (int j = 0; j < D; ++j) {
				if (j == jrnd || RAND.nextDouble() <= CR) {
					out[j] = x[j] + F * (bb[j] - x[j]) + F * (p1[j] - p2[j]);
				} else {
					out[j] = x[j];
				}
			}
			break;
		case 3:

			// DE/best/2:
			for (int j = 0; j < D; ++j) {
				if (j == jrnd || RAND.nextDouble() <= CR) {
					out[j] = bb[j] + F * (p1[j] - p2[j]) + F * (p3[j] - p4[j]);
				} else {
					out[j] = x[j];
				}
			}
			break;
		case 4:

			// DE/rand/2:
			for (int j = 0; j < D; ++j) {
				if (j == jrnd || RAND.nextDouble() <= CR) {
					out[j] = p1[j] + F * (p2[j] - p3[j]) + F * (p4[j] - p5[j]);
				} else {
					out[j] = x[j];
				}
			}
			break;
		default:
			throw new IllegalArgumentException("Invalid mutation strategy.");
		}

		// ensure that the new trial vector lies in [lower, upper]
		// if not, then randomize it within this range
		for (int j = 0; j < D; ++j) {
			if (out[j] > upper[j] || out[j] < lower[j]) {
				out[j] = RAND.nextDouble() * (upper[j] - lower[j]) + lower[j];
			}
		}
	}

	private void rank() {
		double y0 = Double.POSITIVE_INFINITY;
		double y1 = y0;
		double y2 = Double.NEGATIVE_INFINITY;
		double y3 = y2;
		ibw[0] = ibw[1] = ibw[2] = ibw[3] = 0;
		for (int i = 0; i < y.length; ++i) {
			if (y[i] > y3) {
				y2 = y3;
				y3 = y[i];
				ibw[2] = ibw[3];
				ibw[3] = i;
			} else if (y[i] > y2) {
				y2 = y[i];
				ibw[2] = i;
			}
			if (y[i] < y0) {
				y1 = y0;
				y0 = y[i];
				ibw[1] = ibw[0];
				ibw[0] = i;
			} else if (y[i] < y1) {
				y1 = y[i];
				ibw[1] = i;
			}
		}
	}

	private static final int rouletteSample(final double... weights) {
		final int n = weights.length;
		double s = 0.0;
		for (int i = 0; i < n; ++i) {
			s += weights[i];
		}
		double U = RAND.nextDouble() * s;
		for (int k = 0; k < n; ++k) {
			U -= weights[k];
			if (U <= 0.0) {
				return k;
			}
		}
		return n - 1;
	}
}
