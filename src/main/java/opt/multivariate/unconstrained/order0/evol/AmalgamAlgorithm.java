/*
Please note, even though the main code for AMALGAM can be available under MIT 
licensed, the dchdcm subroutine is a derivative of LINPACK code that is licensed
under the 3-Clause BSD license. The other subroutines:


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

package opt.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.function.Function;

import opt.OptimizerSolution;
import opt.multivariate.GradientFreeOptimizer;
import utils.BlasMath;

/**
 * [1] Bosman, Peter AN, J�rn Grahl, and Dirk Thierens. "AMaLGaM IDEAs in
 * noiseless black-box optimization benchmarking." Proceedings of the 11th
 * Annual Conference Companion on Genetic and Evolutionary Computation
 * Conference: Late Breaking Papers. ACM, 2009.
 */
public final class AmalgamAlgorithm extends GradientFreeOptimizer {

	private static final class Solution implements Comparable<Solution> {

		double fx;
		double[] x;

		@Override
		public int compareTo(final Solution o) {
			return Double.compare(fx, o.fx);
		}
	}

	// problem parameters
	private Function<? super double[], Double> myFunc;
	private int myD;
	private double[] myLower, myUpper;

	// algorithm parameters
	private final int myMaxEvals;
	private final boolean myIamalgam;
	private final double myMinCMult = 1e-10;
	private final double myTau = 0.35;
	private final int myNElitist = 1;

	private double myAlphaAms, myDeltaAms, myEtaSigma;
	private double myEtaShift, myEtaDec, myEtaInc;
	private double myCMult, myThetaSdr;
	private int myNAms, myNis, myNisMax;
	private int myPopSize, mySelectSize, myT, myEvals;

	private Solution[] mySols;
	private double[] myMu, myMuOld, myMuShift, myMuShiftOld;
	private double[] myTemp, myXAvg;
	private double[][] myCov, myChol;

	// algorithm parameters for multiple runs
	private final boolean myParamFree, myPrintProgress;
	private final double myGlobalTolF;
	private int myS, myRuns, myNBase, myBudget;
	private double[] myBestX;
	private double myBestF, myBestFRun, myBestFRunOld;

	/**
	 * 
	 * @param toleranceSigmaF
	 * @param toleranceGlobalF
	 * @param maxEvaluations
	 * @param populationSize
	 * @param useIAmalgam
	 * @param useParameterFreeMode
	 * @param printProgress
	 */
	public AmalgamAlgorithm(final double toleranceSigmaF, final double toleranceGlobalF, final int maxEvaluations,
			final int populationSize, final boolean useIAmalgam, final boolean useParameterFreeMode,
			final boolean printProgress) {
		super(toleranceSigmaF);
		myGlobalTolF = toleranceGlobalF;
		myMaxEvals = maxEvaluations;
		myPopSize = populationSize;
		myIamalgam = useIAmalgam;
		myParamFree = useParameterFreeMode;
		myPrintProgress = printProgress;
	}

	/**
	 *
	 * @param toleranceSigmaF
	 * @param toleranceGlobalF
	 * @param maxEvaluations
	 * @param useIAmalgam
	 * @param printProgress
	 */
	public AmalgamAlgorithm(final double toleranceSigmaF, final double toleranceGlobalF, final int maxEvaluations,
			final boolean useIAmalgam, final boolean printProgress) {
		this(toleranceSigmaF, toleranceGlobalF, maxEvaluations, 0, useIAmalgam, true, printProgress);
	}

	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		initialize(func, lo, hi);

	}

	@Override
	public OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> func,
			final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, lo, hi);
	}

	@Override
	public void iterate() {

		// parameter-free version
		if (myParamFree) {

			// figure out the population size and the number of parallel runs
			final int floorS = myS >>> 1;
			if ((myS & 1) == 0) {
				myPopSize = (1 + floorS) * myNBase;
				myRuns = 1 << floorS;
			} else {
				myPopSize = (1 << (1 + floorS)) * myNBase;
				myRuns = 1;
			}

			// run algorithms amalgam in parallel
			runInParallel();

			// print
			if (myPrintProgress) {
				System.out.println(
						myS + "\t" + myRuns + "\t" + myPopSize + "\t" + myBestFRun + "\t" + myBestF + "\t" + myEvals);
			}

			// increment counters
			++myS;
			return;
		}

		// update mean and variance of the Gaussian
		updateDistribution(mySols, myTau, myPopSize, myD, mySelectSize, myMu, myMuOld, myCov, myEtaSigma, myMuShift,
				myMuShiftOld, myEtaShift, myChol, myCMult, myT, myTemp);

		// re-sample parameters
		final int ibest = samplePopulation(mySols, myPopSize, myD, myMu, myChol, myNAms, myDeltaAms, myCMult, myMuShift,
				myTemp, myFunc);
		myEvals += myPopSize;

		// update the rest of the parameters
		if (ibest > 0) {
			myNis = 0;
			if (myCMult < 1.0) {
				myCMult = 1.0;
			}
			final double SDR = computeSDR(mySols, myPopSize, myD, myXAvg, myChol, myMu, myTemp);
			if (SDR > myThetaSdr) {
				myCMult *= myEtaInc;
			}
		} else {
			if (myCMult <= 1.0) {
				++myNis;
			}
			if (myCMult > 1.0 || myNis >= myNisMax) {
				myCMult *= myEtaDec;
			}
			if (myCMult < 1.0 && myNis < myNisMax) {
				myCMult = 1.0;
			}
		}
		++myT;
	}

	/**
	 * 
	 * @param func
	 * @param lower
	 * @param upper
	 */
	public void initialize(final Function<? super double[], Double> func, final double[] lower, final double[] upper) {

		// prepare problem
		myFunc = func;
		myLower = lower;
		myUpper = upper;
		myD = lower.length;

		// parameter-free version
		if (myParamFree) {

			// compute the base population size
			if (myIamalgam) {
				myNBase = (int) (10.0 * Math.sqrt(myD));
			} else {
				myNBase = (int) (17.0 + 3.0 * Math.pow(myD, 1.5));
			}

			// for keeping track of the best global solution
			myBestX = null;
			myBestF = Double.POSITIVE_INFINITY;
			myBestFRun = Double.POSITIVE_INFINITY;
			myBestFRunOld = Double.POSITIVE_INFINITY;

			// initialize counters
			myS = 0;
			myBudget = myMaxEvals;
			myEvals = 0;

			// print headers
			if (myPrintProgress) {
				System.out.println(
						"s\t" + "runs\t" + "pop size\t" + "best f on run\t" + "best f so far\t" + "total evals");
			}
			return;
		}

		// initialize population size
		if (myPopSize <= 0) {
			if (myIamalgam) {
				myPopSize = (int) (10.0 * Math.sqrt(myD));
			} else {
				myPopSize = (int) (17.0 + 3.0 * Math.pow(myD, 1.5));
			}
		}

		// prepare parameters to run i-amalgam algorithm
		myEvals = 0;
		myT = 0;
		mySelectSize = (int) Math.floor(myTau * (myPopSize));
		if (myIamalgam) {
			final double expSigma = -1.1 * Math.pow(mySelectSize, 1.2) / Math.pow(myD, 1.6);
			final double expShift = -1.2 * Math.pow(mySelectSize, 0.31) / Math.sqrt(myD);
			myEtaSigma = 1.0 - Math.exp(expSigma);
			myEtaShift = 1.0 - Math.exp(expShift);
		} else {
			myEtaSigma = 1.0;
			myEtaShift = 1.0;
		}

		// anticipated mean shift
		myAlphaAms = 0.5 * myTau * myPopSize / (myPopSize - myNElitist);
		myNAms = (int) (myAlphaAms * (myPopSize - 1));
		myDeltaAms = 2.0;

		// distribution multipliers
		myNis = 0;
		myNisMax = 25 + myD;
		myEtaDec = 0.9;
		myEtaInc = 1.0 / myEtaDec;
		myThetaSdr = 1.0;
		myCMult = 1.0;

		// initialize the population
		mySols = new Solution[myPopSize];
		for (int m = 0; m < myPopSize; ++m) {
			mySols[m] = new Solution();
			mySols[m].x = new double[myD];
			for (int i = 0; i < myD; ++i) {
				mySols[m].x[i] = myLower[i] + (myUpper[i] - myLower[i]) * RAND.nextDouble();
			}
			mySols[m].fx = myFunc.apply(mySols[m].x);
		}
		myEvals += myPopSize;
		Arrays.sort(mySols);

		// initialize the other arrays
		myMu = new double[myD];
		myMuOld = new double[myD];
		myMuShift = new double[myD];
		myMuShiftOld = new double[myD];
		myTemp = new double[myD];
		myXAvg = new double[myD];
		myCov = new double[myD][myD];
		myChol = new double[myD][myD];

		// estimate mean
		for (int m = 0; m < mySelectSize; ++m) {
			BlasMath.dxpym(myD, mySols[m].x, 1, myMu, 1);
		}
		BlasMath.dscalm(myD, 1.0 / mySelectSize, myMu, 1);

		// estimate covariance
		for (int i = 0; i < myD; ++i) {
			for (int m = 0; m < mySelectSize; ++m) {
				final double xmmu = mySols[m].x[i] - myMu[i];
				myCov[i][i] += xmmu * xmmu;
				myCov[i][i] /= mySelectSize;
			}
		}
	}

	/**
	 * 
	 * @param func
	 * @param lower
	 * @param upper
	 * @return
	 */
	public OptimizerSolution<double[], Double> optimize(final Function<? super double[], Double> func,
			final double[] lower, final double[] upper) {
		initialize(func, lower, upper);
		while (true) {
			iterate();
			if (isConverged()) {
				final double[] sol;
				if (myParamFree) {
					sol = myBestX;
				} else {
					sol = mySols[0].x;
				}
				return new OptimizerSolution<>(sol, myEvals, 0, myEvals < myMaxEvals);
			}
		}
	}

	private void runInParallel() {

		// record best values on this run
		myBestFRunOld = myBestFRun;
		myBestFRun = Double.POSITIVE_INFINITY;

		for (int r = 1; r <= myRuns; ++r) {

			// initialize the optimizer
			final AmalgamAlgorithm algr = new AmalgamAlgorithm(myTol, 0, myBudget, myPopSize, myIamalgam, false, false);

			// perform the optimization
			final OptimizerSolution<double[], Double> sol = algr.optimize(myFunc, myLower, myUpper);
			myEvals += sol.getFEvals();
			myBudget -= sol.getFEvals();
			final double[] optr = sol.getOptimalPoint();

			// get the best fitness
			final double fitr = myFunc.apply(optr);
			++myEvals;
			--myBudget;

			// update local best solution found on this run
			myBestFRun = Math.min(myBestFRun, fitr);

			// update global best solution found
			if (fitr < myBestF) {
				myBestF = fitr;
				myBestX = optr;
			}
		}
	}

	private boolean isConverged() {

		// check number of evaluations
		if (myEvals >= myMaxEvals) {
			return true;
		}

		// parameter-free version
		if (myParamFree) {

			// check convergence in objective tolerance
			// TODO: check this condition is satisfactory - if not find a better
			// convergence criterion for multiple restarts
			if (myBestFRun != myBestFRunOld && Math.abs(myBestFRun - myBestFRunOld) <= myGlobalTolF) {
				return true;
			}
			return false;
		}

		// check minimum multiplier value
		if (myCMult < myMinCMult) {
			return true;
		}

		// compute variance of the population fitness values
		double fmean = 0.0;
		for (final Solution soln : mySols) {
			fmean += soln.fx / myPopSize;
		}
		double fvar = 0.0;
		for (final Solution soln : mySols) {
			fvar += (soln.fx - fmean) * (soln.fx - fmean) / myPopSize;
		}
		if (fvar <= myTol * myTol) {
			return true;
		}
		return false;
	}

	private static double computeSDR(final Solution[] sols, final int n, final int d, final double[] xavg,
			final double[][] cholf, final double[] mu, final double[] temp) {

		// compute the average of all points better than previous best
		Arrays.fill(xavg, 0.0);
		int count = 0;
		for (int m = 1; m < n; ++m) {
			if (sols[m].fx < sols[0].fx) {
				BlasMath.dxpym(d, sols[m].x, 1, xavg, 1);
				++count;
			}
		}
		BlasMath.dscalm(d, 1.0 / count, xavg, 1);

		// subtract mean
		BlasMath.daxpym(d, -1.0, mu, 1, xavg, 1);

		// compute SDR
		double sdr = 0.0;
		for (int i = 0; i < d; ++i) {
			temp[i] = xavg[i];
			temp[i] -= BlasMath.ddotm(i, cholf[i], 1, temp, 1);
			temp[i] /= cholf[i][i];
			sdr = Math.max(sdr, Math.abs(temp[i]));
		}
		return sdr;
	}

	private static void updateDistribution(final Solution[] sols, final double tau, final int n, final int d,
			final int taun, final double[] mu, final double[] muold, final double[][] cov, final double nsigma,
			final double[] mushift, final double[] mushiftold, final double nshift, final double[][] cholf,
			final double cmult, final int t, final double[] temp) {

		// find the best solutions
		Arrays.sort(sols);

		// save current mu
		System.arraycopy(mu, 0, muold, 0, d);

		// compute the mean of the nbest solutions
		Arrays.fill(mu, 0.0);
		for (int m = 0; m < taun; ++m) {
			BlasMath.daxpym(d, 1.0 / taun, sols[m].x, 1, mu, 1);
		}

		// update covariance matrix
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j <= i; ++j) {
				cov[i][j] *= (1.0 - nsigma);
				for (int m = 0; m < taun; ++m) {
					cov[i][j] += (nsigma / taun) * (sols[m].x[i] - mu[i]) * (sols[m].x[j] - mu[j]);
				}
				cov[j][i] = cov[i][j];
			}
		}

		// save old mu shift
		System.arraycopy(mushift, 0, mushiftold, 0, d);

		// compute new shifted mu
		if (t == 1) {
			for (int i = 0; i < d; ++i) {
				mushift[i] = mu[i] - muold[i];
			}
		} else if (t > 1) {
			for (int i = 0; i < d; ++i) {
				mushift[i] *= (1.0 - nshift);
				mushift[i] += nshift * (mu[i] - muold[i]);
			}
		} else {
			Arrays.fill(mushift, 0.0);
		}

		// computes the cholesky factor of the covariance matrix
		for (int i = 0; i < d; ++i) {
			System.arraycopy(cov[i], 0, cholf[i], 0, d);
		}
		dchdcm(cholf, d, d, temp, new int[1]);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < i; ++j) {
				cholf[i][j] = cholf[j][i];
				cholf[j][i] = 0.0;
			}
		}
		final double cmultsqrt = Math.sqrt(cmult);
		for (int i = 0; i < d; ++i) {
			BlasMath.dscalm(d, cmultsqrt, cholf[i], 1);
		}
	}

	private static int samplePopulation(final Solution[] sols, final int n, final int d, final double[] mu,
			final double[][] cholf, final int nams, final double delams, final double cmult, final double[] mushift,
			final double[] temp, final Function<? super double[], Double> func) {

		// sample from the estimated normal distribution
		for (final Solution sol : sols) {
			for (int i = 0; i < d; ++i) {
				temp[i] = RAND.nextGaussian();
			}
			for (int i = 0; i < d; ++i) {
				sol.x[i] = mu[i];
				sol.x[i] += BlasMath.ddotm(d, cholf[i], 1, temp, 1);
			}
		}

		// perturb n_ams random solutions
		// shift the solutions by a multiple of mu_shift
		shuffle(1, n - 1, sols);
		for (int m = 1; m <= nams; ++m) {
			BlasMath.daxpym(d, delams * cmult, mushift, 1, sols[m].x, 1);
		}

		// perform the fitness evaluation
		// find an element that has a better fitness than the best
		int ibest = 0;
		for (int m = 1; m < n; ++m) {
			sols[m].fx = func.apply(sols[m].x);
			if (sols[m].fx < sols[0].fx) {
				ibest = m;
			}
		}
		return ibest;
	}

	// adapted from the LINPACK package
	// j.j. dongarra and g.w. stewart, argonne national laboratory and
	// university of maryland.
	private static void dchdcm(final double[][] a, final int lda, final int p, final double[] work, final int[] info) {

		// internal variables
		int pu, pl, ii, j, k, km1, kp1, l, maxl;
		double temp, maxdia;

		pl = 1;
		pu = 0;
		info[0] = p;

		for (k = 1; k <= p; ++k) {

			// reduction loop
			maxdia = a[k - 1][k - 1];
			kp1 = k + 1;
			maxl = k;

			// determine the pivot element
			if (k >= pl && k < pu) {
				for (l = kp1; l <= pu; ++l) {
					if (a[l - 1][l - 1] > maxdia) {
						maxdia = a[l - 1][l - 1];
						maxl = l;
					}
				}
			}

			// quit if the pivot element is not positive
			if (maxdia <= 0.0) {
				info[0] = k - 1;
				break;
			}

			// start the pivoting and update jpvt
			if (k != maxl) {
				km1 = k - 1;
				for (ii = 1; ii <= km1; ++ii) {
					temp = a[ii - 1][k - 1];
					a[ii - 1][k - 1] = a[ii - 1][maxl - 1];
					a[ii - 1][maxl - 1] = temp;
				}
				a[maxl - 1][maxl - 1] = a[k - 1][k - 1];
				a[k - 1][k - 1] = maxdia;
			}

			// reduction step. pivoting is contained across the rows
			work[k - 1] = Math.sqrt(a[k - 1][k - 1]);
			a[k - 1][k - 1] = work[k - 1];
			if (p >= kp1) {
				for (j = kp1; j <= p; ++j) {
					if (k != maxl) {
						if (j >= maxl) {
							if (j != maxl) {
								temp = a[k - 1][j - 1];
								a[k - 1][j - 1] = a[maxl - 1][j - 1];
								a[maxl - 1][j - 1] = temp;
							}
						} else {
							temp = a[k - 1][j - 1];
							a[k - 1][j - 1] = a[j - 1][maxl - 1];
							a[j - 1][maxl - 1] = temp;
						}
					}
					a[k - 1][j - 1] /= work[k - 1];
					work[j - 1] = a[k - 1][j - 1];
					temp = -a[k - 1][j - 1];
					for (ii = 1; ii <= j - k; ++ii) {
						a[kp1 - 1 + ii - 1][j - 1] += temp * work[kp1 - 1 + ii - 1];
					}
				}
			}
		}
	}

	@SafeVarargs
	private static final <T> void shuffle(final int i1, final int i2, final T... arr) {
		for (int i = i2; i > i1; --i) {
			final int index = RAND.nextInt(i - i1 + 1) + i1;
			swap(arr, index, i);
		}
	}

	private static final <T> void swap(final T[] data, final int i, final int j) {
		if (i == j) {
			return;
		}
		final T temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}
}
