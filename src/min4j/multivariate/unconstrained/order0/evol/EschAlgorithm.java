package min4j.multivariate.unconstrained.order0.evol;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.Function;

import min4j.multivariate.unconstrained.order0.GradientFreeOptimizer;
import min4j.utils.Constants;

// from the NLOpt library
public final class EschAlgorithm extends GradientFreeOptimizer {

	// ==========================================================================
	// STATIC CLASSES
	// ==========================================================================
	private static class Individual {

		public double[] parameters;
		public double fitness;
	}

	// ==========================================================================
	// FIELDS
	// ==========================================================================
	// problem parameters
	private Function<? super double[], Double> myFunc;
	private int myD;
	private double[] myLo, myHi;
	private double[] myGuess;
	private Comparator<Individual> myComparer;

	// algorithm parameters
	private final int myMaxEvals;
	private final int np, no;
	private double[] vetor;

	// algorithm memory
	private Individual[] esparents, esoffsprings, estotal;

	// ==========================================================================
	// CONSTRUCTORS
	// ==========================================================================
	/**
	 * 
	 * @param tolerance
	 * @param numParents
	 * @param numOffspring
	 */
	public EschAlgorithm(final int maxEvaluations, final int numParents, final int numOffspring) {
		super(1e-6);
		myMaxEvals = maxEvaluations;
		np = numParents;
		no = numOffspring;
	}

	// ==========================================================================
	// IMPLEMENTATIONS
	// ==========================================================================
	@Override
	public void initialize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		initialize(func, guess, lo, hi);
	}

	@Override
	public void iterate() {

		/**************************************
		 * Crossover
		 **************************************/
		for (int id = 0; id < no; ++id) {
			final int parent1 = RAND.nextInt(np);
			final int parent2 = RAND.nextInt(np);
			final int crosspoint = RAND.nextInt(myD);
			for (int item = 0; item < crosspoint; ++item) {
				esoffsprings[id].parameters[item] = esparents[parent1].parameters[item];
			}
			for (int item = crosspoint; item < myD; ++item) {
				esoffsprings[id].parameters[item] = esparents[parent2].parameters[item];
			}
		}

		/**************************************
		 * Gaussian Mutation
		 **************************************/
		int totalmutation = (int) (no * myD * 0.1);
		if (totalmutation < 1) {
			totalmutation = 1;
		}
		for (int contmutation = 0; contmutation < totalmutation; ++contmutation) {
			final int idoffmutation = RAND.nextInt(no);
			final int paramoffmutation = RAND.nextInt(myD);
			vetor[1] = myLo[paramoffmutation];
			vetor[2] = myHi[paramoffmutation];
			vetor[7] += contmutation;
			esoffsprings[idoffmutation].parameters[paramoffmutation] = randcauchy(vetor);
		}

		/**************************************
		 * Offsprings fitness evaluation
		 **************************************/
		for (int id = 0; id < no; ++id) {
			esoffsprings[id].fitness = myFunc.apply(esoffsprings[id].parameters);
			estotal[id + np].fitness = esoffsprings[id].fitness;
		}
		myEvals += no;

		/**************************************
		 * Individual selection
		 **************************************/
		// all the individuals are copied to one vector to easily identify best
		// solutions
		System.arraycopy(esparents, 0, estotal, 0, np);
		System.arraycopy(esoffsprings, 0, estotal, np, no);
		Arrays.sort(estotal, myComparer);

		// copy after sorting:
		System.arraycopy(estotal, 0, esparents, 0, np);
		System.arraycopy(estotal, np, esoffsprings, 0, no);
	}

	@Override
	public double[] optimize(final Function<? super double[], Double> func, final double[] guess) {
		final double[] lo = new double[guess.length];
		final double[] hi = new double[guess.length];
		for (int i = 0; i < guess.length; ++i) {
			lo[i] = guess[i] - 4.0;
			hi[i] = guess[i] + 4.0;
		}
		return optimize(func, guess, lo, hi);
	}

	// ==========================================================================
	// PUBLIC METHODS
	// ==========================================================================
	/**
	 * 
	 * @param f
	 * @param guess
	 * @param lb
	 * @param ub
	 */
	public void initialize(final Function<? super double[], Double> f, final double[] guess, final double[] lb,
			final double[] ub) {
		vetor = new double[8];

		// set parameters
		myFunc = f;
		myLo = lb;
		myHi = ub;
		myGuess = guess;
		myD = myGuess.length;
		myComparer = EschAlgorithm::compare;

		/*********************************
		 * controlling the population size
		 *********************************/
		esparents = new Individual[np];
		esoffsprings = new Individual[no];
		estotal = new Individual[np + no];
		for (int id = 0; id < np; ++id) {
			esparents[id] = new Individual();
			estotal[id] = new Individual();
		}
		for (int id = 0; id < no; ++id) {
			esoffsprings[id] = new Individual();
			estotal[np + id] = new Individual();
		}

		// From here the population is initialized
		// main vector of parameters to randcauchy
		vetor[0] = 4;
		vetor[3] = 0;
		vetor[4] = 1;
		vetor[5] = 10;
		vetor[6] = 1;
		vetor[7] = 0;

		/**************************************
		 * Initializing parents population
		 **************************************/
		for (int id = 0; id < np; ++id) {
			esparents[id].parameters = new double[myD];
			for (int item = 0; item < myD; ++item) {
				vetor[1] = lb[item];
				vetor[2] = ub[item];
				vetor[7] += 1;
				esparents[id].parameters[item] = randcauchy(vetor);
			}
		}
		System.arraycopy(myGuess, 0, esparents[0].parameters, 0, myD);

		/**************************************
		 * Initializing offsprings population
		 **************************************/
		for (int id = 0; id < no; ++id) {
			esoffsprings[id].parameters = new double[myD];
			for (int item = 0; item < myD; ++item) {
				vetor[1] = lb[item];
				vetor[2] = ub[item];
				vetor[7] += 1;
				esoffsprings[id].parameters[item] = randcauchy(vetor);
			}
		}

		/**************************************
		 * Parents fitness evaluation
		 **************************************/
		for (int id = 0; id < np; ++id) {
			esparents[id].fitness = f.apply(esparents[id].parameters);
			estotal[id].fitness = esparents[id].fitness;
		}
		myEvals = np;
	}

	/**
	 * 
	 * @param func
	 * @param guess
	 * @param lb
	 * @param ub
	 * @return
	 */
	public double[] optimize(final Function<? super double[], Double> func, final double[] guess, final double[] lb,
			final double[] ub) {
		initialize(func, guess, lb, ub);
		while (myEvals < myMaxEvals) {
			iterate();
		}
		return esparents[0].parameters;
	}

	// ==========================================================================
	// HELPER METHODS
	// ==========================================================================
	private static int compare(final Individual a, final Individual b) {
		return Double.compare(a.fitness, b.fitness);
	}

	private static double randcauchy(final double[] params) {
		double na_unif, cauchy_mit, limit_inf, limit_sup;
		double valor;
		double min = params[1];
		double max = params[2];
		double mi = params[3];
		double t = params[4];
		double band = params[5];
		limit_inf = mi - band * 0.5;
		limit_sup = mi + band * 0.5;
		do {
			na_unif = Math.random();
			cauchy_mit = t * Math.tan((na_unif - 0.5) * Constants.PI) + mi;
		} while (cauchy_mit < limit_inf || cauchy_mit > limit_sup);
		if (cauchy_mit < 0.0) {
			cauchy_mit = -cauchy_mit;
		} else {
			cauchy_mit += band * 0.5;
		}
		valor = cauchy_mit / band;
		valor = min + (max - min) * valor;
		return valor;
	}
}
