package optim4j.testbeds;

import opt.OptimizerSolution;
import opt.multivariate.unconstrained.order0.cmaes.BiPopCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AdaptivePsoAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AmalgamAlgorithm;
import opt.multivariate.unconstrained.order0.evol.CcPsoAlgorithm;

public class Example {

	public static void main(String[] args) {

		// function to optimize: here the Rastrigin function
		// note: the dimension here is determined by the size of the input
		java.util.function.Function<double[], Double> rosenbrock = x -> {
			double y = 0.0;
			for (int i = 0; i < x.length - 1; i++) {
				y += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(x[i] - 1, 2);
			}
			return y;
		};
		
		

		double[] initial = new double[50];
		for (int i = 0; i < initial.length; ++i) {
			initial[i] = Math.random();
		}

		CcPsoAlgorithm optimizer = new CcPsoAlgorithm(1e-6, 1e-6, 5000000, 10, new int[] { 2, 5, 10, 25, 50 });
		OptimizerSolution<double[], Double> solution = optimizer.optimize(rosenbrock, initial);

		System.out.println("solution x = " + java.util.Arrays.toString(solution.getOptimalPoint()));
		System.out.println("solution y = " + rosenbrock.apply(solution.getOptimalPoint()));
		System.out.println("evaluations = " + solution.getFEvals());
		System.out.println("converged = " + solution.converged());
	}
}
