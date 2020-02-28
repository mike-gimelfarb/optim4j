package optim4j.testbeds;

import opt.OptimizerSolution;
import opt.multivariate.unconstrained.order1.*;

public class Example2 {

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

		java.util.function.Function<double[], double[]> drosenbrock = x -> {
			final double[] y = new double[x.length];
			y[0] = -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]);
			for (int i = 1; i < x.length - 1; ++i) {
				y[i] = 200 * (x[i] - x[i - 1] * x[i - 1]) - 400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]);
			}
			y[x.length - 1] = 200 * (x[x.length - 1] - Math.pow(x[x.length - 2], 2));
			return y;
		};

		double[] initial = new double[500];
		for (int i = 0; i < initial.length; ++i) {
			initial[i] = Math.random() * 50 - 25;
		}

		LBFGSBAlgorithm optimizer = new LBFGSBAlgorithm(1e-6, 10);
		OptimizerSolution<double[], Double> solution = optimizer.optimize(rosenbrock, drosenbrock, initial);

		System.out.println("solution x = " + java.util.Arrays.toString(solution.getOptimalPoint()));
		System.out.println("solution y = " + rosenbrock.apply(solution.getOptimalPoint()));
		System.out.println("evaluations = " + solution.getFEvals());
		System.out.println("converged = " + solution.converged());
	}
}
