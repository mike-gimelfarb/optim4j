package min4j.testbeds;

import opt.multivariate.unconstrained.order0.cmaes.BiPopCmaesAlgorithm;

public class Example {

	public static void main(String[] args) {

		// function to optimize: here the Rastrigin function
		// note: the dimension here is determined by the size of the input
		java.util.function.Function<double[], Double> rastrigin = x -> {
			double y = 10 * x.length;
			for (final double e : x) {
				y += e * e - 10 * Math.cos(2 * Math.PI * e);
			}
			return y;
		};

		double[] initial = new double[5];

		BiPopCmaesAlgorithm optimizer = new BiPopCmaesAlgorithm(1e-10, 1e-5, 2.0, 1000000, 1000, true);
		double[] solution = optimizer.optimize(rastrigin, initial);

		System.out.println("solution x = " + java.util.Arrays.toString(solution));
		System.out.println("solution y = " + rastrigin.apply(solution));
		System.out.println("evaluations = " + optimizer.countEvaluations());
	}
}
