package examples;

import java.util.function.Function;

import opt.univariate.UnivariateOptimizerSolution;
import opt.univariate.order0.PiyavskiiAlgorithm;

public class Rastrigin1d {

	public static void main(String[] args) {

		// 1D Rastrigin function
		Function<Double, Double> rastrigin = x -> {
			return 10.0 + x * x - 10.0 * Math.cos(2.0 * Math.PI * x);
		};

		PiyavskiiAlgorithm optimizer = new PiyavskiiAlgorithm(1e-4, 999);
		UnivariateOptimizerSolution solution = optimizer.optimize(rastrigin, -4.0, 5.12);

		System.out.println(solution);
		System.out.println("Final f(x*) = " + rastrigin.apply(solution.getOptimalPoint()));
	}
}
