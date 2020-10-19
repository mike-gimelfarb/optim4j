package examples;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.MultivariateOptimizerSolution;
import opt.multivariate.unconstrained.order0.evol.AdaptiveFireflyAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AdaptiveFireflyAlgorithm.*;
import opt.multivariate.unconstrained.order0.evol.JayaAlgorithm;
import testbeds.MultiUnconstrStandard;

public class MultivariateOptimizationExample {

	public static void main(String[] args) {
		Function<double[], Double> func = MultiUnconstrStandard.ALL_FUNCTIONS.get("rastrigin");
		final double[] lower = new double[10];
		final double[] upper = new double[10];
		Arrays.fill(lower, -5.0);
		Arrays.fill(upper, 4.0);
//		final AdaptiveFireflyAlgorithm alg = new AdaptiveFireflyAlgorithm(20, 0.1, 0.9, 0.5, new Geometric(0.2, 0.995),
//				2, 0.05, 100000);
//		final MultivariateOptimizerSolution result = alg.optimize(func, lower, upper);
//		System.out.println(func.apply(result.getOptimalPoint()));
//		System.out.println(result);
		JayaAlgorithm alg = new JayaAlgorithm(50, 0.01, t -> 0.5, 2, 200000);
		alg.optimize(func, lower, upper);

	}
}
