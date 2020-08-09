package examples;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.MultivariateOptimizerSolution;
import opt.multivariate.unconstrained.order0.evol.AdaptiveFireflyAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AdaptiveFireflyAlgorithm.*;
import testbeds.MultiUnconstrStandard;

public class MultivariateOptimizationExample {

	public static void main(String[] args) {
		Function<double[], Double> func = MultiUnconstrStandard.ALL_FUNCTIONS.get("rastrigin");
		final double[] lower = new double[30];
		final double[] upper = new double[30];
		Arrays.fill(lower, -5.0);
		Arrays.fill(upper, 6.0);
		final AdaptiveFireflyAlgorithm alg = new AdaptiveFireflyAlgorithm(20, 0.1, 0.9, 0.5, new Sh2014(0.2), 2, 0.05,
				100000);
		final MultivariateOptimizerSolution result = alg.optimize(func, lower, upper);
		System.out.println(result);
	}
}
