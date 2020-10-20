package examples;

import java.util.Arrays;
import java.util.function.Function;

import opt.multivariate.MultivariateOptimizerSolution;
import opt.multivariate.unconstrained.order0.evol.AdaptiveFireflyAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AdaptiveFireflyAlgorithm.*;
import opt.multivariate.unconstrained.order0.evol.AMLJayaAlgorithm;
import testbeds.MultiUnconstrStandard;
import utils.BlasMath;

public class MultivariateOptimizationExample {

	public static void main(String[] args) {

		Function<double[], Double> ackley = x -> {
			double sumcos = 0.0;
			for (final double e : x) {
				sumcos += Math.cos(2 * Math.PI * e) / x.length;
			}
			final double norm = BlasMath.denorm(x.length, x);
			return -20.0 * Math.exp(-0.2 * norm / Math.sqrt(x.length)) - Math.exp(sumcos) + 20.0 + Math.E;
		};

		Function<double[], Double> rosenbrock = x -> {
			double result = 0.0;
			for (int i = 0; i < x.length - 1; i++) {
				final double dxi1 = x[i + 1] - x[i] * x[i];
				final double dxi2 = x[i] - 1;
				result += 100 * dxi1 * dxi1 + dxi2 * dxi2;
			}
			return result;
		};

		final double[] lower = new double[20];
		final double[] upper = new double[20];
		Arrays.fill(lower, -10.0);
		Arrays.fill(upper, +10.0);
//		final AdaptiveFireflyAlgorithm alg = new AdaptiveFireflyAlgorithm(20, 0.1, 0.9, 0.5, new Geometric(0.2, 0.995),
//				2, 0.05, 100000);
//		final MultivariateOptimizerSolution result = alg.optimize(func, lower, upper);
//		System.out.println(func.apply(result.getOptimalPoint()));
//		System.out.println(result);
		AMLJayaAlgorithm alg = new AMLJayaAlgorithm(100, t -> 0.01, 1.5, 1, 200000, true, 10);
		alg.optimize(rosenbrock, lower, upper);
	}
}
