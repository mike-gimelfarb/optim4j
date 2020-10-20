package examples;

import java.util.function.Function;

import opt.univariate.UnivariateOptimizerSolution;
import opt.univariate.order1.CubicInterpolationAlgorithm;

public class WeibullFitting {

	public static void main(String[] args) {

		// data
		double[] data = { 509., 660., 386., 753., 811., 613., 848., 725., 315., 872., 487., 512. };
		int n = data.length;

		// log-likelihood function
		Function<Double, Double> f = k -> {
			double sum_pow_k = 0.0;
			double sum_log = 0.0;
			for (final double x : data) {
				sum_pow_k += Math.pow(x, k);
				sum_log += Math.log(x);
			}
			return -(n * Math.log(k) - n * Math.log(sum_pow_k / n) + (k - 1.0) * sum_log - 12.0);
		};

		// derivative of the log-likelihood function
		Function<Double, Double> df = k -> {
			double sum_pow_k = 0.0;
			double sum_log = 0.0;
			double sum_pow_k_log = 0.0;
			for (final double x : data) {
				sum_pow_k += Math.pow(x, k);
				sum_log += Math.log(x);
				sum_pow_k_log += Math.pow(x, k) * Math.log(x);
			}
			return -(n / k - n * sum_pow_k_log / sum_pow_k + sum_log);
		};

		CubicInterpolationAlgorithm optimizer = new CubicInterpolationAlgorithm(1e-8, 1e-14, 999);
		UnivariateOptimizerSolution solution = optimizer.optimize(f, df, 1.0);
		System.out.println(solution);

		// compute the lambda parameter
		double k = solution.getOptimalPoint();
		double sum_pow_k = 0.0;
		for (final double x : data) {
			sum_pow_k += Math.pow(x, k);
		}
		double lambda = Math.pow(sum_pow_k / n, 1.0 / k);

		// print the final parameters of the Weibull
		System.out.println("Final parameters are k=" + k + ", lambda=" + lambda);
	}
}
