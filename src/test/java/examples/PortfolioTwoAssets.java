package examples;

import java.util.function.Function;

import opt.univariate.UnivariateOptimizerSolution;
import opt.univariate.order0.BrentAlgorithm;

public class PortfolioTwoAssets {

	public static void main(String[] args) {

		// stock return data and parameters
		double r_f = 0.01;
		double mean1 = 0.10, std1 = 0.10;
		double mean2 = 0.17, std2 = 0.25;

		// negative Sharpe ratio objective
		Function<Double, Double> sharpe_objective = w -> {
			double mean = w * mean1 + (1.0 - w) * mean2;
			double var = Math.pow(w * std1, 2) + Math.pow((1.0 - w) * std2, 2);
			return -(mean - r_f) / Math.sqrt(var);
		};

		BrentAlgorithm optimizer = new BrentAlgorithm(1e-8, 1e-14, 999);
		UnivariateOptimizerSolution solution = optimizer.optimize(sharpe_objective, 0.0, 1.0);

		System.out.println(solution);
		System.out.println("Final sharpe ratio = " + -sharpe_objective.apply(solution.getOptimalPoint()));
	}
}
