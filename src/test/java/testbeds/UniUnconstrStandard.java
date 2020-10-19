package testbeds;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

import opt.OptimizerSolution;
import opt.univariate.order0.*;

public final class UniUnconstrStandard {

	public static final Map<String, Function<Double, Double>> ALL_FUNCTIONS = new LinkedHashMap<String, Function<Double, Double>>() {

		private static final long serialVersionUID = -3852837778398908234L;

		{
			put("p02", UniUnconstrStandard::p02);
			put("p03", UniUnconstrStandard::p03);
			put("p04", UniUnconstrStandard::p04);
			put("p05", UniUnconstrStandard::p05);
			put("p06", UniUnconstrStandard::p06);
			put("p07", UniUnconstrStandard::p07);
			put("p08", UniUnconstrStandard::p08);
			put("p09", UniUnconstrStandard::p09);
			put("p10", UniUnconstrStandard::p10);
			put("p11", UniUnconstrStandard::p11);
			put("p12", UniUnconstrStandard::p12);
			put("p13", UniUnconstrStandard::p13);
			put("p14", UniUnconstrStandard::p14);
			put("p15", UniUnconstrStandard::p15);
			put("p18", UniUnconstrStandard::p18);
			put("p20", UniUnconstrStandard::p20);
			put("p21", UniUnconstrStandard::p21);
			put("p22", UniUnconstrStandard::p22);
		}
	};

	public static final double taff01(final double oa, final double ob, final double x) {
		return oa + (ob - oa) * x;
	}

	public static final double p02(double x) {
		x = taff01(2.7, 7.5, x);
		return Math.sin(x) + Math.sin(10 * x / 3) + 1.899599;
	}

	public static final double p03(double x) {
		x = taff01(-10, 10, x);
		double sum = 0.0;
		for (int k = 1; k <= 6; ++k) {
			sum += k * Math.sin((k + 1) * x + k);
		}
		return -sum + 16.5322;
	}

	public static final double p04(double x) {
		x = taff01(1.9, 3.9, x);
		return -(16 * x * x - 24 * x + 5) * Math.exp(-x) + 3.85045;
	}

	public static final double p05(double x) {
		x = taff01(0, 1.2, x);
		return -(1.4 - 3 * x) * Math.sin(18 * x) + 1.48907;
	}

	public static final double p06(double x) {
		x = taff01(-10, 10, x);
		return -(x + Math.sin(x)) * Math.exp(-x * x) + 0.824239;
	}

	public static final double p07(double x) {
		x = taff01(2.7, 7.5, x);
		return Math.sin(x) + Math.sin(10 * x / 3) + Math.log(x) - 0.84 * x + 3 + 1.6013;
	}

	public static final double p08(double x) {
		x = taff01(-10, 10, x);
		double sum = 0.0;
		for (int k = 1; k <= 6; ++k) {
			sum += k * Math.cos((k + 1) * x + k);
		}
		return -sum + 20.2526;
	}

	public static final double p09(double x) {
		x = taff01(3.1, 20.4, x);
		return Math.sin(x) + Math.sin(2 * x / 3) + 1.90596;
	}

	public static final double p10(double x) {
		x = taff01(0, 10, x);
		return -x * Math.sin(x) + 7.916727;
	}

	public static final double dp10(double x) {
		x = taff01(0, 10, x);
		return -Math.sin(x) - x * Math.cos(x);
	}

	public static final double p11(double x) {
		x = taff01(-Math.PI / 2, 2 * Math.PI, x);
		return 2 * Math.cos(x) + Math.cos(2 * x) + 1.5;
	}

	public static final double p12(double x) {
		x = taff01(0, 2 * Math.PI, x);
		final double y1 = Math.sin(x);
		final double y2 = Math.cos(x);
		return y1 * y1 * y1 + y2 * y2 * y2 + 1;
	}

	public static final double p13(double x) {
		x = taff01(0.001, 0.99, x);
		return -Math.pow(x, 2. / 3) - Math.pow(1 - x * x, 1. / 3) + 1.5874;
	}

	public static final double p14(double x) {
		x = taff01(0, 4, x);
		return -Math.exp(-x) * Math.sin(2 * Math.PI * x) + 0.788685;
	}

	public static final double p15(double x) {
		x = taff01(-5, 5, x);
		return (x * x - 5 * x + 6) / (x * x + 1) + 0.03553;
	}

	public static final double p18(double x) {
		x = taff01(0, 6, x);
		if (x <= 3) {
			return (x - 2) * (x - 2);
		} else {
			return 2 * Math.log(x - 2) + 1;
		}
	}

	public static final double p20(double x) {
		x = taff01(-10, 10, x);
		return -(x - Math.sin(x)) * Math.exp(-x * x) + 0.0634905;
	}

	public static final double p21(double x) {
		x = taff01(0, 10, x);
		return x * Math.sin(x) + x * Math.cos(2 * x) + 9.50835;
	}

	public static final double p22(double x) {
		x = taff01(0, 20, x);
		return Math.exp(-3 * x) - Math.pow(Math.sin(x), 3) - (Math.exp(-27 * Math.PI / 2) - 1);
	}

	public static void main(String[] args) {

		PiyavskiiAlgorithm alg = new PiyavskiiAlgorithm(1e-6, 500);
		Function<Double, Double> f = x -> -(1.4 - 3. * x) * Math.sin(18. * x);
		System.out.println(alg.optimize(f, 0., 1.2));

		//
//		int success = 0;
//		for (String key : ALL_FUNCTIONS.keySet()) {
//			final Function<Double, Double> func = ALL_FUNCTIONS.get(key);
//			final BrentAlgorithm optimizer = new BrentAlgorithm(1e-12, 1e-12, 3000);
//			final double min = optimizer.optimize(func, 0, 1).getOptimalPoint();
//			final double fmin = func.apply(min);
//			System.out.println(key + ": " + "error: " + fmin);
//			if (Math.abs(fmin - 0) < 1e-5) {
//				success += 1;
//			}
//		}
//		double success_rate = 1. * success / ALL_FUNCTIONS.size();
//		System.out.println("success rate: " + success_rate);
//
//		Function<Double, Double> f = UniUnconstrStandard::p10;
//		CalvinAlgorithm alg = new CalvinAlgorithm(1e-4, 500);
//		OptimizerSolution<Double, Double> opt = alg.optimize(f, 0, 1);
//		System.out.println("sol = " + opt.getOptimalPoint());
//		System.out.println("val = " + f.apply(opt.getOptimalPoint()));
//		System.out.println("fev = " + opt.getFEvals());
//		System.out.println("dfev = " + opt.getDFEvals());
//		System.out.println("conv = " + opt.converged());
	}

	private UniUnconstrStandard() {
	}
}
