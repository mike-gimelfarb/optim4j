/*
Copyright (c) 2020 Mike Gimelfarb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the > "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, > subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package min4j.testbeds;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import opt.multivariate.unconstrained.order0.cmaes.BiPopCmaesAlgorithm;
import utils.BlasMath;

public final class MultiUnconstrStandard {

	public static final Map<String, Function<double[], Double>> ALL_FUNCTIONS = new HashMap<String, Function<double[], Double>>() {

		private static final long serialVersionUID = -3852837778398908234L;

		{
			put("ackley", MultiUnconstrStandard::ackley);
			put("alpine1", MultiUnconstrStandard::alpine1);
			put("alpine2", MultiUnconstrStandard::alpine2);
			put("brown", MultiUnconstrStandard::brown);
			put("chung_reynolds", MultiUnconstrStandard::chung_reynolds);
			put("cosine_mixture", MultiUnconstrStandard::cosine_mixture);
			put("csendes", MultiUnconstrStandard::csendes);
			put("dixon_price", MultiUnconstrStandard::dixon_price);
			put("exponential", MultiUnconstrStandard::exponential);
			put("griewank", MultiUnconstrStandard::griewank);
			put("max_abs", MultiUnconstrStandard::max_abs);
			put("plateau", MultiUnconstrStandard::plateau);
			put("qing", MultiUnconstrStandard::qing);
			put("quartic", MultiUnconstrStandard::quartic);
			put("rastrigin", MultiUnconstrStandard::rastrigin);
			put("rosenbrock", MultiUnconstrStandard::rosenbrock);
			put("rotated_hyper_ellipsoid", MultiUnconstrStandard::rotated_hyper_ellipsoid);
			put("salomon", MultiUnconstrStandard::salomon);
			put("schwefel", MultiUnconstrStandard::schwefel);
			put("shubert", MultiUnconstrStandard::shubert);
			put("sphere", MultiUnconstrStandard::sphere);
			put("styblinski_tang", MultiUnconstrStandard::styblinski_tang);
			put("sum_abs", MultiUnconstrStandard::sum_abs);
			put("sum_diff_pow", MultiUnconstrStandard::sum_diff_pow);
			put("sum_squares", MultiUnconstrStandard::sum_squares);
			put("trid", MultiUnconstrStandard::trid);
			put("zakharov", MultiUnconstrStandard::zakharov);
		}
	};

	private static final Random RAND = new Random();

	public static final double[] taff5(final double oa, final double ob, final double[] x) {
		final double[] y = new double[x.length];
		for (int i = 0; i < x.length; ++i) {
			y[i] = (oa + ob) / 2 + (ob - oa) / 10 * x[i];
		}
		return y;
	}

	public static final double ackley(final double[] x) {
		final double[] y = taff5(-32, 32, x);
		final int d = y.length;
		double sumcos = 0.0;
		for (final double e : y) {
			sumcos += Math.cos(2 * Math.PI * e) / d;
		}
		final double norm = BlasMath.denorm(d, y);
		return -20.0 * Math.exp(-0.2 * norm / Math.sqrt(d)) - Math.exp(sumcos) + 20.0 + Math.E;
	};

	public static final double alpine1(final double[] x) {
		final double[] y = taff5(-10, 10, x);
		double result = 0.0;
		for (final double e : y) {
			result += Math.abs(e * Math.sin(e) + 0.1 * e);
		}
		return result;
	};

	public static final double alpine2(final double[] x) {
		final double[] y = taff5(0, 10, x);
		double result = 1.0;
		for (final double e : y) {
			result *= Math.sqrt(Math.abs(e)) * Math.sin(e);
		}
		result = -(result - Math.pow(2.808, y.length));
		return result;
	};

	public static final double brown(final double[] x) {
		final double[] y = taff5(-1, 4, x);
		double result = 0.0;
		for (int i = 0; i < y.length - 1; i++) {
			final double x2i = y[i] * y[i];
			final double x2i2 = y[i + 1] * y[i + 1];
			result += Math.pow(x2i, x2i2 + 1) + Math.pow(x2i2, x2i + 1);
		}
		return result;
	};

	public static final double chung_reynolds(final double[] x) {
		final double[] y = taff5(-50, 50, x);
		final double arg = BlasMath.ddotm(y.length, y, 1, y, 1);
		return arg * arg;
	};

	public static final double cosine_mixture(final double[] x) {
		final double[] y = taff5(-1, 1, x);
		double result = 0.0;
		for (final double e : y) {
			result += 0.1 * Math.cos(5 * Math.PI * e);
		}
		result += BlasMath.ddotm(y.length, y, 1, y, 1);
		result -= 0.1 * y.length;
		return result;
	};

	public static final double csendes(final double[] x) {
		final double[] y = taff5(-1, 1, x);
		double result = 0.0;
		for (final double e : y) {
			result += Math.pow(e, 6) * (2 + Math.sin(1 / e));
		}
		return result;
	};

	public static final double dixon_price(final double[] x) {
		final double[] y = taff5(-10, 10, x);
		double result = (y[0] - 1) * (y[0] - 1);
		for (int i = 1; i < y.length; i++) {
			final double dxi1 = 2 * y[i] * y[i] - y[i - 1];
			result += (i + 1) * dxi1 * dxi1;
		}
		return result;
	};

	public static final double exponential(final double[] x) {
		final double[] y = taff5(-1, 1, x);
		final double arg = BlasMath.ddotm(y.length, y, 1, y, 1);
		return -Math.exp(-0.5 * arg) + 1;
	};

	public static final double griewank(final double[] x) {
		final double[] y = taff5(-20, 20, x);
		double result1 = BlasMath.ddotm(y.length, y, 1, y, 1) / 4000;
		double result2 = 1.0;
		for (int i = 0; i < y.length; i++) {
			result2 *= Math.cos(y[i] / Math.sqrt(i + 1));
		}
		return result1 - result2 + 1;
	};

	public static final double max_abs(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result = Math.max(result, Math.abs(e));
		}
		return result;
	};

	public static final double plateau(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result += Math.floor(Math.abs(e));
		}
		return result;
	};

	public static final double qing(final double[] x) {
		final double[] y = taff5(-50, 50, x);
		double result = 0.0;
		for (int i = 0; i < y.length; i++) {
			final double arg = y[i] * y[i] - (i + 1);
			result += arg * arg;
		}
		return result;
	};

	public static final double quartic(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length; i++) {
			result += (i + 1) * Math.pow(x[i], 4);
		}
		return result;
	};

	public static final double rastrigin(final double[] x) {
		double result = 10 * x.length;
		for (final double e : x) {
			result += e * e - 10 * Math.cos(2 * Math.PI * e);
		}
		return result;
	};

	public static final double rosenbrock(final double[] x) {
		final double[] y = taff5(-5, 10, x);
		double result = 0.0;
		for (int i = 0; i < y.length - 1; i++) {
			final double dxi1 = y[i + 1] - y[i] * y[i];
			final double dxi2 = y[i] - 1;
			result += 100 * dxi1 * dxi1 + dxi2 * dxi2;
		}
		return result;
	};

	public static final double rotated_hyper_ellipsoid(final double[] x) {
		final double[] y = taff5(-65, 65, x);
		double result = 0.0;
		for (int i = 0; i < y.length; i++) {
			for (int j = 0; j <= i; j++) {
				result += y[j] * y[j];
			}
		}
		return result;
	};

	public static final double salomon(final double[] x) {
		final double[] y = taff5(-10, 10, x);
		final double norm = BlasMath.denorm(y.length, y);
		return 1 - Math.cos(2 * Math.PI * norm) + 0.1 * norm;
	};

	public static final double schwefel(final double[] x) {
		final double[] y = taff5(-500, 500, x);
		double result = 418.982887 * y.length;
		for (final double e : y) {
			result -= e * Math.sin(Math.sqrt(Math.abs(e)));
		}
		return result;
	};

	public static final double shubert(final double[] x) {
		final double[] y = taff5(-10, 10, x);
		double result = 0.0;
		for (final double e : y) {
			for (int j = 1; j <= 5; ++j) {
				result += j * Math.sin((j + 1) * e + j);
			}
		}
		return result + 29.6733337;
	};

	public static final double sphere(final double[] x) {
		return BlasMath.ddotm(x.length, x, 1, x, 1);
	};

	public static final double styblinski_tang(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			final double x2i = e * e;
			result += 0.5 * (x2i * x2i - 16 * x2i + 5 * e);
		}
		return result + 39.16599 * x.length;
	};

	public static final double sum_abs(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result += Math.abs(e);
		}
		return result;
	};

	public static final double sum_diff_pow(final double[] x) {
		final double[] y = taff5(-1, 1, x);
		double result = 0.0;
		for (int i = 1; i <= y.length; i++) {
			result += Math.pow(Math.abs(y[i - 1]), i + 1);
		}
		return result;
	};

	public static final double sum_squares(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length; i++) {
			result += (i + 1) * x[i] * x[i];
		}
		return result;
	};

	public static final double trid(final double[] x) {
		final int d = x.length;
		final double[] y = taff5(-d * d, d * d, x);
		double result = 0.0;
		for (int i = 0; i < y.length; i++) {
			result += (y[i] - 1) * (y[i] - 1);
			if (i >= 1) {
				result -= y[i] * y[i - 1];
			}
		}
		return result + (1. / 6) * d * (d + 4) * (d - 1);
	};

	public static final double zakharov(final double[] x) {
		final double[] y = taff5(-5, 10, x);
		double result1 = BlasMath.ddotm(y.length, y, 1, y, 1);
		double result2 = 0.0;
		for (int i = 0; i < y.length; i++) {
			result2 += 0.5 * (i + 1) * y[i];
		}
		final double args = result2 * result2;
		return result1 + args + args * args;
	};

	public static void main(String[] args) {
		final int scales = 7;
		final int n = 5;
		for (int i = 0; i <= scales * 5; ++i) {
			final int fevs = (int) Math.pow(10, i / 5.0);
			int passed = 0;
			for (final String key : ALL_FUNCTIONS.keySet()) {
				final Function<double[], Double> func = ALL_FUNCTIONS.get(key);
				double besterr = Double.POSITIVE_INFINITY;
				double bestfit = Double.POSITIVE_INFINITY;
				for (int j = -8; j <= 2; ++j) {
					BiPopCmaesAlgorithm alg = new BiPopCmaesAlgorithm(-1, 1e-12, 2.0, fevs, 1000, false);
					final double[] guess = new double[n];
					for (int l = 0; l < n; ++l) {
						guess[l] = RAND.nextDouble() * 2 - 1;
					}
					double[] res = alg.optimize(func, guess);
					if (res != null) {
						double fit = func.apply(res);
						double ferr = Math.abs(fit - 0.0);
						besterr = Math.min(ferr, besterr);
						bestfit = Math.min(bestfit, fit);
						if (ferr <= Math.pow(10, j)) {
							++passed;
						} else {
							System.out.println(key + " err " + ferr + " > " + Math.pow(10, j));
						}
					} else {
						System.out.println(key + " err " + "null" + " > " + Math.pow(10, j));
					}
				}
			}
			double passpercent = (double) passed / (ALL_FUNCTIONS.size() * 11);
			System.out.println(Math.log10(fevs) + "\t" + passpercent);
		}
	}

	private MultiUnconstrStandard() {
	}
}
