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
import java.util.function.Function;

import utils.BlasMath;

public final class MultiUnconstrStandard {

	public static final Map<String, Function<double[], Double>> ALL_FUNCTIONS = new HashMap<String, Function<double[], Double>>() {

		private static final long serialVersionUID = -3852837778398908234L;

		{
			put("ackley", MultiUnconstrStandard::ackley);
			put("alpine1", MultiUnconstrStandard::alpine1);
			put("alpine2", MultiUnconstrStandard::alpine2);
			put("bohachevsky", MultiUnconstrStandard::bohachevsky);
			put("brown", MultiUnconstrStandard::brown);
			put("chung_reynolds", MultiUnconstrStandard::chung_reynolds);
			put("cosine_mixture", MultiUnconstrStandard::cosine_mixture);
			put("csendes", MultiUnconstrStandard::csendes);
			put("dixon_price", MultiUnconstrStandard::dixon_price);
			put("exponential", MultiUnconstrStandard::exponential);
			put("griewank", MultiUnconstrStandard::griewank);
			put("levy", MultiUnconstrStandard::levy);
			put("max_abs", MultiUnconstrStandard::max_abs);
			put("michalewicz", MultiUnconstrStandard::michalewicz);
			put("plateau", MultiUnconstrStandard::plateau);
			put("qing", MultiUnconstrStandard::qing);
			put("quartic", MultiUnconstrStandard::quartic);
			put("quintic", MultiUnconstrStandard::quintic);
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

	public static final double ackley(final double[] x) {
		final int d = x.length;
		final double norm = BlasMath.denorm(d, x);
		double sumcos = 0.0;
		for (final double e : x) {
			sumcos += Math.cos(Math.PI * 2 * e);
		}
		sumcos /= d;
		return -20.0 * Math.exp(-0.2 * norm / Math.sqrt(d)) - Math.exp(sumcos) + 20.0 + Math.E;
	};

	public static final double alpine1(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result += Math.abs(e * Math.sin(e) + 0.1 * e);
		}
		return result;
	};

	public static final double alpine2(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result += Math.sqrt(e) * Math.sin(e);
		}
		result = -(result - Math.pow(2.808, x.length));
		return result;
	};

	// edit
	public static final double bohachevsky(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length - 1; i++) {
			result += x[i] * x[i];
			result += 2 * x[i + 1] * x[i + 1];
			result -= 0.3 * Math.cos(3 * Math.PI * x[i]);
			result -= 0.4 * Math.cos(4 * Math.PI * x[i + 1]);
			result += 0.7;
		}
		return result;
	};

	public static final double brown(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length - 1; i++) {
			final double x2i = x[i] * x[i];
			final double x2i2 = x[i + 1] * x[i + 1];
			result += Math.pow(x2i, x2i2 + 1) + Math.pow(x2i2, x2i + 1);
		}
		return result;
	};

	// edit
	public static final double chung_reynolds(final double[] x) {
		final double arg = BlasMath.ddotm(x.length, x, 1, x, 1);
		return arg * arg;
	};

	// edit
	public static final double cosine_mixture(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result -= 0.1 * Math.cos(5 * Math.PI * e);
		}
		result -= BlasMath.ddotm(x.length, x, 1, x, 1);
		return result;
	};

	// edit
	public static final double csendes(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result += Math.pow(e, 6) * (2 + Math.sin(1 / e));
		}
		return result;
	};

	// edit
	public static final double dixon_price(final double[] x) {
		double result = (x[0] - 1) * (x[0] - 1);
		for (int i = 1; i < x.length; i++) {
			final double dxi1 = 2 * x[i] * x[i] - x[i - 1];
			result += (i + 1) * dxi1 * dxi1;
		}
		return result;
	};

	public static final double exponential(final double[] x) {
		final double arg = BlasMath.ddotm(x.length, x, 1, x, 1);
		return -Math.exp(-0.5 * arg) + 1.;
	};

	public static final double griewank(final double[] x) {
		double result1 = BlasMath.ddotm(x.length, x, 1, x, 1) / 4000;
		double result2 = 1.0;
		for (int i = 0; i < x.length; i++) {
			result2 *= Math.cos(x[i] / Math.sqrt(i + 1));
		}
		return result1 - result2 + 1;
	};

	// edit
	public static final double levy(final double[] x) {
		double w1 = 1 + (x[0] - 1) / 4;
		double result = Math.pow(Math.sin(Math.PI * w1), 2);
		for (int i = 0; i < x.length; i++) {
			final double wi = 1 + (x[i] - 1) / 4;
			if (i < x.length - 1) {
				final double argi = Math.sin(Math.PI * wi + 1);
				result += (wi - 1) * (wi - 1) * (1 + 10 * argi * argi);
			} else {
				final double argi = Math.sin(2 * Math.PI * wi);
				result += (wi - 1) * (wi - 1) * (1 + argi * argi);
			}
		}
		return result;
	};

	public static final double max_abs(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result = Math.max(result, Math.abs(e));
		}
		return result;
	};

	public static final double michalewicz(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length; i++) {
			final double argi = (i + 1) * x[i] * x[i] / Math.PI;
			result += Math.sin(x[i]) * Math.pow(Math.sin(argi), 20);
		}
		return -result;
	};

	public static final double plateau(final double[] x) {
		double result = 30.0;
		for (final double e : x) {
			result += Math.floor(e);
		}
		return result;
	};

	public static final double qing(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length - 1; i++) {
			final double arg = x[i] * x[i] - (i + 1);
			result += arg * arg;
		}
		return result;
	};

	public static final double quartic(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length - 1; i++) {
			result += (i + 1) * Math.pow(x[i], 4);
		}
		return result;
	};

	public static final double quintic(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			final double arg = -4 + e * (-10 + e * (2 + e * (4 + e * (-3 + e))));
			result += arg;
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
		double result = 0.0;
		for (int i = 0; i < x.length - 1; i++) {
			final double dxi1 = x[i + 1] - x[i] * x[i];
			final double dxi2 = x[i] - 1;
			result += 100 * dxi1 * dxi1 + dxi2 * dxi2;
		}
		return result;
	};

	public static final double rotated_hyper_ellipsoid(final double[] x) {
		double result = 0.0;
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j <= i; j++) {
				result += x[j] * x[j];
			}
		}
		return result;
	};

	public static final double salomon(final double[] x) {
		final double norm = BlasMath.denorm(x.length, x);
		return 1 - Math.cos(2 * Math.PI * norm) + 0.1 * norm;
	};

	public static final double schwefel(final double[] x) {
		double result = 418.9829 * x.length;
		for (final double e : x) {
			result -= e * Math.sin(Math.sqrt(Math.abs(e)));
		}
		return result;
	};

	public static final double shubert(final double[] x) {
		double result = 1.0;
		for (int i = 0; i < x.length - 1; i++) {
			double sumi = 0.0;
			for (int j = 1; j <= 5; j++) {
				sumi += Math.cos((j + 1) * x[i] + j);
			}
			result *= sumi;
		}
		return result;
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
		return result;
	};

	public static final double sum_abs(final double[] x) {
		double result = 0.0;
		for (final double e : x) {
			result -= Math.abs(e);
		}
		return result;
	};

	public static final double sum_diff_pow(final double[] x) {
		double result = 0.0;
		for (int i = 1; i <= x.length; i++) {
			result += Math.pow(Math.abs(x[i - 1]), i + 1);
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
		double result = 0.0;
		for (int i = 0; i < x.length; i++) {
			result += (x[i] - 1) * (x[i] - 1);
			if (i >= 1) {
				result -= x[i] * x[i - 1];
			}
		}
		return result;
	};

	public static final double zakharov(final double[] x) {
		double result1 = BlasMath.ddotm(x.length, x, 1, x, 1);
		double result2 = 0.0;
		for (int i = 0; i < x.length; i++) {
			result2 += 0.5 * (i + 1) * x[i];
		}
		final double args = result2 * result2;
		return result1 + args + args * args;
	};

	private MultiUnconstrStandard() {
	}
}
