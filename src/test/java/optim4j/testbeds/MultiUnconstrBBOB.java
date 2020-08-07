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
package optim4j.testbeds;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import opt.OptimizerSolution;
import opt.multivariate.unconstrained.order0.cmaes.ActiveCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.cmaes.BiPopCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.cmaes.CholeskyCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.cmaes.CmaesAlgorithm;
import opt.multivariate.unconstrained.order0.cmaes.IPopCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.cmaes.LmCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.cmaes.SepCmaesAlgorithm;
import opt.multivariate.unconstrained.order0.direct.CrsAlgorithm;
import opt.multivariate.unconstrained.order0.direct.DirectAlgorithm;
import opt.multivariate.unconstrained.order0.direct.NelderMeadAlgorithm;
import opt.multivariate.unconstrained.order0.direct.RosenbrockAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AdaptivePsoAlgorithm;
import opt.multivariate.unconstrained.order0.evol.AmalgamAlgorithm;
import opt.multivariate.unconstrained.order0.evol.CcPsoAlgorithm;
import opt.multivariate.unconstrained.order0.evol.CsoAlgorithm;
import opt.multivariate.unconstrained.order0.evol.DifferentialSearchAlgorithm;
import opt.multivariate.unconstrained.order0.evol.EschAlgorithm;
import opt.multivariate.unconstrained.order0.evol.SadeAlgorithm;
import opt.multivariate.unconstrained.order0.quad.BobyqaAlgorithm;
import opt.multivariate.unconstrained.order0.quad.NewuoaAlgorithm;
import utils.BlasMath;
import utils.RealMath;

public class MultiUnconstrBBOB {

	public static final Map<String, Function<double[], Double>> ALL_FUNCTIONS = new LinkedHashMap<String, Function<double[], Double>>() {

		private static final long serialVersionUID = -3852837778398908234L;

		{
			put("sphere", MultiUnconstrBBOB::sphere);
			put("ellipsoidal", MultiUnconstrBBOB::ellipsoidal);
			put("rastrigin", MultiUnconstrBBOB::rastrigin);
			put("bucheRastrigin", MultiUnconstrBBOB::bucheRastrigin);
			put("linearSlope", MultiUnconstrBBOB::linearSlope);
			put("attractiveSector", MultiUnconstrBBOB::attractiveSector);
			put("stepEllipsoidal", MultiUnconstrBBOB::stepEllipsoidal);
			put("rosenbrock", MultiUnconstrBBOB::rosenbrock);
			put("rotatedRosenbrock", MultiUnconstrBBOB::rotatedRosenbrock);
			put("ellipsoidal2", MultiUnconstrBBOB::ellipsoidal2);
			put("discus", MultiUnconstrBBOB::discus);
			put("bentCigar", MultiUnconstrBBOB::bentCigar);
			put("sharpRidge", MultiUnconstrBBOB::sharpRidge);
			put("differentPowers", MultiUnconstrBBOB::differentPowers);
			put("rastrigin2", MultiUnconstrBBOB::rastrigin2);
			put("weierstrass", MultiUnconstrBBOB::weierstrass);
			put("schafferF7", MultiUnconstrBBOB::schafferF7);
			put("schafferF7ill", MultiUnconstrBBOB::schafferF7ill);
			put("griewankRosenbrock", MultiUnconstrBBOB::griewankRosenbrock);
			put("schwefel", MultiUnconstrBBOB::schwefel);
			put("gallagher101", MultiUnconstrBBOB::gallagher101);
			put("gallagher21", MultiUnconstrBBOB::gallagher21);
			put("katsuura", MultiUnconstrBBOB::katsuura);
			put("lunacekBiRastrigin", MultiUnconstrBBOB::lunacekBiRastrigin);
		}
	};

	private static final Random RAND = new Random();

	private static final double[] lambda_mult(final int ialpha, final double[] x) {
		final double base = lambda_base[ialpha];
		double cumpow = 1.0;
		final double[] result = new double[x.length];
		for (int i = 0; i < x.length; ++i) {
			result[i] = cumpow * x[i];
			cumpow *= base;
		}
		return result;
	}

	private static final double fpen(final double[] x) {
		double res = 0.0;
		for (final double xi : x) {
			final double resi = Math.max(0, Math.abs(xi) - 5);
			res += resi * resi;
		}
		return res;
	}

	private static final double[] onepm(final int d) {
		final double[] res = new double[d];
		for (int i = 0; i < d; ++i) {
			res[i] = RAND.nextBoolean() ? 1 : -1;
		}
		return res;
	}

	private static final double[] tasy(final double beta, final double[] x) {
		final int d = x.length;
		final double[] res = new double[d];
		for (int i = 0; i < d; ++i) {
			res[i] = x[i] > 0 ? Math.pow(x[i], 1 + beta * i / (d - 1) * Math.sqrt(x[i])) : x[i];
		}
		return res;
	}

	private static final double tosz(final double x) {
		final double c1 = x > 0 ? 10 : 5.5;
		final double c2 = x > 0 ? 7.9 : 3.1;
		final double xhat = x > 0 || x < 0 ? Math.log(Math.abs(x)) : 0;
		final double sign = Math.signum(x);
		return sign * Math.exp(xhat + 0.049 * (Math.sin(c1 * xhat) + Math.sin(c2 * xhat)));
	}

	private static final double[] tosz(final double[] x) {
		final double[] res = new double[x.length];
		for (int i = 0; i < x.length; ++i) {
			res[i] = tosz(x[i]);
		}
		return res;
	}

	private static final double[] mmult(final double[][] mat, final double[] x) {
		final double[] res = new double[mat.length];
		for (int i = 0; i < mat.length; ++i) {
			res[i] = BlasMath.ddotm(x.length, mat[i], 1, x, 1);
		}
		return res;
	}

	private static final double[][] orthogonal(final int dim) {
		final double[][] mat = new double[dim][dim];
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				mat[i][j] = RAND.nextGaussian();
			}
		}
		return QR.qr(mat)[0];
	}

	// cache fixed quantities for quick computation
	private static int D;
	private static double[] one_pm;
	private static double[][] Q, R;
	private static double[] lambda_base;

	private static double[] pow2j;
	private static double[] pow05k, pow3k;
	private static double lbr_mu1, lbr_s;
	private static double[][] y_101, C_101;
	private static double[][] y_21, C_21;
	private static double base1, base2, base3, base4;

	private static void shuffle(final double[] values, final int istart) {
		for (int i = istart; i < values.length; i++) {
			final int randomIndexToSwap = RAND.nextInt(values.length - istart) + istart;
			final double temp = values[randomIndexToSwap];
			values[randomIndexToSwap] = values[i];
			values[i] = temp;
		}
	}

	public static final void prepare(final int dim) {
		D = dim;
		one_pm = onepm(D);
		Q = orthogonal(D);
		R = orthogonal(D);

		// lambda base
		// 0.01, 10, 100, 1000
		lambda_base = new double[] { Math.pow(0.01, 1 / (2. * (dim - 1))), Math.pow(10, 1 / (2. * (dim - 1))),
				Math.pow(100, 1 / (2. * (dim - 1))), Math.pow(1000, 1 / (2. * (dim - 1))) };

		// lookup tables for Weierstrass function
		pow05k = new double[12];
		pow3k = new double[12];
		for (int k = 0; k <= 11; ++k) {
			pow05k[k] = Math.pow(0.5, k);
			pow3k[k] = Math.pow(3, k);
		}

		// lookup table for Katsuura function
		pow2j = new double[33];
		for (int j = 0; j <= 32; ++j) {
			pow2j[j] = Math.pow(2., j);
		}

		// values for Lunacek bi-Rastrigin
		lbr_s = 1. - 1. / (2 * Math.sqrt(dim + 20) - 8.2);
		lbr_mu1 = -Math.sqrt((2.5 * 2.5 - 1.) / lbr_s);

		// for Gallagher 101 function
		y_101 = new double[101][dim];
		for (int i = 0; i < D; ++i) {
			y_101[0][i] = -4.0 + 8.0 * RAND.nextDouble();
		}
		for (int k = 1; k < 101; ++k) {
			for (int i = 0; i < D; ++i) {
				y_101[k][i] = -5.0 + 10.0 * RAND.nextDouble();
			}
		}
		double[] alpha = new double[101];
		for (int j = 0; j <= 99; ++j) {
			alpha[j + 1] = Math.pow(1000, 2.0 * j / 99);
		}
		shuffle(alpha, 1);
		alpha[0] = 1000.;
		C_101 = new double[101][dim];
		for (int i = 0; i < 101; ++i) {
			final double base = Math.pow(alpha[i], 1 / (2. * (dim - 1)));
			double cumpow = 1.0;
			for (int j = 0; j < dim; ++j) {
				C_101[i][j] = cumpow;
				cumpow *= base;
			}
			shuffle(C_101[i], 0);
			final double denom = Math.pow(alpha[i], 0.25);
			for (int j = 0; j < dim; ++j) {
				C_101[i][j] /= denom;
			}
		}

		// for Gallagher 21 function
		y_21 = new double[21][dim];
		for (int i = 0; i < D; ++i) {
			y_21[0][i] = -3.92 + 3.92 * 2 * RAND.nextDouble();
		}
		for (int k = 1; k < 21; ++k) {
			for (int i = 0; i < D; ++i) {
				y_21[k][i] = -4.9 + 4.9 * 2 * RAND.nextDouble();
			}
		}
		alpha = new double[21];
		for (int j = 0; j <= 19; ++j) {
			alpha[j + 1] = Math.pow(1000, 2.0 * j / 19);
		}
		shuffle(alpha, 1);
		alpha[0] = 1000000.;
		C_21 = new double[21][dim];
		for (int i = 0; i < 21; ++i) {
			final double base = Math.pow(alpha[i], 1 / (2. * (dim - 1)));
			double cumpow = 1.0;
			for (int j = 0; j < dim; ++j) {
				C_21[i][j] = cumpow;
				cumpow *= base;
			}
			shuffle(C_21[i], 0);
			final double denom = Math.pow(alpha[i], 0.25);
			for (int j = 0; j < dim; ++j) {
				C_21[i][j] /= denom;
			}
		}

		// other values
		base1 = Math.pow(10, 6. / (dim - 1));
		base2 = Math.pow(10, .5 / (dim - 1));
		base3 = Math.pow(10, 1. / (dim - 1));
		base4 = Math.pow(10, 2. / (dim - 1));
	}

	// separable functions
	public static final double sphere(final double[] x) {
		return BlasMath.ddotm(x.length, x, 1, x, 1);
	}

	public static final double ellipsoidal(final double[] x) {
		final int d = x.length;
		final double base = base1;
		final double[] z = tosz(x);
		double cumpow = 1.0;
		double res = 0.0;
		for (int i = 0; i < d; ++i) {
			res += cumpow * z[i] * z[i];
			cumpow *= base;
		}
		return res;
	}

	public static final double rastrigin(final double[] x) {
		final int d = x.length;
		final double[] z = lambda_mult(1, tasy(0.2, tosz(x)));
		double sum = d;
		for (final double e : z) {
			sum -= Math.cos(2 * Math.PI * e);
		}
		final double result = 10 * sum + BlasMath.ddotm(d, z, 1, z, 1);
		return result;
	}

	public static final double bucheRastrigin(final double[] x) {
		final int d = x.length;
		final double[] z = tosz(x);
		final double base = base2;
		double cumpow = 1.0;
		for (int i = 1; i <= d; ++i) {
			if (z[i - 1] > 0 && (i % 2 != 0)) {
				z[i - 1] *= 10 * cumpow;
			} else {
				z[i - 1] *= cumpow;
			}
			cumpow *= base;
		}
		double sum = d;
		for (final double e : z) {
			sum -= Math.cos(2 * Math.PI * e);
		}
		final double result = 10 * sum + BlasMath.ddotm(d, z, 1, z, 1) + 100 * fpen(x);
		return result;
	}

	public static final double linearSlope(final double[] x) {
		final int d = x.length;
		final double base = base3;
		double cumpow = 1.0;
		double sum = 0.0;
		for (int i = 0; i < d; ++i) {
			final double xopti = 5 * one_pm[i];
			final double zi = xopti * x[i] < 25 ? x[i] : xopti;
			final double si = Math.signum(xopti) * cumpow;
			sum += 5 * Math.abs(si);
			sum -= si * zi;
			cumpow *= base;
		}
		return sum;
	}

	// functions with low or moderate conditioning
	public static final double attractiveSector(final double[] x) {
		final double[] z = mmult(Q, lambda_mult(1, mmult(R, x)));
		double res = BlasMath.ddotm(z.length, z, 1, z, 1);
		res = tosz(res);
		res = Math.pow(res, 0.9);
		return res;
	}

	public static final double stepEllipsoidal(final double[] x) {
		final int d = x.length;
		final double[] zhat = lambda_mult(1, mmult(R, x));
		final double zhat1 = zhat[0];
		for (int i = 0; i < d; ++i) {
			zhat[i] = Math.abs(zhat[i]) > 0.5 ? Math.floor(0.5 + zhat[i]) : Math.floor(0.5 + 10.0 * zhat[i]) / 10.0;
		}
		final double[] z = mmult(Q, zhat);
		final double base = base4;
		double cumpow = 1.0;
		double sum = 0.0;
		for (int i = 0; i < d; ++i) {
			sum += cumpow * z[i] * z[i];
			cumpow *= base;
		}
		final double result = 0.1 * Math.max(Math.abs(zhat1) / 10000., sum) + fpen(x);
		return result;
	}

	public static final double rosenbrock(final double[] x) {
		final int d = x.length;
		final double[] z = new double[d];
		for (int i = 0; i < d; ++i) {
			z[i] = Math.max(1.0, Math.sqrt(d) / 8) * x[i] + 1;
		}
		double res = 0.0;
		for (int i = 0; i < d - 1; ++i) {
			final double argi = z[i] * z[i] - z[i + 1];
			res += 100.0 * argi * argi + (z[i] - 1) * (z[i] - 1);
		}
		return res;
	}

	public static final double rotatedRosenbrock(final double[] x) {
		final int d = x.length;
		final double[] z = mmult(R, x);
		for (int i = 0; i < d; ++i) {
			z[i] = Math.max(1.0, Math.sqrt(d) / 8) * z[i] + 0.5;
		}
		double res = 0.0;
		for (int i = 0; i < d - 1; ++i) {
			final double argi = z[i] * z[i] - z[i + 1];
			res += 100.0 * argi * argi + (z[i] - 1) * (z[i] - 1);
		}
		return res;
	}

	public static final double ellipsoidal2(final double[] x) {
		final int d = x.length;
		final double[] z = tosz(mmult(R, x));
		final double base = base1;
		double cumpow = 1.0;
		double res = 0.0;
		for (int i = 0; i < d; ++i) {
			res += cumpow * z[i] * z[i];
			cumpow *= base;
		}
		return res;
	}

	public static final double discus(final double[] x) {
		final int d = x.length;
		final double[] z = tosz(mmult(R, x));
		double res = 1000000 * z[0] * z[0];
		for (int i = 1; i < d; ++i) {
			res += z[i] * z[i];
		}
		return res;
	}

	public static final double bentCigar(final double[] x) {
		final double[] z = mmult(R, tasy(0.5, mmult(R, x)));
		double sum = 0.0;
		for (int i = 1; i < x.length; ++i) {
			sum += z[i] * z[i];
		}
		final double result = z[0] * z[0] + 1000000 * sum;
		return result;
	}

	public static final double sharpRidge(final double[] x) {
		final double[] z = mmult(Q, lambda_mult(1, mmult(R, x)));
		double res = 0.0;
		for (int i = 1; i < x.length; ++i) {
			res += z[i] * z[i];
		}
		final double result = z[0] * z[0] + 100 * Math.sqrt(res);
		return result;
	}

	public static final double differentPowers(final double[] x) {
		final int d = x.length;
		final double[] z = mmult(R, x);
		double res = 0.0;
		for (int i = 0; i < d; ++i) {
			res += Math.pow(Math.abs(z[i]), 2 + 4. * i / (d - 1));
		}
		res = Math.sqrt(res);
		return res;
	}

	// multi-modal functions
	public static final double rastrigin2(final double[] x) {
		final double[] z = mmult(R, lambda_mult(1, mmult(Q, tasy(0.2, tosz(mmult(R, x))))));
		double res = x.length;
		for (final double e : z) {
			res -= Math.cos(2.0 * Math.PI * e);
		}
		final double result = 10 * res + BlasMath.ddotm(z.length, z, 1, z, 1);
		return result;
	}

	public static final double weierstrass(final double[] x) {
		final double[] z = mmult(R, lambda_mult(0, mmult(Q, tosz(mmult(R, x)))));
		final double f0 = -4095.0 / 2048.0;
		double res = 0.0;
		for (int k = 0; k <= 11; ++k) {
			for (final double e : z) {
				res += pow05k[k] * Math.cos(2 * Math.PI * pow3k[k] * (e + 0.5));
			}
		}
		final double result = 10 * Math.pow(res / x.length - f0, 3) + (10. / x.length) * fpen(x);
		return result;
	}

	public static final double schafferF7(final double[] x) {
		final double[] z = lambda_mult(1, mmult(Q, tasy(0.5, mmult(R, x))));
		double res = 0.0;
		for (int i = 0; i < x.length - 1; ++i) {
			final double si = RealMath.hypot(z[i], z[i + 1]);
			final double sqrtsi = Math.sqrt(si);
			final double argi = Math.sin(50 * Math.pow(si, 0.2));
			final double term = sqrtsi + sqrtsi * argi * argi;
			res += term;
		}
		res /= (x.length - 1);
		final double result = res * res + 10 * fpen(x);
		return result;
	}

	public static final double schafferF7ill(final double[] x) {
		final double[] z = lambda_mult(3, mmult(Q, tasy(0.5, mmult(R, x))));
		double res = 0.0;
		for (int i = 0; i < x.length - 1; ++i) {
			final double si = RealMath.hypot(z[i], z[i + 1]);
			final double sqrtsi = Math.sqrt(si);
			final double argi = Math.sin(50 * Math.pow(si, 0.2));
			final double term = sqrtsi + sqrtsi * argi * argi;
			res += term;
		}
		res /= (x.length - 1);
		final double result = res * res + 10 * fpen(x);
		return result;
	}

	public static final double griewankRosenbrock(final double[] x) {
		final int d = x.length;
		final double[] z = mmult(R, x);
		for (int i = 0; i < d; ++i) {
			z[i] = Math.max(1.0, Math.sqrt(d) / 8) * z[i] + 0.5;
		}
		double res = 0.0;
		for (int i = 0; i < d - 1; ++i) {
			final double argi = z[i] * z[i] - z[i + 1];
			final double si = 100 * argi * argi + (z[i] - 1) * (z[i] - 1);
			res += si / 4000 - Math.cos(si);
		}
		final double result = 10. / (d - 1) * res + 10;
		return result;
	}

	// multi-modal functions with weak global structure
	public static final double schwefel(final double[] x) {
		final int d = x.length;
		final double[] xhat = new double[d];
		final double[] xopt = new double[d];
		for (int i = 0; i < d; ++i) {
			xhat[i] = 2 * one_pm[i] * x[i];
			xopt[i] = 4.2096874633 / 2 * one_pm[i];
		}
		double[] z = new double[d];
		z[0] = xhat[0] - 2 * Math.abs(xopt[0]);
		for (int i = 1; i < d; ++i) {
			z[i] = xhat[i] + 0.25 * (xhat[i - 1] - 2 * Math.abs(xopt[i - 1]));
			z[i] -= 2 * Math.abs(xopt[i]);
		}
		z = lambda_mult(2, z);
		for (int i = 0; i < d; ++i) {
			z[i] = 100 * (z[i] + 2 * Math.abs(xopt[i]));
		}
		double res = 0.0;
		for (int i = 0; i < d; ++i) {
			res += z[i] * Math.sin(Math.sqrt(Math.abs(z[i])));
		}
		res *= (-0.01 / d);
		res += 4.189828872724339;
		for (int i = 0; i < d; ++i) {
			z[i] /= 100.0;
		}
		res += 100.0 * fpen(z);
		return res;
	}

	public static final double gallagher101(final double[] x) {
		double res = Double.NEGATIVE_INFINITY;
		final double[] work = new double[x.length];
		for (int i = 1; i <= 101; ++i) {
			for (int j = 0; j < x.length; ++j) {
				work[j] = x[j] - y_101[i - 1][j];
			}
			final double[] rxmy = mmult(R, work);
			double sum = 0.0;
			for (int j = 0; j < x.length; ++j) {
				sum += rxmy[j] * C_101[i - 1][j] * rxmy[j];
			}
			final double wi = i == 1 ? 10.0 : 1.1 + 8.0 * (i - 2) / 99.0;
			final double valuei = wi * Math.exp(-0.5 * sum / x.length);
			res = Math.max(res, valuei);
		}
		final double result = Math.pow(tosz(10 - res), 2) + fpen(x);
		return result;
	}

	public static final double gallagher21(final double[] x) {
		double res = Double.NEGATIVE_INFINITY;
		final double[] work = new double[x.length];
		for (int i = 1; i <= 21; ++i) {
			for (int j = 0; j < x.length; ++j) {
				work[j] = x[j] - y_21[i - 1][j];
			}
			final double[] rxmy = mmult(R, work);
			double sum = 0.0;
			for (int j = 0; j < x.length; ++j) {
				sum += rxmy[j] * C_21[i - 1][j] * rxmy[j];
			}
			final double wi = i == 1 ? 10.0 : 1.1 + 8.0 * (i - 2) / 19.0;
			final double valuei = wi * Math.exp(-0.5 * sum / x.length);
			res = Math.max(res, valuei);
		}
		final double result = Math.pow(tosz(10 - res), 2) + fpen(x);
		return result;
	}

	public static final double katsuura(final double[] x) {
		final int d = x.length;
		final double[] z = mmult(Q, lambda_mult(2, mmult(R, x)));
		double prod = 1.0;
		for (int i = 1; i <= d; ++i) {
			double sum = 0.0;
			for (int j = 1; j <= 32; ++j) {
				sum += Math.abs(pow2j[j] * z[i - 1] - Math.round(pow2j[j] * z[i - 1])) / pow2j[j];
			}
			prod *= (1 + i * sum);
		}
		final double coeff = 10. / d / d;
		final double result = coeff * Math.pow(prod, 10. / Math.pow(d, 1.2)) - coeff + fpen(x);
		return result;
	}

	public static final double lunacekBiRastrigin(final double[] x) {
		final int d = x.length;
		final double[] xhat = new double[d];
		double[] z = new double[d];
		for (int i = 0; i < d; ++i) {
			final double xopti = 2.5 / 2 * one_pm[i];
			xhat[i] = 2 * Math.signum(xopti) * x[i];
			z[i] = xhat[i] - 2.5;
		}
		z = mmult(Q, lambda_mult(2, mmult(R, z)));
		double ss0 = 0.0;
		double ss1 = 0.0;
		double scos = 0.0;
		for (int i = 0; i < d; ++i) {
			ss0 += (xhat[i] - 2.5) * (xhat[i] - 2.5);
			ss1 += (xhat[i] - lbr_mu1) * (xhat[i] - lbr_mu1);
			scos += Math.cos(2 * Math.PI * z[i]);
		}
		final double result = Math.min(ss0, d + lbr_s * ss1) + 10 * (d - scos) + 10000 * fpen(x);
		return result;
	}

	public static void main(String[] args) {
		final int scales = 7;
		final int n = 20;
		final int m = 1;
		for (int i = 0; i <= scales * m; ++i) {
			final int fevs = (int) Math.pow(10, i / (m * 1.)) * n;
			int passed = 0;
			for (final String key : ALL_FUNCTIONS.keySet()) {
				final Function<double[], Double> func = ALL_FUNCTIONS.get(key);
				double besterr = Double.POSITIVE_INFINITY;
				double bestfit = Double.POSITIVE_INFINITY;
				for (int j = -8; j <= 2; ++j) {
					prepare(n);
					final double[] lb = new double[n];
					final double[] ub = new double[n];
					final double[] guess = new double[n];
					for (int l = 0; l < n; ++l) {
						guess[l] = RAND.nextDouble() * 2 - 1;
						lb[l] = -5.0;
						ub[l] = 5.0;
					}
					BiPopCmaesAlgorithm alg = new BiPopCmaesAlgorithm(-1, 1e-13, 2.0, fevs, 2000, false);
					double[] res = alg.optimize(func, guess).getOptimalPoint();
					if (res != null) {
						double fit = func.apply(res);
						double ferr = Math.abs(fit - 0.0);
						besterr = Math.min(ferr, besterr);
						bestfit = Math.min(bestfit, fit);
						if (ferr <= Math.pow(10, j)) {
							++passed;
						} else {
							// System.out.println(key + " err " + ferr + " > " + Math.pow(10, j));
						}
					} else {
						// System.out.println(key + " err " + "null" + " > " + Math.pow(10, j));
					}
				}
			}
			double passpercent = (double) passed / (24 * 11);
			System.out.println(Math.log10(fevs / n) + "\t" + passpercent);
		}
	}
}
