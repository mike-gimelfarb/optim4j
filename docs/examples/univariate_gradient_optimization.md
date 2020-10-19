---
# Optimizing Continuous Functions of One Variable - Using Derivative Information
---

Estimating the Parameters of a Weibull Distribution
-------------------------

The Weibull distribution is often used in reliability engineering to model the lifetime of machines. It is a two parameter family of density functions, which are defined on all positive x as 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x; \lambda, k)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}" title="Weibull" />.

Given data assumed to follow a Weibull distribution with unknown parameters, our goal will be to estimate its parameters using maximum likelihood analysis. To illustrate, suppose the times of failure of 12 identical light bulbs are given in the following table:

<table>
  <tr>
    <td></td>
    <th scope="col">1</th>
    <th scope="col">2</th>
    <th scope="col">3</th>
    <th scope="col">4</th>
    <th scope="col">5</th>
    <th scope="col">6</th>
    <th scope="col">7</th>
    <th scope="col">8</th>
    <th scope="col">9</th>
    <th scope="col">10</th>
    <th scope="col">11</th>
    <th scope="col">12</th>
  </tr>
  <tr>
    <th scope="row">Lifetimes</th>
    <td>509</td>
    <td>660</td>
    <td>386</td>
    <td>753</td>
    <td>811</td>
    <td>613</td>
    <td>848</td>
    <td>725</td>
    <td>315</td>
    <td>872</td>
    <td>487</td>
    <td>512</td>
  </tr>
</table>

The likelihood function can be written as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\prod_{i=1}^{12} f(x_i; \lambda, k) = \left(\frac{k}{\lambda}\right)^{12} \left(\frac{x_1 x_2 \dots x_{12}}{\lambda^{12}}\right)^{k-1} e^{-\sum_{i=1}^{12} (x_i/\lambda)^k}" title="likelihood" />.

Since the natural log is a monotone transformation, it does not change the optimal solution, so taking the natural log of the likelihood function yields the log-likelihood function

<img src="https://latex.codecogs.com/svg.latex?\Large&space;LL(\lambda, k) = 12\log k - 12 k \log\lambda +(k-1) \sum_{i=1}^{12}\log x_i - \frac{1}{\lambda^k} \sum_{i=1}^{12} x_i^k" title="log_likelihood" />.

We now proceed to compute the maximum of the log-likelihood with respect to the parameters lambda and k. First, treating k as constant and maximizing with respect to lambda yields a closed form expression for lambda in terms of k:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda^k=\frac{1}{12}\sum_{i=1}^{12}x_i^k" title="solution_for_lambda" />.

Since lambda can be determined analytically in terms of k, this reduces to the problem of optimizing the log-likelihood for k. To do this, we first substitute the expression for lambda back into the log-likelihood, and after simplifying, we have:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;LL(k) = 12\log k - 12 \log\left(\frac{1}{12}\sum_{i=1}^{12}x_i^k\right) +(k-1) \sum_{i=1}^{12}\log x_i - 12" title="log_likelihood" />.

This expression cannot be optimized analytically. In this tutorial, we will be using a gradient-based optimization algorithm to compute k. Since Java does not automatically compute gradients by default, this will have to be provided to the algorithm. Specifically, the gradient of the log-likelihood is

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{d LL(k)}{d k} = \frac{12}{k} - \frac{12 \sum_{i=1}^{12}x_i^k \log x_i}{\sum_{i=1}^{12}x_i^k} + \sum_{i=1}^{12}\log x_i" title="log_likelihood" />.

Solving the Optimization Problem with optim4j
-------------------------

1. Import the necessary packages

```java
import java.util.function.Function;

import opt.univariate.UnivariateOptimizerSolution;
import opt.univariate.order1.CubicInterpolationAlgorithm;
```

2. Define the negative log-likelihood function to optimize and its gradient

```java
// negative log-likelihood function
Function<Double, Double> f = k -> {
	double sum_pow_k = 0.0;
	double sum_log = 0.0;
	for (final double x : data) {
		sum_pow_k += Math.pow(x, k);
		sum_log += Math.log(x);
	}
	return -(n * Math.log(k) - n * Math.log(sum_pow_k / n) + (k - 1.0) * sum_log - 12.0);
};

// derivative of the negative log-likelihood function
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
```

3. We minimize the negative log-likelihood function using its gradient information, using an algorithm based on cubic interpolation and an initial arbitrary guess of 1:

```java
CubicInterpolationAlgorithm optimizer = new CubicInterpolationAlgorithm(1e-8, 1e-14, 999);
UnivariateOptimizerSolution solution = optimizer.optimize(f, df, 1.0);
```

4. We now print the result of this optimization to find that it converged to within the desired tolerances for errors

x*: 4.141937624372837<br>
calls to f: 25<br>
calls to df/dx: 21<br>
converged: true

The following code computes lambda as a function of k:

```java
// compute the lambda parameter
double k = solution.getOptimalPoint();
double sum_pow_k = 0.0;
for (final double x : data) {
	sum_pow_k += Math.pow(x, k);
}
double lambda = Math.pow(sum_pow_k / n, 1.0 / k);

// print the final parameters of the Weibull
System.out.println("Final parameters are k=" + k + ", lambda=" + lambda);
```

We find that the parameters are lambda=689.807 and k=4.142 to three decimal places.