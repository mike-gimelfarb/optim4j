---
# Optimizing Continuous Functions of One Variable - A First Example
---

Portfolio Optimization with Two Stocks
-------------------------

Suppose that we are able to invest a fixed sum of money in two stocks A and B. Based on historical data, the mean and standard deviation of their annualized return is estimated as follows:

<table>
  <tr>
    <td></td>
    <th scope="col">Mean</th>
    <th scope="col">Std.</th>
  </tr>
  <tr>
    <th scope="row">Stock A</th>
    <td>10%</td>
    <td>10%</td>
  </tr>
  <tr>
    <th scope="row">Stock B</th>
    <td>17%</td>
    <td>25%</td>
  </tr>
</table>

Furthermore, the returns of the two stocks are assumed to be independent, and the risk-free rate is estimated at 1%. 

Letting w be the weight of the portfolio allocated to stock A, we can compute the mean return of the portfolio as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu(w)=0.10*w + 0.17*(1-w)" title="Portfolio Mean" />

and the variance of the portfolio as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma^2(w)=0.10^2*w^2 + 0.17^2*(1-w)^2" title="Portfolio Var" />.

We will avoid short selling. Therefore, we seek to find the w in [0, 1] that obtains the best risk-adjusted return based on historic data. Here, we shall measure the risk-adjusted return by dividing the return of the portfolio (above the risk-free rate) by its standard deviation, which is also known as the Sharpe ratio:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;SR(w)=\frac{\mu(w)-r_f}{\sqrt{\sigma^2(w)}}=\frac{0.10*w + 0.17*(1-w)-0.01}{\sqrt{0.10^2*w^2 + 0.17^2*(1-w)^2}}" title="Portfolio Sharpe" />.

The algorithms in optim4j are designed for minimizing functions, so our optimization problem becomes

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\min_{w\in[0,1]}-SR(w)" title="Portfolio Objective" />.

Solving the Optimization Problem with optim4j
-------------------------

1. Import the necessary packages

```java
import java.util.function.Function;

import opt.univariate.UnivariateOptimizerSolution;
import opt.univariate.order0.BrentAlgorithm;
```

2. Define the cost function to optimize

```java
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
```

3. To optimize the Sharpe ratio, we will use the Brent algorithm, which does not require derivatives and works well for reasonably smooth functions without too many local optima, and subject to box constraints

```java
BrentAlgorithm optimizer = new BrentAlgorithm(1e-8, 1e-14, 999);
UnivariateOptimizerSolution solution = optimizer.optimize(sharpe_objective, 0.0, 1.0);
```

4. Report and interpret the final result

```java
System.out.println(solution);
System.out.println("Final sharpe ratio = " + -sharpe_objective.apply(solution.getOptimalPoint()));
```

The output of this program should be

x*: 0.7785467052466715<br>
calls to f: 13<br>
calls to df/dx: 0<br>
converged: true<br>
Final sharpe ratio = 1.1043550153822816


The final Sharpe ratio of 1.10 could be obtained by investing about 78% of the wealth into stock A, and the rest into stock B. 