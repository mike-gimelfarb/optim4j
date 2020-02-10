# min4j
This is a self-contained library of subroutines for performing local and global minimization of non-linear functions (some state-of-the-art), written entirely in Java. It consists of:
1. existing subroutines, often written originally in other programming languages (mostly Fortran and C/C++) and provided generously for the benefit of the public, that were translated to Java
2. algorithms implemented from scratch based on existing literature.

**This library is a work in progress (read: beta version). Detailed testing of current algorithms still remains to be done, and many additional features and algorithms will be added in the future.**

## Requirements and Installation
This package only requires JRE >= 1.8. No dependencies!

## Coverage

The current version supports the optimization algorithms list below. Please note: this list may be updated in the future as new algorithms are introduced:

1. line search methods:
	- Backtracking
	- Constant Step Size
	- Fletcher
	- Hager-Zhang
	- More-Thuente
	- Strong Wolfe Conditions
2. univariate problems:
	- derivative-free methods:
		- Brent (local version)
		- Brent (global version)
		- Calvin
		- Davies-Swann-Campey
		- Fibonacci Search
		- Gaussian Estimation-of-Distribution (Gaussian-EDA)
		- Golden Section Search
		- Modified Piyavskii
	- first-order methods:
		- Cubic Interpolation
		- Modified Secant
3. multivariate:
	- unconstrained and box-constrained problems:
		- derivative-free methods:
			- quadratic approximation methods:
				- BOBYQA
				- NEWUOA
				- UOBYQA
			- CMA-ES methods and variants:
				- Vanilla (CMA-ES)
				- Active (aCMA-ES)
				- Cholesky (cholesky CMA-ES)
				- Limited Memory (LM-CMA-ES)
				- Separable (sep-CMA-ES)
				- Restarts with Increasing Pop. (IPOP, NIPOP...)
				- Restarts with Two Pop. (BIPOP, NBIPOP...)
			- direct search methods:
				- Controlled Random Search (CRS)
				- Dividing Rectangles (DIRECT)
				- Nelder-Mead Simplex
				- Praxis
				- Rosenbrock
			- other evolutionary and swarm-based methods:
				- Adaptive PSO
				- Cooperatively Co-Evolving PSO
				- Competitive Swarm Optimization (CSO)
				- AMALGAM
				- Differential Search
				- ESCH
				- PIKAIA
				- Self-Adaptive Differential Evolution with Neighborhood Search (SaNSDE)
		- first and second-order methods:
			- Conjugate Gradient (CG+)
			- Conjugate Variable Metric (PLIC)
			- Limited-Memory BFGS (LBFGS)
			- Box-Constrained Limited-Memory BFGS (LBFGS-B)
			- Truncated Newton
			- Trust-Region Newton
	- constrained problems:
		- derivative-free methods:
			- Box Complex
			- COBYLA
			- LINCOA
		- first and second-order methods:
			- Shor (SOLVOPT)
			- SQP Variable Metric (PSQP)
			- TOLMIN
	- problems with specific structure:
		- linear programming problems:
			- Revised Simplex
		- least-squares problems:
			- Levenberg-Marquardt
		
	
## Usage
This program is designed to work with Java functions or lambda expressions. Below is a worked example.

1. Define the optimization problem:

```java
// function to optimize: here it is the standard Rosenbrock (aka banana) function
// note: the dimension here is determined by the size of the input
java.util.function.Function<double[], Double> rosenbrock = x -> {
	double y = 0.0;
	for (int i = 0; i < x.length - 1; i++) {
		y += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(x[i] - 1, 2);
	}
	return y;
};

// initial condition (here set to zeros) and set the problem dimension to 20
double[] initial = new double[20];
```

2. Define the optimization algorithm:
```java
import opt.multivariate.unconstrained.order0.cmaes.BiPopCmaesAlgorithm;
    
// use the (very recent) algorithm NBIPOP-aCMA-ES, which performs well on most functions
BiPopCmaesAlgorithm optimizer = new BiPopCmaesAlgorithm(1e-6, 1e-6, 2.0, 1000000, 1000, true);
```
  
3. Run the optimize method, and print out the solution:
  
```java
double[] solution = optimizer.optimize(rosenbrock, initial);

System.out.println("evaluations = " + optimizer.countEvaluations());
System.out.println("solution x = " + java.util.Arrays.toString(solution));
System.out.println("solution y = " + rosenbrock.apply(solution));
```
  
The output of this program is:
  
```
Run	Mode	Run1	Run2	Budget1	Budget2	MaxBudget	Pop	Sigma	F	BestF
0	0	0	0	0	0	92820	12	2.0	3.9866238558561284	3.9866238558561284
0	0	1	0	20185	0	131976	24	1.25	1.2495827942169573E-8	1.2495827942169573E-8
1	1	1	1	20185	10093	10092	12	0.4164128031738537	0.01431248471633458	1.2495827942169573E-8
2	0	2	1	51146	10093	188016	48	0.78125	3.0843111308598474E-8	1.2495827942169573E-8
solution x = [1.0000004569427632, 0.999998642485224, 0.9999998170872698, 1.000000106110834, 0.9999979697844478, 1.0000006289248426, 0.9999991405914844, 1.0000013915830155, 1.000001264139961, 0.9999993514560533, 1.000001270682207, 1.0000002499343588, 1.0000016271980299, 1.000001364348365, 1.0000005575345872, 0.9999982218665536, 1.0000002596222186, 1.000004037335844, 1.000009106509799, 1.000017430772891]
solution y = 1.2495827942169573E-8
evaluations = 79000
```
  
## Credits

## License
The code, packaged as a single library, is licensed under the GNU Lesser General Public License (version 2 or later). However, some subroutines can be used independently under more flexible licenses (typically MIT or BSD license). The license type or license header are listed at the top of each code file.
