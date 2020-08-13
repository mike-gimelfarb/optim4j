[![Build Status](https://travis-ci.com/mike-gimelfarb/optim4j.svg?branch=master)](https://travis-ci.com/mike-gimelfarb/optim4j)
[![Documentation Status](https://readthedocs.org/projects/optim4j/badge/?version=latest)](https://optim4j.readthedocs.io/en/latest/?badge=latest)
[![License: LGPL v2](https://img.shields.io/badge/License-LGPL%20v2-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![HitCount](http://hits.dwyl.com/mike-gimelfarb/optim4j.svg)](http://hits.dwyl.com/mike-gimelfarb/optim4j)



# optim4j
This is a self-contained library of subroutines for performing local and global minimization of non-linear functions (some state-of-the-art), written entirely in Java. It consists of:
1. existing subroutines, often written originally in other programming languages (mostly Fortran and C/C++) and provided generously for the benefit of the public, that were translated to Java
2. algorithms implemented from scratch based on existing literature.

**This library is a work in progress (read: beta version). There can still be errors in translation (e.g. from other packages) or errors in code written for this package. Detailed testing of current algorithms still remains to be done, and many additional features and algorithms will be added in the future.**

## Requirements and Installation
This package only requires JRE >= 1.8. No dependencies!

## Support

The project documentation site is found here: https://optim4j.readthedocs.io/en/latest/.
	
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
 
## License
The code, packaged as a single library, is licensed under the GNU Lesser General Public License (version 2 or later). However, some subroutines can be used independently under more flexible licenses (typically MIT or BSD license). The license type or license header are listed at the top of each code file.

## Credits

We are grateful to all the authors who have provided optimization libraries under open-source licenses:

- This product includes software developed by the University of Chicago, as Operator of Argonne National Laboratory.

- This product includes translations of the Fortran subroutines (PLIC, PSQP) by Ladislav Luksan licensed acccording to LGPL. Used by permission http://www.cs.cas.cz/~luksan/subroutines.html.

- This product includes translations of the subroutine for Hager-Zhang line search from the Julia package https://github.com/JuliaNLSolvers/LineSearches.jl.

- This product includes translations of Fortran software written by Alfred Morris at the Naval Surface Warfare Center, translated by Alan Miller at CSIRO Mathematical & Information Sciences https://jblevins.org/mirror/amiller/smplx.f90.

- This product includes translations of Fortran subroutines by Professor M. J. D. Powell, University of Cambridge https://zhangzk.net/software.html.

- This product inclues translations of subroutines from the SolvOpt package written by Alexei V. Kuntsevich and Franz Kappel https://imsc.uni-graz.at/kuntsevich/solvopt/index.html.

- This product includes translations of subroutines (CRS, ESCH) from the NLOpt package written by Steven G. Johnson and Carlos Henrique da Silva Santos http://github.com/stevengj/nlopt.

- This product includes a translation of subroutines in the DIRECTL package written by Joerg M. Gablonsky https://ctk.math.ncsu.edu/matlab_darts.html.

- This product includes a translation of the Fortran 77 version of Nelder-Mead Simplex written by R. O'Neill and translated into Fortran 90 by John Burkardt https://people.sc.fsu.edu/~jburkardt/cpp_src/asa047/asa047.html.

- This product includes translations of Fortran subroutines written by Richard P. Brent http://www.netlib.org/opt/ 

- This product includes translations of the Differential Search algorithm written by Pinar Civicioglu https://www.mathworks.com/matlabcentral/fileexchange/43390-differential-search-algorithm-a-modernized-particle-swarm-optimization-algorithm

- This product includes a translation of the PIKAIA subroutines written by the High Altitude Observatory https://www.hao.ucar.edu/modeling/pikaia/pikaia.php#sec4

- This product includes translations of software L-BFGS-B written by Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales http://users.iems.northwestern.edu/~nocedal/lbfgsb.html.

- This product includes translations of software CG+ written by Guanghui Liu, Jorge Nocedal and Richard Waltz http://users.iems.northwestern.edu/~nocedal/CG+.html.

- This product includes a translation of the Truncated Newton algorithm written by Stephen G. Nash at George Mason University https://www.netlib.org/opt/.

- This product includes translations of software from the Pytron package https://github.com/fabianp/pytron.

- This product includes software from the SLATEC library http://www.netlib.org/slatec/index.html.

- This product includes code from the JAMA matrix package https://math.nist.gov/javanumerics/jama/.

- This product includes translations of software from the LINPACK project developed by Jack Dongarra, Jim Bunch, Cleve Moler and Pete Stewart http://www.netlib.org/linpack/.
