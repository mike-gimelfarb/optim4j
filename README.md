# min4j
This is a self-contained library of subroutines for performing local and global minimization of non-linear functions (some state-of-the-art), written entirely in Java. It consists of:
1. existing subroutines, often written originally in other programming languages (mostly Fortran and C/C++) and provided generously for the benefit of the public, that were translated to Java
2. algorithms implemented from scratch based on existing literature.

**This library is a work in progress (read: beta version). Detailed testing of current algorithms still remains to be done, and many additional features and algorithms will be added in the future.**

## Requirements and Installation
This package only requires JRE >= 1.8. That's it!

## Usage
This program is designed to work with Java functions or lambda expressions. Below is a worked example.

  1. Define the optimization problem:

  ```
    // function to optimize: here it is the standard Rosenbrock (aka banana) function
    // note: the dimension here is determined by the size of the input
    java.util.function.Function<double[], Double> rosenbrock = x -> {
      double y = 0.0;
      for (int i = 0; i < x.length - 1; i++) {
        y += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(x[i] - 1, 2);
      }
      return y;
    };

    // initial condition (here set to zeros)
    // note: here we set the dimension of the problem to 20
    double[] initial = new double[20];
   ```

2. Define the optimization algorithm:
  
  ```
    import opt.multivariate.unconstrained.order0.cmaes.BiPopCmaesAlgorithm;
    
    // use the (very recent) algorithm NBIPOP-aCMA-ES, which performs well on most functions
    BiPopCmaesAlgorithm optimizer = new BiPopCmaesAlgorithm(0.0, 1e-12, 2.0, 100000, 1000, false);
  ```
  
  3. Run the optimize method, and print out the solution:
  
  ```
    double[] solution = optimizer.optimize(rosenbrock, initial);
    
    System.out.println("solution x = " + java.util.Arrays.toString(solution));
    System.out.println("solution y = " + rosenbrock.apply(solution));
  ```
  
  The output of this program is:
  
  ```
    solution x = [0.9999999988227944, 0.9999999994579266, 0.9999999997732237, 1.0000000000086318, 0.9999999996160766,      0.9999999999191883, 0.9999999994717175, 0.999999999194096, 0.9999999994144408, 0.9999999991521973, 0.999999999774935, 0.9999999999304536, 0.9999999995479101, 0.9999999996878903, 0.9999999993741902, 0.9999999997143544, 0.9999999991338988, 0.9999999978151639, 0.9999999979244827, 0.9999999958230025]
    solution y = 1.561116349654454E-15.
   ```
  
## Credits

## License
The code, packaged as a single library, is licensed under the GNU Lesser General Public License (version 2 or later). However, some subroutines can be used independently under more flexible licenses (typically MIT or BSD license). The license type or license header are listed at the top of each code file.
