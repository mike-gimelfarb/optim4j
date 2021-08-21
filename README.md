[![Build Status](https://travis-ci.com/mike-gimelfarb/optim4j.svg?branch=master)](https://travis-ci.com/mike-gimelfarb/optim4j)
[![Documentation Status](https://readthedocs.org/projects/optim4j/badge/?version=latest)](https://optim4j.readthedocs.io/en/latest/?badge=latest)
[![License: LGPL v2](https://img.shields.io/badge/License-LGPL%20v2-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![HitCount](http://hits.dwyl.com/mike-gimelfarb/optim4j.svg)](http://hits.dwyl.com/mike-gimelfarb/optim4j)

| :exclamation:  This package is now discontinued and will no longer be updated. An up-to-date C++ version with many bugs fixed and several algorithms re-implemented (correctly this time) and python wrappers (coming soon) can be found here: https://github.com/mike-gimelfarb/cocoa. |
|-----------------------------------------|

# optim4j
This is a self-contained library of algorithms for performing local and global optimization of functions written in Java. Main features:
1. support for univariate problems up to multivariate problems with tens of thousands of variables, and constrained problems
1. many algorithms are re-implementations of recently published algorithms (e.g. adaptive PSO, firefly) and can be seen as state-of-the-art
2. some algorithms are translations of professional implementations of classical algorithms (e.g. LBFGS)
2. flexible licensing (full project under LGPL, but some algorithms can be used under the MIT license)

## Requirements and Installation
This package only requires JRE >= 1.8. No dependencies!

## Support

The project documentation site is found here: https://optim4j.readthedocs.io/en/latest/.
 
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
