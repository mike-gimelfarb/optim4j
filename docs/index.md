![center|300px](images/optim4j.png)


Optim4J
-------------------------

*optim4j* is a free and open-source library of subroutines for performing local and global minimization of non-linear functions (some state-of-the-art), written entirely in Java. It consists of:

1. existing subroutines, often written originally in other programming languages (mostly Fortran and C/C++) and provided generously for the benefit of the public, that were translated to Java
2. algorithms implemented from scratch based on existing literature.

**This library is a work in progress (read: beta version). There can still be errors in translation (e.g. from other packages) or errors in code written for this package. Detailed testing of current algorithms still remains to be done, and many additional features and algorithms will be added in the future.**


Download and Installation
-------------------------

There are currently only two ways to install the project: 

1. Download the java archive and incorporate directly into your projects

-   [v0.1.jar](https://github-production-release-asset-2e65be.s3.amazonaws.com/238916392/5262f800-da09-11ea-89b9-537934df890c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20200810%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200810T084346Z&X-Amz-Expires=300&X-Amz-Signature=7262bc65c8ae24c55f22f880a58537ecb42a17c920f89ea2b489d70e9186b64d&X-Amz-SignedHeaders=host&actor_id=35513382&repo_id=238916392&response-content-disposition=attachment%3B filename%3Doptim4j-v01.jar&response-content-type=application%2Foctet-stream)

2. Download the source code and compile it locally on your machine:

-	[v0.1.zip](https://codeload.github.com/mike-gimelfarb/optim4j/zip/v0.1)
-	[v0.1.tar.gz](https://codeload.github.com/mike-gimelfarb/optim4j/tar.gz/v0.1)

Maven support will be provided in the near future.


Licensing
----------------

The project, as a whole, is licensed under the Lesser General Public License, version 2 or later. However, algorithms written entirely by me can be used separately under the MIT license.


Acknowledgements
----------------

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
