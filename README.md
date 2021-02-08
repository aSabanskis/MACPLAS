# MACPLAS
MAcroscopic Crystal PLAsticity Simulator

Uses [deal.II](https://www.dealii.org/) C++ finite element library to model thermal stresses and dislocation density dynamics during single crystal growth.
Also includes the temperature solver and utilities for interpolation of the external boundary conditions.


# Installation
## Library
The MACPLAS library is designed as header-only, hence no precompilation is needed. Simply include the necessary headers from the ```include``` directory and compile as usual. MACPLAS relies on the [deal.II](https://www.dealii.org/) library which should be installed beforehand.

## Examples
Several examples are included in the ```applications``` and ```tests``` directories. To compile, go to the desired subdirectory and type
```
cmake .
make release
```
To compile the debug version of the program, type ```make debug``` or simply ```make```.


# Documentation
To generate documentation in HTML and LaTeX formats, execute the command ```doxygen doxygen.conf``` in the ```doc``` directory.


# License
[GNU Lesser General Public License (LGPL) v2.1 or later](LICENSE)


# Contributors
 - Andrejs Sabanskis (University of Latvia, post-doctoral project "Effect of thermal stresses and growth conditions on the point defect and dislocation distributions in semiconductor crystals" No. 1.1.1.2/VIAA/2/18/280, financed by the European Regional Development Fund)
