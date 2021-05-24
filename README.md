# MACPLAS
MAcroscopic Crystal PLAsticity Simulator

Uses [deal.II](https://www.dealii.org/) C++ finite element library to model thermal stresses and dislocation density dynamics during single crystal growth.
Also includes the temperature solver and utilities for interpolation of the external boundary conditions.

*The development is still ongoing, so beware that the functionality of the library, including public member functions, could change in the future.*

# Installation
## Library
The MACPLAS library is designed as header-only, hence no precompilation is needed. Simply include the necessary headers from the ```include``` directory and compile as usual. MACPLAS relies on the [deal.II](https://www.dealii.org/) library which should be installed beforehand.
Optionally, the header files can be installed from the top-level directory as
```
cmake -DCMAKE_INSTALL_PREFIX=/my/location/include .
make install
```

## Examples
Several examples are included in the ```applications``` and ```tests``` directories. To compile, go to the desired subdirectory and type
```
cmake .
make release
```
To compile the debug version of the program, type ```make debug``` or simply ```make```.


# Documentation
To generate documentation in HTML and LaTeX formats, execute the command ```doxygen doxygen.conf``` in the ```doc``` directory or ```cmake --build . --target doc``` from the top-level directory.


# License
[GNU Lesser General Public License (LGPL) v2.1 or later](LICENSE)


# Contributors
 - Andrejs Sabanskis (University of Latvia, post-doctoral project "Effect of thermal stresses and growth conditions on the point defect and dislocation distributions in semiconductor crystals" No. 1.1.1.2/VIAA/2/18/280, financed by the European Regional Development Fund)
