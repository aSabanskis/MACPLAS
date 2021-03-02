#!/bin/bash

if [[ -f probes-dislocation-3d.txt ]]
then
    echo probes-dislocation-3d.txt exists, remove to rerun.
    exit 0
fi

./macplas-cooling init

tmax=198000  # 55 h
nthreads=0

sed -Ei "s/(set Number of threads *= *).*/\1$nthreads/" stress.prm
sed -Ei "s/(set Number of threads *= *).*/\1$nthreads/" temperature.prm
sed -Ei "s/(set Log convergence final *= *).*/\1 false/" stress.prm
sed -Ei "s/(set Log convergence final *= *).*/\1 false/" temperature.prm
sed -Ei "s/(set Max time *= *).*/\1$tmax/" temperature.prm
sed -Ei "s/(set Max time *= *).*/\1$tmax/" dislocation.prm
sed -Ei "s/(set Initial dislocation density *= *).*/\1 1e7/" dislocation.prm
sed -Ei "s/(set Time scheme *= *).*/\1 Linearized N_m/" dislocation.prm
sed -Ei "s/(set Time step *= *).*/\1 0.1/" dislocation.prm
sed -Ei "s/(set Max time step *= *).*/\1 60/" dislocation.prm
sed -Ei "s/(set Min time step *= *).*/\1 0.1/" dislocation.prm
sed -Ei "s/(set Max relative time step increase *= *).*/\1 0.1/" dislocation.prm
sed -Ei "s/(set Max dstrain_c *= *).*/\1 1e-6/" dislocation.prm
sed -Ei "s/(set Max relative dN_m *= *).*/\1 0.1/" dislocation.prm
sed -Ei "s/(set Output frequency *= *).*/\1 1000/" problem.prm

# http://dx.doi.org/10.1016/j.jcrysgro.2016.06.007
sed -Ei "s/(set Thermal conductivity *= *).*/\1 217.873, -0.398349, 0.000276322, -6.48418e-08/" temperature.prm
sed -Ei "s/(set Bottom heat transfer coefficient *= *).*/\1 2000/" problem.prm
sed -Ei "s/(set Top heat transfer coefficient *= *).*/\1 2000/" problem.prm
sed -Ei "s/(set Bottom reference temperature *= *).*/\1 t/3600<10 ? 1685-t/3600*200/10 : t/3600<47 ? 1485-(t/3600-10)*812/37 : 673-(t/3600-47)*400/8/" problem.prm
sed -Ei "s/(set Top reference temperature *= *).*/\1 t/3600<10 ? 1685-t/3600*12/10 : t/3600<40 ? 1673-(t/3600-10)*660/30 : t/3600<47 ? 1013-(t/3600-40)*340/7 : 673-(t/3600-47)*400/8/" problem.prm

./macplas-cooling

./plot-probes-minmax.gnu
