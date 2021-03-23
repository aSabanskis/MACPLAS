#!/bin/bash

set -e # exit script on error

calculate () {
    r=$1
    echo Calculating $r

    if [[ ! -f $r/probes-dislocation-3d.txt ]]
    then
        mkdir -p $r
        ./macplas-cooling > $r/log
        ./plot-probes-minmax.gnu
        cp -- *.prm *.vtk $r
        mv probes* $r
    else
        echo $r/probes-dislocation-3d.txt exists, remove to rerun.
    fi
}

if [[ -f probes-dislocation-3d.txt ]]
then
    echo probes-dislocation-3d.txt exists, remove to rerun.
    exit 0
fi

./macplas-cooling init

tmax=196200  # 54.5 h
nthreads=0

sed -Ei "s/(set Number of threads *= *).*/\1$nthreads/" stress.prm
sed -Ei "s/(set Number of threads *= *).*/\1$nthreads/" temperature.prm
sed -Ei "s/(set Log convergence final *= *).*/\1 false/" stress.prm
sed -Ei "s/(set Log convergence final *= *).*/\1 false/" temperature.prm
sed -Ei "s/(set Max time *= *).*/\1$tmax/" temperature.prm
sed -Ei "s/(set Max time *= *).*/\1$tmax/" dislocation.prm
sed -Ei "s/(set Initial dislocation density *= *).*/\1 0/" dislocation.prm
sed -Ei "s/(set Time scheme *= *).*/\1 Linearized N_m/" dislocation.prm
sed -Ei "s/(set Time step *= *).*/\1 120/" dislocation.prm
sed -Ei "s/(set Time step *= *).*/\1 120/" temperature.prm
sed -Ei "s/(set Max time step *= *).*/\1 120/" dislocation.prm

sed -Ei "s/(set Max relative time step increase *= *).*/\1 0.1/" dislocation.prm
sed -Ei "s/(set Max dstrain_c *= *).*/\1 1e-6/" dislocation.prm
sed -Ei "s/(set Max relative dN_m *= *).*/\1 0.1/" dislocation.prm
sed -Ei "s/(set Max v*dt *= *).*/\1 5e-4/" dislocation.prm
sed -Ei "s/(set Output frequency *= *).*/\1 60/" problem.prm
sed -Ei "s/(set Temperature only *= *).*/\1 false/" problem.prm

# http://dx.doi.org/10.1016/j.jcrysgro.2016.06.007
sed -Ei "s/(set Density *= *).*/\1 2337.77-0.025044*T-3.75768e-06*T^2/" temperature.prm
sed -Ei "s|(set Specific heat capacity *= *).*|\1 1046.43-0.0426946*T+3.07177e-05*T^2-109539/T|" temperature.prm
sed -Ei "s/(set Thermal conductivity *= *).*/\1 217.873, -0.398349, 0.000276322, -6.48418e-08/" temperature.prm

sed -Ei "s/(set Bottom heat transfer coefficient *= *).*/\1 2000/" problem.prm
sed -Ei "s/(set Top heat transfer coefficient *= *).*/\1 2000/" problem.prm
sed -Ei "s|(set Top reference temperature *= *).*|\1 t<36000 ? 1683 : t<144000 ? 1683-(t-36000)*660/108000 : t<170000 ? 1023-(t-144000)*350/26000 : 673-(t-170000)*370/26000|" problem.prm
sed -Ei "s|(set Bottom reference temperature *= *).*|\1 t<36000 ? 1683-t*210/36000 : t<144000 ? 1473-(t-36000)*650/108000 : t<170000 ? 823-(t-144000)*150/26000 : 673-(t-170000)*370/26000|" problem.prm
sed -Ei "s|(set Initial temperature *= *).*|\1 1683|" problem.prm

sed -Ei "s/(set Reference temperature *= *).*/\1 1683/" stress.prm
sed -Ei "s/(set Poisson's ratio *= *).*/\1 0.25/" stress.prm
sed -Ei "s/(set Young's modulus *= *).*/\1 1.7e11-2.771e4*T^2/" stress.prm
sed -Ei "s/(set Thermal expansion coefficient *= *).*/\1 3.4795e-06*(1-exp(-0.00326426*(T+57.2677)))+3.69166e-10*T/" stress.prm

sed -Ei "s/(set Average Schmid factor *= *).*/\1 0.56984471569/" dislocation.prm
sed -Ei "s/(set Average Taylor factor *= *).*/\1 1.7782388291/" dislocation.prm
sed -Ei "s/(set Burgers vector *= *).*/\1 3.83e-10/" dislocation.prm
sed -Ei "s/(set Material constant k_0 *= *).*/\1 8.58e-4/" dislocation.prm
sed -Ei "s|(set Peierls potential *= *).*|\1 2.185+0.1*atan((T-1347.5)/100)|" dislocation.prm
sed -Ei "s/(set Strain hardening factor *= *).*/\1 2.0*0.4*(1.7e11-2.771e4*T^2)*3.83e-10/" dislocation.prm

calculate 1-elastic


sed -Ei "s/(set Initial dislocation density *= *).*/\1 1e7/" dislocation.prm
sed -Ei "s/(set Time step *= *).*/\1 30/" dislocation.prm
sed -Ei "s/(set Min time step *= *).*/\1 10/" dislocation.prm
sed -Ei "s/(set Max time step *= *).*/\1 60/" dislocation.prm

calculate 2-plastic


./plot-probes-compare.gnu
