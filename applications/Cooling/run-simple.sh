#!/bin/bash

set -e # exit script on error

calculate () {
    r=$1
    echo Calculating $r

    probes=probes-temperature-3d.txt

    if [[ ! -f $r/$probes ]]
    then
        mkdir -p $r
        ./macplas-cooling order $order > $r/log
        ./plot-probes-minmax.gnu
        cp -- *.prm $r
        mv *.vtk probes* $r
    else
        echo $r/$probes exists, remove to rerun.
    fi
}

if [[ -f $probes ]]
then
    echo $probes exists, remove to rerun.
    exit 0
fi

./macplas-cooling init

tmax=196200  # 54.5 h
nthreads=0
order=2

sed -Ei "s/(set Lx *= *).*/\1 0.84/" problem.prm
sed -Ei "s/(set Ly *= *).*/\1 0.84/" problem.prm
sed -Ei "s/(set Lz *= *).*/\1 0.40/" problem.prm

sed -Ei "s/(set Probe coordinates z *= *).*/\1 -0.2, -0.1975, -0.195, 0, 0.0025, 0.005, 0.195, 0.1975, 0.2/" problem.prm

sed -Ei "s/(set Nx *= *).*/\1 21/" problem.prm
sed -Ei "s/(set Ny *= *).*/\1 21/" problem.prm
sed -Ei "s/(set Nz *= *).*/\1 10/" problem.prm

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
sed -Ei "s|(set Density *= *).*|\1 2337.77-0.025044*T-3.75768e-06*T^2|" temperature.prm
sed -Ei "s|(set Specific heat capacity *= *).*|\1 1046.43-0.0426946*T+3.07177e-05*T^2-109539/T|" temperature.prm
sed -Ei "s|(set Thermal conductivity *= *).*|\1 T<250 ? 195 : T<300 ? 195-39*(T-250)/50 : T<400 ? 156-51*(T-300)/100 : T<500 ? 105-25*(T-400)/100 : T<600 ? 80-16*(T-500)/100 : T<700 ? 64-12*(T-600)/100 : T<800 ? 52-9*(T-700)/100 : T<900 ? 43-7.4*(T-800)/100 : T<1000 ? 35.6-4.6*(T-900)/100 : T<1100 ? 31-3*(T-1000)/100 : T<1200 ? 28-1.9*(T-1100)/100 : T<1300 ? 26.1-1.3*(T-1200)/100 : T<1400 ? 24.8-1.1*(T-1300)/100 : T<1500 ? 23.7-1*(T-1400)/100 : T<1600 ? 22.7-0.8*(T-1500)/100 : T<1681 ? 21.9-0.3*(T-1600)/81 : 21.6|" temperature.prm
sed -Ei "s/(set Thermal conductivity derivative *= *).*/\1 -0.398349 +2*0.000276322*T -3*6.48418e-08*T^2/" temperature.prm

sed -Ei "s/(set Bottom heat transfer coefficient *= *).*/\1 2000/" problem.prm
sed -Ei "s/(set Top heat transfer coefficient *= *).*/\1 2000/" problem.prm
sed -Ei "s|(set Top reference temperature *= *).*|\1 t<36000 ? 1683+0*(t-0)/36000 : t<144000 ? 1683-660*(t-36000)/108000 : t<170000 ? 1023-350*(t-144000)/26000 : t<196000 ? 673-370*(t-170000)/26000 : 303|" problem.prm
sed -Ei "s|(set Bottom reference temperature *= *).*|\1 t<36000 ? 1683-210*(t-0)/36000 : t<144000 ? 1473-650*(t-36000)/108000 : t<170000 ? 823-150*(t-144000)/26000 : t<196000 ? 673-370*(t-170000)/26000 : 303|" problem.prm
sed -Ei "s|(set Initial temperature *= *).*|\1 1683|" problem.prm

sed -Ei "s/(set Reference temperature *= *).*/\1 1683/" stress.prm
sed -Ei "s/(set Poisson's ratio *= *).*/\1 0.25/" stress.prm
sed -Ei "s/(set Young's modulus *= *).*/\1 1.7e11-2.771e4*T^2/" stress.prm
sed -Ei "s|(set Thermal expansion coefficient *= *).*|\1 3.725e-6*(1-exp(-5.88e-3*(T-124)))+5.548e-10*T|" stress.prm

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
