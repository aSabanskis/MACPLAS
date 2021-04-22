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
sed -Ei "s|(set Density *= *).*|\1 T<280 ? 2330 : T<300 ? 2330+0*(T-280)/20 : T<400 ? 2330-3*(T-300)/100 : T<500 ? 2327-2*(T-400)/100 : T<600 ? 2325-3*(T-500)/100 : T<700 ? 2322-4*(T-600)/100 : T<800 ? 2318-3*(T-700)/100 : T<900 ? 2315-3*(T-800)/100 : T<1000 ? 2312-3*(T-900)/100 : T<1100 ? 2309-3*(T-1000)/100 : T<1200 ? 2306-4*(T-1100)/100 : T<1300 ? 2302-3*(T-1200)/100 : T<1400 ? 2299-4*(T-1300)/100 : T<1500 ? 2295-3*(T-1400)/100 : T<1600 ? 2292-4*(T-1500)/100 : T<1685 ? 2288-3*(T-1600)/85 : 2285|" temperature.prm
sed -Ei "s|(set Specific heat capacity *= *).*|\1 T<300 ? 671 : T<400 ? 671+87*(T-300)/100 : T<500 ? 758+61*(T-400)/100 : T<600 ? 819+31*(T-500)/100 : T<700 ? 850+25*(T-600)/100 : T<800 ? 875+18*(T-700)/100 : T<900 ? 893+14*(T-800)/100 : T<1000 ? 907+18*(T-900)/100 : T<1100 ? 925+12*(T-1000)/100 : T<1200 ? 937+12*(T-1100)/100 : T<1300 ? 949+10*(T-1200)/100 : T<1400 ? 959+11*(T-1300)/100 : T<1500 ? 970+11*(T-1400)/100 : T<1600 ? 981+9*(T-1500)/100 : T<1687 ? 990+3*(T-1600)/87 : 993|" temperature.prm
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
sed -Ei "s|(set Thermal expansion coefficient *= *).*|\1 T<280 ? 2.432e-06 : T<300 ? 2.432e-06+6.8e-08*(T-280)/20 : T<400 ? 2.5e-06+3.42e-07*(T-300)/100 : T<500 ? 2.842e-06+2.61e-07*(T-400)/100 : T<600 ? 3.103e-06+1.91e-07*(T-500)/100 : T<700 ? 3.294e-06+1.49e-07*(T-600)/100 : T<800 ? 3.443e-06+1.21e-07*(T-700)/100 : T<900 ? 3.564e-06+9.7e-08*(T-800)/100 : T<1000 ? 3.661e-06+7.7e-08*(T-900)/100 : T<1100 ? 3.738e-06+6.7e-08*(T-1000)/100 : T<1200 ? 3.805e-06+5.9e-08*(T-1100)/100 : T<1300 ? 3.864e-06+5.3e-08*(T-1200)/100 : T<1400 ? 3.917e-06+4.9e-08*(T-1300)/100 : T<1500 ? 3.966e-06+4.6e-08*(T-1400)/100 : T<1600 ? 4.012e-06+4.3e-08*(T-1500)/100 : T<1684 ? 4.055e-06+3.5e-08*(T-1600)/84 : T<1688 ? 4.09e-06+1e-09*(T-1684)/4 : 4.091e-06|" stress.prm

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
