#!/bin/bash

source ./helper.sh

# dim, order, threads
initialize # 2 2 0

r=results-transient
dt=0.2

if [[ -f $r/$probes ]]
then
    echo "$r/$probes" exists, remove to rerun.
    exit 0
fi

clean_results

mkdir -p "$r"

cp parameters/* .

setup_parameters

sed -Ei "s|(set Field change relaxation factor *= *).*|\1 0|" problem.prm
sed -Ei "s|(set Field change relaxation factor at boundary *= *).*|\1 0|" problem.prm
sed -Ei "s|(set Field relaxation factor *= *).*|\1 0|" problem.prm
sed -Ei "s|(set Field relaxation factor at boundary *= *).*|\1 0|" problem.prm
sed -Ei "s|(set Fix interface fields *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Reset interface fields *= *).*|\1 false|" problem.prm
sed -Ei "s|(set Interpolation method *= *).*|\1 advection|" problem.prm
sed -Ei "s|(set Stabilization multiplier *= *).*|\1 0|" advection.prm
sed -Ei "s|(set Linear solver type *= *).*|\1 fgmres|" advection.prm
sed -Ei "s|(set Linear solver iterations *= *).*|\1 100|" advection.prm
sed -Ei "s|(set Time stepping theta *= *).*|\1 1|" advection.prm
sed -Ei "s|(set Courant number *= *).*|\1 0|" advection.prm
sed -Ei "s|(set Limit field minmax *= *).*|\1 true|" advection.prm

sed -Ei "s|(set Output time step *= *).*|\1 300|" problem.prm
sed -Ei "s|(set Temperature only *= *).*|\1 false|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 $dt|" dislocation.prm
sed -Ei "s|(set Time step *= *).*|\1 $dt|" temperature.prm
sed -Ei "s|(set Max time step *= *).*|\1 $dt|" dislocation.prm
sed -Ei "s|(set Min time step *= *).*|\1 $dt|" dislocation.prm

sed -Ei "s|(set Initial dislocation density *= *).*|\1 1e4|" dislocation.prm

sed -Ei "s|(set Max dstrain_c *= *).*|\1 1e-6|" dislocation.prm
sed -Ei "s|(set Max relative dN_m *= *).*|\1 0.2|" dislocation.prm
sed -Ei "s|(set Max relative dtau_eff *= *).*|\1 0.2|" dislocation.prm
sed -Ei "s|(set Max relative time step increase *= *).*|\1 0.2|" dislocation.prm
sed -Ei "s|(set Max v\*dt *= *).*|\1 5e-5|" dislocation.prm

sed -Ei "s|(set Method *= *).*|\1 global|" stress.prm

# sed -Ei "s|(d1*= *).*|\1 0|" problem.ini
# sed -Ei "s|(d_function*= *).*|\1 parabola|" problem.ini
./parametric-setup.py

./create-mesh.sh

cp -- *.prm "$r"

./macplas-crystal-growth "$dim"d order "$order" > $r/log
./plot-probes-minmax.gnu

mv -- *-"$dim"d-order"$order"* probes* "$r"
