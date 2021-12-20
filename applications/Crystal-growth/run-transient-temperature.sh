#!/bin/bash

source ./helper.sh

# dim, order, threads
initialize # 2 2 0

r=results-transient
dt=1

if [[ -f $r/$probes ]]
then
    echo "$r/$probes" exists, remove to rerun.
    exit 0
fi

clean_results

mkdir -p "$r"

cp parameters/* .

setup_parameters

sed -Ei "s|(set Output time step *= *).*|\1 300|" problem.prm
sed -Ei "s|(set Temperature only *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 $dt|" temperature.prm

# sed -Ei "s|(d1*= *).*|\1 0|" problem.ini
# sed -Ei "s|(d_function*= *).*|\1 parabola|" problem.ini
./parametric-setup.py

./create-mesh.sh

cp -- *.prm "$r"

./macplas-crystal-growth "$dim"d order "$order" > $r/log
./plot-probes-minmax.gnu

mv -- *-"$dim"d-order"$order"* probes* "$r"
