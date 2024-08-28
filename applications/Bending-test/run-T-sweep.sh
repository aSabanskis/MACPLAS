#!/bin/bash

source ./helper.sh

# order, threads
initialize # 2 0

# Temperature
arr_T=(400 500 600 700 800 900)

# Pressure
p=2e5

clean_results
setup_parameters

./create-mesh.sh

for i in "${!arr_T[@]}"; do
    T=${arr_T[$i]}
    id="p$p-T$T"
    r="results-$id"
    echo "$id"

    if [[ -f "$r/probes-dislocation-3d.txt" ]]; then
        echo "$id" calculated, skipping
        continue
    fi

    sed -Ei "s|(set Initial temperature *= *).*|\1$T|"   problem.prm
    sed -Ei "s|(set Reference temperature *= *).*|\1$T|" stress.prm

    sed -Ei "s|(set Pressure *= *).*|\1$p|" problem.prm

    mkdir -p $r

    ./macplas-bending order "$order" >"$r/log"

    ./plot-probes-minmax.gnu

    mv probes-dislocation-3d.txt probes-minmax.pdf result-dislocation-3d-order$order-*.vtk "$r/"
done

sed -Ei "s|(p *= *).*|\1'$p'|" plot-probes-compare.gnu
./plot-probes-compare.gnu
