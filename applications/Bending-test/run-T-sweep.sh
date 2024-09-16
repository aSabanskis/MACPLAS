#!/bin/bash

source ./helper.sh

# order, threads
initialize # 2 0

# Temperature
arr_T=(400 500 600 700 800 900)

# Force
F=5

clean_results
setup_parameters

./create-mesh.sh

for i in "${!arr_T[@]}"; do
    T=${arr_T[$i]}
    id="F$F-T$T"
    r="results-$id"
    echo "$id"

    if [[ -f "$r/probes-dislocation-3d.txt" ]]; then
        echo "$id" calculated, skipping
        continue
    fi

    sed -Ei "s|(set Initial temperature *= *).*|\1$T|"   problem.prm
    sed -Ei "s|(set Reference temperature *= *).*|\1$T|" stress.prm

    sed -Ei "s|(set Max force *= *).*|\1$F|" problem.prm

    mkdir -p $r

    ./macplas-bending order "$order" >"$r/log"

    ./plot-probes-minmax.gnu

    cp *prm "$r/"
    mv probes-dislocation-3d.txt probes-minmax.pdf result-dislocation-3d-order$order-*.vtk "$r/"
done

sed -Ei "s|(F *= *).*|\1'$F'|" plot-probes-compare.gnu
./plot-probes-compare.gnu
