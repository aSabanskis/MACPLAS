#!/bin/bash

source ./helper.sh

# dim, order, threads
initialize # 2 0

# Temperature
arr_T=(500 600 700)

clean_results
setup_parameters

./create-mesh.sh

for i in "${!arr_T[@]}"; do
    T=${arr_T[$i]}
    id="T$T"
    r="results-$id"
    echo "$id"

    if [[ -f "$r/probes-dislocation-3d.txt" ]]; then
        echo "$id" calculated, skipping
        continue
    fi

    sed -Ei "s|(set Initial temperature *= *).*|\1$T|"   problem.prm
    sed -Ei "s|(set Reference temperature *= *).*|\1$T|" stress.prm

    mkdir -p $r

    ./macplas-bending order "$order" >"$r/log"
    mv probes-dislocation-3d.txt "$r/"
    mv result-dislocation-3d-order$order-*.vtk "$r/"
done

./plot-probes-compare.gnu
