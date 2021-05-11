#!/bin/bash

r=results-simple

mkdir -p $r

rm *.prm
./macplas-test-4 init

sed -Ei "s|(set Time scheme *= *).*|\1 Linearized N_m|" dislocation.prm
sed -Ei "s|(set Time step *= *).*|\1 0.1|" dislocation.prm
sed -Ei "s|(set Max time *= *).*|\1 1000|" dislocation.prm
sed -Ei "s|(set Max time step *= *).*|\1 1|" dislocation.prm
sed -Ei "s|(set Min time step *= *).*|\1 0.001|" dislocation.prm
sed -Ei "s|(set Max relative time step increase *= *).*|\1 0.1|" dislocation.prm
sed -Ei "s|(set Max dstrain_c *= *).*|\1 1e-6|" dislocation.prm
sed -Ei "s|(set Max relative dN_m *= *).*|\1 0.1|" dislocation.prm
sed -Ei "s|(set Number of threads *= *).*|\1 1|" stress.prm
sed -Ei "s|(set Log convergence final *= *).*|\1 false|" stress.prm

for T in 700 800 900 1000 1100 1200 1300 1400 1500 1600; do
    sed -Ei "s|(set Initial temperature *= *).*|\1 $T|" problem.prm
    sed -Ei "s|(set Reference temperature *= *).*|\1 $T|" stress.prm

    id="T$T"
    echo "$id"

    if [[ -f "$r/probes-$id.txt" ]]; then
        echo "$id" exists, skipping
        continue
    fi

    ./macplas-test-4 >"$r/log-$id"
    mv probes-dislocation-3d.txt "$r/probes-$id.txt"
done

./plot-probes-simple.gnu
