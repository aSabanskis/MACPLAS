#!/bin/bash

r=results-simple

mkdir -p $r

rm *.prm
./macplas-test-4 init

sed -i "s/set Time scheme.*/set Time scheme = Linearized N_m/g" dislocation.prm
sed -i "s/set Time step.*/set Time step = 0.1/g" dislocation.prm
sed -i "s/set Max time  .*/set Max time  = 1000/g" dislocation.prm
sed -i "s/set Max time step.*/set Max time step = 1/g" dislocation.prm
sed -i "s/set Min time step.*/set Min time step = 0.001/g" dislocation.prm
sed -i "s/set Max relative time step increase.*/set Max relative time step increase = 0.1/g" dislocation.prm
sed -i "s/set Max dstrain_c.*/set Max dstrain_c = 1e-6/g" dislocation.prm
sed -i "s/set Max relative dN_m.*/set Max relative dN_m = 0.1/g" dislocation.prm
sed -i "s/set Number of threads.*/set Number of threads = 1/g" stress.prm
sed -i "s/set Log convergence final.*/set Log convergence final = false/g" stress.prm

for T in 700 800 900 1000 1100 1200 1300 1400 1500 1600; do
    sed -i "s/set Initial temperature.*/set Initial temperature = $T/g" problem.prm
    sed -i "s/set Reference temperature.*/set Reference temperature = $T/g" stress.prm

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
