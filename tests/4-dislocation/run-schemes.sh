#!/bin/bash

mkdir -p results

for T in 1000 1100 1200 1300 1400 1500 1600; do
    sed -Ei "s|(set Initial temperature *= *).*|\1 $T|" problem.prm
    sed -Ei "s|(set Reference temperature *= *).*|\1 $T|" stress.prm
    for s in "Euler" "Midpoint" "Linearized N_m" "Linearized N_m midpoint" "Implicit"; do
        sed -Ei "s|(set Time scheme *= *).*|\1 $s|" dislocation.prm
        for t in 5 2 1 0.5 0.2 0.1 0.05 0.02; do
            sed -Ei "s|(set  Time step *= *).*|\1 $t|" dislocation.prm

            id="T$T $s dt$t"
            id="${id// /_}"
            echo "$id"

            if [[ -f "results/probes-$id.txt" ]]; then
                echo "$id" exists, skipping
                continue
            fi

            ./macplas-test-4 >"results/log-$id"
            mv probes-dislocation-3d.txt "results/probes-$id.txt"
        done
    done
done
