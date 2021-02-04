#!/bin/bash

r=results-calibration

mkdir -p $r

# Reproduction of model calibration results from http://dx.doi.org/10.1016/j.jcrysgro.2016.06.007
# Temperature, Young's modulus, Peierls potential, strain hardening factor
arr_T=(1173  1373 1573)
arr_E=(8.4e9 23e9 1.3e9)
arr_Q=(2.08  2.21 2.3)
arr_D=(38.8518 34.6948 29.8847)

cp data-ref/*.prm .
sed -i "s/set Max time.*/set Max time = 1000/g" dislocation.prm

for i in "${!arr_T[@]}"; do
    T=${arr_T[$i]}
    E=${arr_E[$i]}
    Q=${arr_Q[$i]}
    D=${arr_D[$i]}

    sed -i "s/set Initial temperature.*/set Initial temperature = $T/g" problem.prm
    sed -i "s/set Reference temperature.*/set Reference temperature = $T/g" stress.prm
    sed -i "s/set Young's modulus.*/set Young's modulus = $E/g" stress.prm
    sed -i "s/set Peierls potential.*/set Peierls potential = $Q/g" dislocation.prm
    sed -i "s/set Strain hardening factor.*/set Strain hardening factor = $D/g" dislocation.prm

    for s in "Linearized N_m"; do
        sed -i "s/set Time scheme.*/set Time scheme = $s/g" dislocation.prm
        for t in 0.1; do
            sed -i "s/set Time step.*/set Time step = $t/g" dislocation.prm

            id="T$T $s dt$t"
            id="${id// /_}"
            echo "$id"

            if [[ -f "$r/probes-$id.txt" ]]; then
                echo "$id" exists, skipping
                continue
            fi

            ./macplas-test-4 >"$r/log-$id"
            mv probes-dislocation-3d.txt "$r/probes-$id.txt"
        done
    done
done

./plot-probes-calibration.gnu
