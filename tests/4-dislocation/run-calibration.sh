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

for i in "${!arr_T[@]}"; do
    T=${arr_T[$i]}
    E=${arr_E[$i]}
    Q=${arr_Q[$i]}
    D=${arr_D[$i]}

    sed -Ei "s/(set Initial temperature *= *).*/\1$T/"     problem.prm
    sed -Ei "s/(set Reference temperature *= *).*/\1$T/"   stress.prm
    sed -Ei "s/(set Young's modulus *= *).*/\1$E/"         stress.prm
    sed -Ei "s/(set Peierls potential *= *).*/\1$Q/"       dislocation.prm
    sed -Ei "s/(set Strain hardening factor *= *).*/\1$D/" dislocation.prm

    id="T$T"
    echo "$id"

    if [[ -f "$r/probes-$id.txt" ]]; then
        echo "$id" exists, skipping
        continue
    fi

    ./macplas-test-4 >"$r/log-$id"
    mv probes-dislocation-3d.txt "$r/probes-$id.txt"
done

./plot-probes-calibration.gnu
