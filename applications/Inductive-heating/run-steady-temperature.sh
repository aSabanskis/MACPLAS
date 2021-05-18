#!/bin/bash

set -e # exit script on error

./clean-results.sh

cp parameters/*.prm .

sed -Ei "s|(set Temperature only *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 0|" temperature.prm
sed -Ei "s|(set Newton step length *= *).*|\1 0.8|" temperature.prm

z0=0.26685
sed -Ei "s|(set Thermal conductivity *= *).*|\1 25|" temperature.prm
sed -Ei "s|(set Electrical conductivity *= *).*|\1 100*10^(4.247-2924.0/T)|" problem.prm
sed -Ei "s|(set Emissivity *= *).*|\1 0.57|" problem.prm
sed -Ei "s|(set Inductor position *= *).*|\1 $z0|" problem.prm
sed -Ei "s|(set Reference inductor position *= *).*|\1 $z0|" problem.prm
sed -Ei "s|(set Probe coordinates x *= *).*|\1 0.01, 0.01|" problem.prm
sed -Ei "s|(set Probe coordinates z *= *).*|\1 $z0, 0.26145|" problem.prm


# 2D or 3D simulation mode
dim=2

# Finite element order
order=2

# Inductor currents, A
arr_I=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120)

for i in "${!arr_I[@]}"; do
    I=${arr_I[$i]}
    r="T-${dim}d-order$order-I$I"

    if [[ -d "$r" ]]; then
        echo "$r" exists, skipping
        continue
    fi

    echo "$r"
    mkdir -p "$r"

    sed -Ei "s|(set Inductor current *= *).*|\1 $I|" problem.prm
    cp -- *.prm "$r"

    ./macplas-inductive-heating ${dim}d order $order > "$r/log"
    ./process-probes.py
    mv -- *-${dim}d-order$order-t0* probes*.txt "$r"
done

./plot-steady-temperature.gnu
./plot-temperature-current.gnu

r="T-${dim}d-order$order"
mkdir -p "$r"
mv results-*.dat results-*.pdf "$r"
