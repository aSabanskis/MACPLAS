#!/bin/bash

set -e # exit script on error

cp parameters/*.prm .

sed -Ei "s|(set Temperature only *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 0|" temperature.prm
sed -Ei "s|(set Newton step length *= *).*|\1 0.8|" temperature.prm

sed -Ei "s|(set Electrical conductivity *= *).*|\1 100*10^(4.247-2924.0/T)|" problem.prm
sed -Ei "s|(set Emissivity *= *).*|\1 0.57|" problem.prm
sed -Ei "s|(set Inductor position *= *).*|\1 0|" problem.prm


# 2D or 3D simulation mode
dim=2

# Finite element order
order=2

# Inductor currents, A
arr_I=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140)

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

    mv -- *-${dim}d-order$order-t0* "$r"
done

./plot-steady-temperature.gnu
