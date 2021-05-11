#!/bin/bash

# Inductor currents, A
arr_I=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140)

for i in "${!arr_I[@]}"; do
    I=${arr_I[$i]}
    r="I$I"

    if [[ -d "$r" ]]; then
        echo "$r" exists, skipping
        continue
    fi

    echo "$r"
    mkdir -p "$r"

    sed -Ei "s|(set Inductor current *= *).*|\1 $I|" problem.prm
    cp -- *.prm "$r"

    ./macplas-inductive-heating > "$r/log"

    mv -- *-3d* "$r"
done
