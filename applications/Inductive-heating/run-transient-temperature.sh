#!/bin/bash

set -e # exit script on error


# 2D or 3D simulation mode
dim=2

# Finite element order
order=2


r=results-transient
probes=probes-temperature-${dim}d.txt

if [[ -f $r/$probes ]]
then
    echo $r/$probes exists, remove to rerun.
    exit 0
fi

./clean-results.sh

cp parameters/*.prm .

mkdir -p "$r"

sed -Ei "s|(set Start from steady temperature *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Temperature only *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 10|" temperature.prm
sed -Ei "s|(set Max Newton iterations *= *).*|\1 8|" temperature.prm
sed -Ei "s|(set Newton step length *= *).*|\1 0.8|" temperature.prm

sed -Ei "s|(set Thermal conductivity *= *).*|\1 25|" temperature.prm
sed -Ei "s|(set Electrical conductivity *= *).*|\1 100*10^(4.247-2924.0/T)|" problem.prm
sed -Ei "s|(set Emissivity *= *).*|\1 0.57|" problem.prm
sed -Ei "s|(set Custom probes x relative to inductor *= *).*|\1 0.01, 0.01, 0.01, 0.01, 0.01|" problem.prm
sed -Ei "s|(set Custom probes z relative to inductor *= *).*|\1 -0.00825, -0.0054, -0.00255, 0, 0.0054|" problem.prm

cp -- *.prm "$r"

./macplas-inductive-heating ${dim}d order $order > $r/log
./plot-transient-temperature-time.gnu

mv -- *-${dim}d-order$order* probes*.txt "$r"
