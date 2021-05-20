#!/bin/bash

set -e # exit script on error


# 2D or 3D simulation mode
dim=2

# Finite element order
order=2


probes=probes-temperature-${dim}d.txt

if [[ -f $probes ]]
then
    echo $probes exists, remove to rerun.
    exit 0
fi

./clean-results.sh

cp parameters/*.prm .

sed -Ei "s|(set Start from steady temperature *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Temperature only *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 10|" temperature.prm
sed -Ei "s|(set Newton step length *= *).*|\1 0.8|" temperature.prm

./macplas-inductive-heating ${dim}d order $order > log
