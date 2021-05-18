#!/bin/bash

set -e # exit script on error

for dim in 2 3; do
    geo="mesh-${dim}d.geo"

    if [[ ! -f "$geo" ]]; then
        echo "$geo" does not exist, skipping
        continue
    fi

    echo "$geo"
    gmsh -"$dim" -format msh2 "$geo"
done
