#!/bin/bash

set -e # exit script on error

dim=2

geo="mesh-${dim}d.geo"

if [[ ! -f "$geo" ]]; then
    echo "$geo" does not exist, skipping
else
    echo "$geo"
    gmsh -"$dim" -format msh2 "$geo"
fi
