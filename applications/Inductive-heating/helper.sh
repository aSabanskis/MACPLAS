#!/bin/bash

set -e # exit script on error


clean_prm(){
    for f in *.prm
    do
        echo "$f"
        sed -Ei "/^#/d" "$f"
        sed -Ei "/ *#/d" "$f"
        sed -Ei "/^$/d" "$f"
    done
}

clean_results(){
    rm -df -- *d-order* probes*.txt results-*.dat || true
}

initialize(){
    # 2D or 3D simulation mode
    dim=2
    if [[ "$#" -ge 1 ]]; then
        dim=$1
    fi

    # Finite element order
    order=2
    if [[ "$#" -ge 2 ]]; then
        order=$2
    fi

    probes=probes-temperature-${dim}d.txt
}

setup_parameters(){
    sed -Ei "s|(set Use LF EM field *= *).*|\1 false|" problem.prm

    # custom code can be added
}
