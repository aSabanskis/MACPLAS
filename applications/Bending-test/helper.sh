#!/bin/bash

clean_prm(){
    for f in *.prm
    do
        echo "$f"
        sed -Ei "/^#/d" "$f"
        sed -Ei "s/#.*//" "$f"
        sed -Ei "/^$/d" "$f"
    done
}

clean_results(){
    rm -df -- *d-order* probes*.txt results-*.dat || true
}

initialize(){
    # Finite element order
    order=2
    if [[ "$#" -ge 1 ]]; then
        order=$1
    fi

    # Number of threads (0: automatic)
    threads=0
    if [[ "$#" -ge 2 ]]; then
        threads=$2
    fi
}

setup_parameters(){
    sed -Ei "s|(set Number of threads *= *).*|\1$threads|" stress.prm

    # custom code can be added
    # sed -Ei "s|(set Pressure *= *).*|\1 8e5|" problem.prm
    # sed -Ei "s|(set Pressure ramp *= *).*|\1 10|" problem.prm
}
