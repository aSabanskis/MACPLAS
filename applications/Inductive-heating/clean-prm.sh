#!/bin/bash

set -e # exit script on error

for f in *.prm
do
    echo $f
    sed -Ei "/^#/d" $f
    sed -Ei "/^$/d" $f
done
