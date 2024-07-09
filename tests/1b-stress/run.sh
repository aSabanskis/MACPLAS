#!/bin/bash

for p in 1 2 3 ; do
    ./macplas-test-1b order $p BCf load 1.56e7 0 0
    ./macplas-test-1b order $p BCf load 0 0 1
    ./macplas-test-1b order $p BCf load 0 0 1 boundary 4
done

./plot-x.plt
