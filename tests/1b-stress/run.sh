#!/bin/bash

for p in 1 2 3 ; do
    ./macplas-test-1b 3D order $p BCf force 1.56e7 0 0
done

./plot-x.plt
