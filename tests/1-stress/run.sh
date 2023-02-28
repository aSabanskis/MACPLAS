#!/bin/bash

for d in 2D 3D ; do
  for p in 1 2 3 ; do
    ./macplas-test-1 $d order $p
  done
done

./plot-x.plt
