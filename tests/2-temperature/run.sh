#!/bin/bash

./macplas-test-2 1D BC1
mv probes-temperature-1d.txt probes-temperature-1d-BC1.txt

./macplas-test-2 1D BC2
mv probes-temperature-1d.txt probes-temperature-1d-BC2.txt

./macplas-test-2 1D BC3
mv probes-temperature-1d.txt probes-temperature-1d-BC3.txt

./macplas-test-2 2D BC1b steady

./plot-probes.plt
