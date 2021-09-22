#!/bin/bash

./macplas-test-2 1D BC1
mv probes-temperature-1d.txt probes-temperature-1d-BC1.txt

./macplas-test-2 1D BC2
mv probes-temperature-1d.txt probes-temperature-1d-BC2.txt

./macplas-test-2 1D BC3
mv probes-temperature-1d.txt probes-temperature-1d-BC3.txt


./macplas-test-2 1D BC1b steady vol_heat_source 4e5
mv result-1d-order2-x.dat result-1d-q4e5-x.dat

./macplas-test-2 1D BC1b steady vol_heat_source -4e5
mv result-1d-order2-x.dat result-1d-q-4e5-x.dat


./macplas-test-2 1D BC1b steady velocity 1e-3
mv result-1d-order2-x.dat result-1d-V1e-3-x.dat

./macplas-test-2 1D BC1b steady velocity -1e-3
mv result-1d-order2-x.dat result-1d-V-1e-3-x.dat


./macplas-test-2 2D BC1b steady

./plot-probes.plt
