#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',8'

set grid
set key top left Left reverse

dim=2
order=2

filename = sprintf('probes-inductor-temperature-%gd.txt', dim)

set xlabel 'Time, s'
set ylabel 'Temperature, K'

set output sprintf('results-T(t)-%gd-order%g.pdf', dim, order)

p \
for [i=0:4] filename u 't[s]':'T_'.i.'[K]' w l ti 'T '.i, \
filename u 't[s]':'T_max[K]' w l dt 2 ti 'T max'
