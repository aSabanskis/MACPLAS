#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',8'

set grid
set key top left Left reverse

dim=2
order=2

filename_crys = sprintf('probes-temperature-%gd.txt', dim)
filename_ind = sprintf('probes-inductor-temperature-%gd.txt', dim)

set xlabel 'Time, s'
set ylabel 'Temperature, K'

set output sprintf('results-T(t)-%gd-order%g.pdf', dim, order)

p \
for [i=0:1] filename_crys u (column('t[s]')>0?column('t[s]'):NaN):'T_'.i.'[K]' w l ti 'T crys.'.i, \
for [i=0:4] filename_ind u 't[s]':'T_'.i.'[K]' w l ti 'T ind.'.i, \
filename_ind u 't[s]':'T_max[K]' w l dt 2 ti 'T max'
