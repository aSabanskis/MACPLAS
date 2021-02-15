#!/usr/bin/env gnuplot

set term pdfcairo size 15cm,10cm font ',10'

set grid
set key bottom left Left reverse

set output 'probes.pdf'
set multiplot layout 2,3

# number of probes
N = 3

col='T_%g[K]'
set title col noenh
p for [i=0:N-1] 'probes-dislocation-3d.txt' u 1:sprintf(col, i) w lp pt 6 ps 0.4 ti sprintf('%g', i)

unset key
col='N_m_%g[m^-2]'
set title col
rep

col='dot_N_m_%g[m^-2s^-1]'
set title col
rep

col='tau_eff_%g[Pa]'
set title col
rep

col='strain_c_2_%g[-]'
set title col
rep

col='dot_strain_c_2_%g[s^-1]'
set title col
rep
