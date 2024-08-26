#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,25cm font ',10'

set grid
set key top left Left reverse width -2

array T = [500, 600, 700]

set datafile separator '\t'

set output 'probes-compare.pdf'
set multiplot layout 4,2

col='N_m_0[m^-2]'
set title col noenh
p \
for[k=1:|T|] sprintf('results-T%g/probes-dislocation-3d.txt', T[k]) \
u 1:(abs(column(col))) w l ti sprintf('T = %g K', T[k])

unset key
col='dot_N_m_0[m^-2s^-1]'
set title col
rep

col='v_0[ms^-1]'
set title col
rep

col='stress_0_0[Pa]'
set title col
rep

col='strain_c_0_0[-]'
set title col
rep

col='dot_strain_c_0_0[s^-1]'
set title col
rep

col='tau_eff_0[Pa]'
set title col
set xlabel 't, s'
rep

col='displacement_2_0[m]'
set title col
rep

unset multiplot
