#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,25cm font ',10'

set grid
#set xrange[0:1]
set key top left Left reverse width -2

dir = "results-calibration"
array T[3] = [1173, 1373, 1573]

case = "Linearized_N_m"
dt = 0.1
set datafile separator '\t'

set output dir.'/probes-compare.pdf'
set multiplot layout 4,2

col='N_m_0[m^-2]'
col_ref='N'
set title col noenh
p \
for[k=1:|T|] sprintf('%s/probes-T%g_%s_dt%g.txt', dir, T[k], case, dt) \
u 1:(abs(column(col))) w l ti sprintf('T = %g K', T[k]), \
for[k=1:|T|] sprintf('data-ref/calibration_%g.tsv', T[k]) \
u 1:(column(col_ref)) w l lt k dt 2 ti sprintf('T = %g K (Dadzis2016)', T[k])

unset key
col='dot_N_m_0[m^-2s^-1]'
col_ref='dN/dt'
set title col
rep

col='v_0[ms^-1]'
col_ref='v, m/s'
set title col
rep

col='stress_0_0[Pa]'
col_ref='sigma'
set title col
rep

col='strain_c_0_0[-]'
col_ref='eps_c'
set title col
rep

col='dot_strain_c_0_0[s^-1]'
col_ref='deps_c/dt'
set title col
rep

col='tau_eff_0[Pa]'
col_ref='taueff'
set title col
rep
