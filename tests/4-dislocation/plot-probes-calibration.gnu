#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,25cm font ',10'

set grid
set key top left Left reverse width -2

dir = "results-calibration"
array T[3] = [1173, 1373, 1573]

set datafile separator '\t'

set output dir.'/probes-all.pdf'
set multiplot layout 4,2

col='N_m_0[m^-2]'
col_ref='N'
set title col noenh
p \
for[k=1:|T|] sprintf('%s/probes-T%g.txt', dir, T[k]) \
u 1:(abs(column(col))) w l ti sprintf('T = %g K', T[k]), \
for[k=1:|T|] sprintf('data-ref/calibration_%g.tsv', T[k]) \
u 1:(column(col_ref)) w l lc black dt k+1 ti sprintf('T = %g K (Dadzis2016)', T[k])

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
set xlabel 't, s'
rep

col='dt[s]'
col_ref=''
set title col
rep

unset multiplot


set output dir .'/probes-stress-strain.pdf'
set term pdfcairo rounded size 15cm,10cm
set key top left
unset title
set xlabel 'Strain, %'
set ylabel 'Stress, MPa'

p \
for[k=1:|T|] sprintf('%s/probes-T%g.txt', dir, T[k]) \
u (abs(column('strain_total[-]')*100)):(abs(column('stress_0_0[Pa]')/1e6)) w l ti sprintf('T = %g K', T[k]), \
for[k=1:|T|] sprintf('data-ref/calibration_%g.tsv', T[k]) \
u (column('eps_tot')*100):(column('sigma')/1e6) w l lc black dt k+1 ti sprintf('T = %g K (Dadzis2016)', T[k])
