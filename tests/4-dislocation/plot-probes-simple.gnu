#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,25cm font ',10'

set grid
set key top left Left reverse width -2

dir = "results-simple"
array T[8] = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

set datafile separator '\t'

set output dir.'/probes-all.pdf'
set multiplot layout 4,2

col='N_m_0[m^-2]'
set title col noenh
p \
for[k=1:|T|] sprintf('%s/probes-T%g.txt', dir, T[k]) \
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

col='dt[s]'
set title col
rep

unset multiplot


set output dir .'/probes-stress-strain.pdf'
set term pdfcairo rounded size 15cm,10cm
set key top center
unset title
set xlabel 'Strain, %'
set ylabel 'Stress, MPa'

set xrange [0:0.5]
set yrange [0:50]

p \
for[k=1:|T|] sprintf('%s/probes-T%g.txt', dir, T[k]) \
u (abs(column('strain_total[-]')*100)):(abs(column('stress_0_0[Pa]')/1e6)) w l ti sprintf('T = %g K', T[k])
