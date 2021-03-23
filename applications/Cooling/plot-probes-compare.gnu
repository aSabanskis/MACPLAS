#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,13cm font ',10'

set grid
set key bottom left Left reverse width -2

f0_temp  = '1-elastic/probes-temperature-3d.txt'
f1_elast = '1-elastic/probes-dislocation-3d.txt'
f1_plast = '2-plastic/probes-dislocation-3d.txt'
f2_elast = 'data-ref/PP43e_neu.rpt.tsv'
f2_plast = 'data-ref/PPP43_neu.rpt.tsv'

set datafile separator '\t'

format_time(t) = t/3600.
set xrange [0:54.5]

# number of probes
N = 3
array positions[N] = ['bottom', 'center', 'top']

set output 'probes-compare.pdf'
set multiplot layout 2,2

cM = 1e6
T0 = 273
col='T_%g[K]'
col_ref='TEMP PI %g: BLOCK_G1'
set title 'Temperature, K'
p \
for [i=0:N-1] f0_temp  u (format_time($1)):(column(sprintf(col, i))) w l ti sprintf('T %s', positions[i+1]), \
f0_temp u (format_time($1)):'T_bot[K]' w l dt 2 lc black ti 'T bot (BC)', \
f0_temp u (format_time($1)):'T_top[K]' w l dt 3 lc black ti 'T top (BC)', \
for [i=0:N-1] f2_elast u (format_time($1)):(column(sprintf(col_ref, i))+T0) w l lc i+1 dt 2+i ti sprintf('T %s (Dadzis2016)', positions[i+1])

unset key
col='stress_1_%g[Pa]'
col_ref='S:S22 PI %g: BLOCK_G'
set title 'σ_{yy} elastic, MPa'
p \
for [i=0:N-1] f1_elast u (format_time($1)):(column(sprintf(col, i))/cM) w l, \
for [i=0:N-1] f2_elast u (format_time($1)):(column(sprintf(col_ref, i))) w l lc i+1 dt 2+i

set xlabel 'Time, h'
col='N_m_%g[m^-2]'
col_ref='SDV1 PI %g: BLOCK_G1'
set title 'Dislocation density, m^{-2}'
p \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i))) w l, \
for [i=0:N-1] f2_plast u (format_time($1)):(column(sprintf(col_ref, i))*1e4) w l lc i+1 dt 2+i

col='stress_1_%g[Pa]'
col_ref='S:S22 PI %g: BLOCK_G'
set title 'σ_{yy} plastic, MPa'
p \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i))/cM) w l, \
for [i=0:N-1] f2_plast u (format_time($1)):(column(sprintf(col_ref, i))) w l lc i+1 dt 2+i

unset multiplot
