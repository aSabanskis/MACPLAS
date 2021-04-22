#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,18cm font ',10'

set grid
set key bottom left Left reverse width -5

f0_temp  = '1-elastic/probes-temperature-3d.txt'
f1_elast = '1-elastic/probes-dislocation-3d.txt'
f1_plast = '2-plastic/probes-dislocation-3d.txt'
f2_elast = 'data-ref/PP43e_neu.rpt.tsv'
f2_plast = 'data-ref/PPP43_neu.rpt.tsv'

set datafile separator '\t'

T_bot(t) = t<36000 ? 1683-210*(t-0)/36000 : t<144000 ? 1473-650*(t-36000)/108000 : t<170000 ? 823-150*(t-144000)/26000 : t<196000 ? 673-370*(t-170000)/26000 : 303
T_top(t) = t<36000 ? 1683+0*(t-0)/36000 : t<144000 ? 1683-660*(t-36000)/108000 : t<170000 ? 1023-350*(t-144000)/26000 : t<196000 ? 673-370*(t-170000)/26000 : 303
w_bot = 0.46
T_mid(t) = w_bot*T_bot(t) + (1-w_bot)*T_top(t)

format_time(t) = t/3600.
set xrange [0:54.5]

# number of probes
N = 3
array positions[N] = ['bottom', 'center', 'top']

set macros
calc_style = 'w l lw 0.5+0.5*i'
ref_style = 'w p pt 6 ps 0.3'

set output 'probes-compare.pdf'
set multiplot layout 3,2

cM = 1e6
T0 = 273
col='T_%g[K]'
col_ref='TEMP PI %g: BLOCK_G1'
set title 'Temperature, K'
p \
for [i=0:N-1] f0_temp u (format_time($1)):(column(sprintf(col, i+0))) @calc_style lc 1 ti sprintf('T %s 1', positions[1]), \
for [i=0:N-1] f0_temp u (format_time($1)):(column(sprintf(col, i+3))) @calc_style lc 2 ti sprintf('T %s 2', positions[2]), \
for [i=0:N-1] f0_temp u (format_time($1)):(column(sprintf(col, i+6))) @calc_style lc 3 ti sprintf('T %s 3', positions[3]), \
f0_temp u (format_time($1)):'T_bot[K]' w l dt 2 lc black ti 'T bot (BC)', \
f0_temp u (format_time($1)):'T_top[K]' w l dt 3 lc black ti 'T top (BC)', \
f0_temp u (format_time($1)):(T_mid($1)) w l dt 4 lc black ti 'T mid (artificial)', \
for [i=0:N-1] f2_elast u (format_time($1)):(column(sprintf(col_ref, i))+T0) @ref_style lc i+1 ti sprintf('T %s ref.', positions[i+1])

unset key
set title 'dT, K'
p \
for [i=0:N-1] f0_temp  u (format_time($1)):(column(sprintf(col, i+0))-T_bot($1)) @calc_style lc 1, \
for [i=0:N-1] f0_temp  u (format_time($1)):(column(sprintf(col, i+3))-T_mid($1)) @calc_style lc 2, \
for [i=0:N-1] f0_temp  u (format_time($1)):(column(sprintf(col, i+6))-T_top($1)) @calc_style lc 3, \
f2_elast u (format_time($1)):(column(sprintf(col_ref, 0))+T0-T_bot($1)) @ref_style lc 1, \
f2_elast u (format_time($1)):(column(sprintf(col_ref, 1))+T0-T_mid($1)) @ref_style lc 2, \
f2_elast u (format_time($1)):(column(sprintf(col_ref, 2))+T0-T_top($1)) @ref_style lc 3

col='stress_1_%g[Pa]'
col_ref='S:S22 PI %g: BLOCK_G'
set title 'σ_{yy} elastic, MPa'
p \
for [i=0:N-1] f1_elast u (format_time($1)):(column(sprintf(col, i+0))/cM) @calc_style lc 1, \
for [i=0:N-1] f1_elast u (format_time($1)):(column(sprintf(col, i+3))/cM) @calc_style lc 2, \
for [i=0:N-1] f1_elast u (format_time($1)):(column(sprintf(col, i+6))/cM) @calc_style lc 3, \
for [i=0:N-1] f2_elast u (format_time($1)):(column(sprintf(col_ref, i))) @ref_style lc i+1

set xlabel 'Time, h'
col='N_m_%g[m^-2]'
col_ref='SDV1 PI %g: BLOCK_G1'
set title 'Dislocation density, m^{-2}'
p \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i+0))) @calc_style lc 1, \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i+3))) @calc_style lc 2, \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i+6))) @calc_style lc 3, \
for [i=0:N-1] f2_plast u (format_time($1)):(column(sprintf(col_ref, i))*1e4) @ref_style lc i+1

col='stress_1_%g[Pa]'
col_ref='S:S22 PI %g: BLOCK_G'
set title 'σ_{yy} plastic, MPa'
p \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i+0))/cM) @calc_style lc 1, \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i+3))/cM) @calc_style lc 2, \
for [i=0:N-1] f1_plast u (format_time($1)):(column(sprintf(col, i+6))/cM) @calc_style lc 3, \
for [i=0:N-1] f2_plast u (format_time($1)):(column(sprintf(col_ref, i))) @ref_style lc i+1

unset multiplot
