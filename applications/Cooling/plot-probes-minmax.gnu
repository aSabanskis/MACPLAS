#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',10'

set grid
set key bottom left Left reverse

set output 'probes-minmax.pdf'
set multiplot layout 2,3

# number of probes
N = 3

col='T_%g[K]'
col_s='T_%s[K]'
set title col noenh
p for [i=0:N-1] 'probes-dislocation-3d.txt' u 1:sprintf(col, i) w l ti sprintf('%g', i),\
'' u 1:sprintf(col_s, 'min') w l dt 2 lc 'black' noti,\
'' u 1:sprintf(col_s, 'max') w l dt 2 lc 'black' noti

unset key
col='N_m_%g[m^-2]'
col_s='N_m_%s[m^-2]'
set title col
rep

col='dot_N_m_%g[m^-2s^-1]'
col_s='dot_N_m_%s[m^-2s^-1]'
set title col
rep

col='tau_eff_%g[Pa]'
col_s='tau_eff_%s[Pa]'
set title col
rep

col='strain_c_1_%g[-]'
col_s='strain_c_%s[-]'
set title col
rep

col='dot_strain_c_1_%g[s^-1]'
col_s='dot_strain_c_%s[s^-1]'
set title col
rep
