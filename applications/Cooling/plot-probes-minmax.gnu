#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,20cm font ',10'

set grid
set key bottom left Left reverse

set output 'probes-minmax.pdf'
set multiplot layout 4,3

# number of probes
N = 3

col='T_%g[K]'
col_s='T_%s[K]'
set title col noenh
p for [i=0:N-1] 'probes-dislocation-3d.txt' u ($1/3600):sprintf(col, i) w l ti sprintf('%g', i),\
'' u ($1/3600):sprintf(col_s, 'min') w l dt 2 lc 'black' noti,\
'' u ($1/3600):sprintf(col_s, 'max') w l dt 2 lc 'black' noti

unset key
col='N_m_%g[m^-2]'
col_s='N_m_%s[m^-2]'
set title col
rep

col='dot_N_m_%g[m^-2s^-1]'
col_s='dot_N_m_%s[m^-2s^-1]'
set title col
rep

# yy and zz
do for [j=1:2] {
    col='stress_'.j.'_%g[Pa]'
    col_s='stress_%s[Pa]'
    set title col
    rep

    col='strain_c_'.j.'_%g[-]'
    col_s='strain_c_%s[-]'
    set title col
    rep

    col='dot_strain_c_'.j.'_%g[s^-1]'
    col_s='dot_strain_c_%s[s^-1]'
    set title col
    rep
}

set xlabel 't, h'

col='tau_eff_%g[Pa]'
col_s='tau_eff_%s[Pa]'
set title col
rep

col='v_%g[ms^-1]'
col_s='v_%s[ms^-1]'
set title col
rep

col='dt[s]'
set title col
p 'probes-dislocation-3d.txt' u ($1/3600):col w l noti
