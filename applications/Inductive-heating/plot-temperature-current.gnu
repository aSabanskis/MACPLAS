#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',8'

set grid
set key top left Left reverse

dim=2
order=2

filename = sprintf('results-temperature-%gd.dat', dim)

#set xrange []
set xlabel 'I, A'
set yrange [300:1700]
set ylabel 'Temperature, K'

set output sprintf('results-T(I)-%gd-order%g.pdf', dim, order)

set macros
style = "w lp pt 6"

p \
filename u 'I[A]':'T_min[K]' @style ti 'T min', \
filename u 'I[A]':'T_max[K]' @style ti 'T max', \
filename u 'I[A]':'T_1[K]' @style ti 'T probe'
