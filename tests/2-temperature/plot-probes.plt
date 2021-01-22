#!/usr/bin/env gnuplot

rho = 500.
c_p = 1000.
lambda = 200.
a = lambda / (rho*c_p)

T_1d(x, t) = 1000 * (1 - erf(x/sqrt(4*a*t)))

f = 'probes-temperature-1d.txt'

set terminal pdfcairo
set grid
set key opaque box reverse Left width 1

set output 'result-probes.pdf'

p \
for [i=0:5] f u 1:'T_'.i.'[K]' w l ti 'T_'.i,\
for [i=1:5] T_1d(0.1*i, x) w l lt -1 dt 2 noti
