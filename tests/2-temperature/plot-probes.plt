#!/usr/bin/env gnuplot

rho = 500.
c_p = 1000.
lambda = 200.
a = lambda / (rho*c_p)

T_1d(x, t) = 1000 * (1 - erf(x/sqrt(4*a*t)))

f = 'probes-temperature-1d.txt'

set terminal pdfcairo rounded
set grid
set key opaque box reverse Left width 1

set output 'result-probes-1d.pdf'

p \
for [i=0:5] f u 1:'T_'.i.'[K]' w l lw 2 ti 'T_'.i,\
for [i=1:5] T_1d(0.1*i, x) w l lt -1 dt 2 noti


set output 'result-probes-2d.pdf'
set key top left
T_2d(r) = 500 + (1000-500) * log(r/0.5) / log(1/0.5)

p \
'result-2d-order2-x.dat' u 1:3 smooth unique w l lw 2 ti 'T',\
T_2d(x) w l lt -1 dt 2 noti
