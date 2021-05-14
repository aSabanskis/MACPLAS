#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',8'

set grid
set key top left Left reverse

dim=2
order=2

filename(time, boundary) = \
sprintf('result-temperature-%gd-order%g-t%g-boundary%g.dat', dim, order, time, boundary)

set xrange [0:0.49]
set xlabel 'z, m'
set yrange [300:1600]
set ylabel 'Temperature, K'

set output sprintf('results-T(z,t)-%gd-order%g.pdf', dim, order)

t0 = 0
dt = 600
N = 24
Nstyles = 8
set key maxrows Nstyles

do for [i=1:N] { set linetype i lw 2 dt 1+(i-1)/Nstyles lc i%Nstyles }

set macros
style = "u 'z[m]':'T[K]' smooth unique w l"

p \
for [i=1:N] t=t0+(i-1)*dt, filename(t, 0) @style ti sprintf('%g s', t)
