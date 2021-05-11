#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',8'

set grid
set key top left Left reverse

dim=2
order=2

filename(I, boundary) = \
sprintf('T-%gd-order%g-I%g/result-temperature-2d-order2-t0-boundary%g.dat', dim, order, I, boundary)

set xrange [0.2:0.334] # symmetric to T maximum
set xlabel 'z, m'
set yrange [600:1600]
set ylabel 'Temperature, K'

set output sprintf('T-%gd-order%g.pdf', dim, order)

I0 = 30
dI = 10
N = 8
set key maxrows N

set macros
style = "u 'z[m]':'T[K]' smooth unique w l lt i"

p \
for [i=1:N] I=I0+(i-1)*dI, filename(I, 0) @style dt 1 ti sprintf('%g A surf.', I), \
for [i=1:N] I=I0+(i-1)*dI, filename(I, 1) @style dt 2 ti sprintf('%g A axis', I)
