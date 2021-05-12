#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,10cm font ',8'

set grid
set key top left Left reverse

dim=2
order=2

filename(I, boundary) = \
sprintf('T-%gd-order%g-I%g/result-temperature-2d-order2-t0-boundary%g.dat', dim, order, I, boundary)

z0 = 0.26685
dz = 0.06
set xrange [z0-dz:z0+dz] # symmetric to T maximum
set xlabel 'z, m'
set yrange [600:1600]
set ylabel 'Temperature, K'

set arrow from z0, graph 0 to z0, graph 1 lw 2 dt 3 nohead
z0_meas = z0-.0054
dz_meas = .0057
set arrow from z0_meas+dz_meas/2, graph 0 to z0_meas+dz_meas/2, graph 1 lw 2 dt 3 nohead
set arrow from z0_meas-dz_meas/2, graph 0 to z0_meas-dz_meas/2, graph 1 lw 2 dt 3 nohead

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
