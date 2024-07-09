#!/usr/bin/env gnuplot

L = 0.92
b = 0.025
h = 0.002
S = b * h

E = 1.56e11
F = 1.56e7 * S

dim = 3
col_x = 'x'

# analytical solutions
u1_x(x) = x * F / S / E

I = b * h**3 / 12
# force on end
u2_z(x) = F * x**2 * (3*L-x) / (6*E*I)
# constant force on horizontal surface
u3_z(x) = F/L * x**2 * (x**2 + 6*L**2 - 4*L*x) / (24*E*I)

set terminal pdfcairo rounded
set grid
set key top left

set macros
styleSim = "smooth unique w lp ps 0.3 ti 'order='.order"

set output 'result1-'.dim.'d-x.pdf'
p \
for [order=1:3] 'result-'.dim.'d-order'.order.'-BCf1-1.56e+07_0_0-x.dat' u col_x:'d_0[m]' @styleSim, \
u1_x(x) lt -1 ti 'Analytical'

F = 1 * S
set output 'result2-'.dim.'d-x.pdf'
p \
for [order=1:3] 'result-'.dim.'d-order'.order.'-BCf1-0_0_1-x.dat' u col_x:'d_2[m]' @styleSim, \
u2_z(x) lt -1 ti 'Analytical'

F = 1 * b*L
set output 'result3-'.dim.'d-x.pdf'
p \
for [order=1:3] 'result-'.dim.'d-order'.order.'-BCf4-0_0_1-x.dat' u col_x:'d_2[m]' @styleSim, \
u3_z(x) lt -1 ti 'Analytical'
