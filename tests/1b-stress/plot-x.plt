#!/usr/bin/env gnuplot

L = 0.92
b = 0.025
h = 0.002

E = 1.56e11
F = 1.56e7

dim = 3
col_x = 'x'

# analytical solutions
u1_z(x) = x * F / (E)#*b*h)

set terminal pdfcairo rounded
set grid
set key top left

do for [order=1:3] {
    f = 'result-'.dim.'d-order'.order.'-x.dat'
    set output 'result-'.dim.'d-order'.order.'-x.pdf'

    p \
    u1_z(x), \
    f u col_x:'d_0[m]' w p
}
