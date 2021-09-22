#!/usr/bin/env gnuplot

rho = 500.
c_p = 1000.
lambda = 200.
a = lambda / (rho*c_p)
T1 = 500.
T2 = 1000.
L = 1.
# BC2
q = 3e5
# BC3
h = 2000.
T_ref = 1000.

T_1d_1(x, t) = 1000 * (1 - erf(x/sqrt(4*a*t)))
T_1d_2(x, t) = 2*q/lambda*sqrt(a*t/pi)*exp(-x**2/(4*a*t)) - q*x/lambda*(1 - erf(x/sqrt(4*a*t)))
T_1d_3(x, t) = T_ref * (1 - erf(x/sqrt(4*a*t)) - exp(h*x/lambda+h**2*a*t/lambda**2)*(1 - erf(x/sqrt(4*a*t)+h*sqrt(a*t)/lambda)))

f1 = 'probes-temperature-1d-BC1.txt'
f2 = 'probes-temperature-1d-BC2.txt'
f3 = 'probes-temperature-1d-BC3.txt'

set terminal pdfcairo rounded
set grid
set key bottom right opaque box reverse Left width 1
set samples 1001


set output 'result-probes-1d-BC1.pdf'
p \
for [i=0:5] f1 u 1:'T_'.i.'[K]' w l lw 2 ti 'T_'.i,\
for [i=0:5] T_1d_1(0.1*i, x) w l lt -1 dt 2 noti

set output 'result-probes-1d-BC2.pdf'
p \
for [i=0:5] f2 u 1:'T_'.i.'[K]' w l lw 2 ti 'T_'.i,\
for [i=0:5] T_1d_2(0.1*i, x) w l lt -1 dt 2 noti

set output 'result-probes-1d-BC3.pdf'
p \
for [i=0:5] f3 u 1:'T_'.i.'[K]' w l lw 2 ti 'T_'.i,\
for [i=0:5] T_1d_3(0.1*i, x) w l lt -1 dt 2 noti


set output 'result-probes-1d-BC1b-q.pdf'
T_1d_b(x, dot_q) = T1 + (T2-T1)*x/L + dot_q/(2*lambda)*x*(L-x)
p \
'result-1d-q4e5-x.dat' u 1:2 smooth unique w l lw 2 ti 'T',\
'result-1d-q-4e5-x.dat' u 1:2 smooth unique w l lw 2 ti 'T',\
T_1d_b(x, 4e5) w l lt -1 dt 2 noti,\
T_1d_b(x, -4e5) w l lt -1 dt 2 noti


set output 'result-probes-1d-BC1b-V.pdf'
T_1d_b(x, V) = T1 + (T2-T1)/(exp(L*rho*c_p*V/lambda)-1)*(exp(x*rho*c_p*V/lambda)-1)
p \
'result-1d-V1e-3-x.dat' u 1:2 smooth unique w l lw 2 ti 'T',\
'result-1d-V-1e-3-x.dat' u 1:2 smooth unique w l lw 2 ti 'T',\
T_1d_b(x, 1e-3) w l lt -1 dt 2 noti,\
T_1d_b(x, -1e-3) w l lt -1 dt 2 noti


set output 'result-probes-2d.pdf'
set key top left
T_2d(r) = T1 + (T2-T1) * log(r/0.5) / log(1/0.5)

p \
'result-2d-order2-x.dat' u 1:3 smooth unique w l lw 2 ti 'T',\
T_2d(x) w l lt -1 dt 2 noti
