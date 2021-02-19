#!/usr/bin/env gnuplot

r_a = 0.5
r_b = 1
T_a = 500
T_b = 1000
c1 = -500
c2 = 1500

E=1.56e+11
alpha=3.2e-06
nu=0.25
T(r) = c1/r + c2

col_x = 'x'

# Hetnarski, R. B. (Ed.). (2014). Encyclopedia of thermal stresses.
# integral of T*r^2 from r_a to r
Tr2dr(r) = c1*r**2/2 + c2*r**3/3
# analytical solution
stress_r(r) = 2*alpha*E/(1-nu) * ( (r**3-r_a**3)/((r_b**3-r_a**3)*r**3) * (Tr2dr(r_b)-Tr2dr(r_a))  -  (Tr2dr(r)-Tr2dr(r_a))/r**3 )
stress_theta(r) = 2*alpha*E/(1-nu) * ( (2*r**3 + r_a**3)/(2*(r_b**3-r_a**3)*r**3) * (Tr2dr(r_b)-Tr2dr(r_a))  +  (Tr2dr(r)-Tr2dr(r_a))/(2*r**3)  -  T(r)/2 )
stress_r_theta(r) = 0

set terminal pdfcairo rounded
set grid
set xrange [r_a:r_b]
set yrange [-1e8:3e8]

do for [dim=2:3] {
    do for [order=1:2] {
        n_components = 2 * dim
        f = 'result-'.dim.'d-order'.order.'-x.dat'
        set output 'result-'.dim.'d-order'.order.'-x.pdf'

        p \
        stress_r(x),\
        stress_theta(x),\
        stress_r_theta(x),\
        for [k=0:n_components-1] f u col_x:'stress_'.k.'[Pa]' w p pt k+1 # phi = theta = 0 (x-axis)
    }
}
