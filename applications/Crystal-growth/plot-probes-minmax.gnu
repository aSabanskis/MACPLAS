#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,8cm font ',8'

set grid
set key center left Left reverse width -2

set output 'probes-minmax.pdf'
set multiplot layout 2,4

f = 'probes-temperature-2d.txt'

col='T_0[K]'
col_min='T_min[K]'
col_max='T_max[K]'
set title col noenh
p \
f u ($1/60):col w l ti 'Probe 1', \
f u ($1/60):col_min w l ti 'Min', \
f u ($1/60):col_max w l ti 'Max'

f = 'probes-dislocation-2d.txt'
f2 = 'probes-crystal-2d.txt'

col='N_m_0[m^-2]'
col_min='N_m_min[m^-2]'
col_max='N_m_max[m^-2]'
col_a='N_m_axis[m^-2]'
col_t='N_m_triple[m^-2]'
set title col
p \
f u ($1/60):col w l ti 'Probe 1', \
f u ($1/60):col_min w l ti 'Min', \
f u ($1/60):col_max w l ti 'Max', \
f2 u ($1/60):col_a w l ti 'Axis', \
f2 u ($1/60):col_t w l ti 'Triple'

unset key
col='dot_N_m_0[m^-2s^-1]'
col_min='dot_N_m_min[m^-2s^-1]'
col_max='dot_N_m_max[m^-2s^-1]'
set title col
p \
f u ($1/60):col w l ti 'Probe 1', \
f u ($1/60):col_min w l ti 'Min', \
f u ($1/60):col_max w l ti 'Max'

col='v_0[ms^-1]'
col_min='v_min[ms^-1]'
col_max='v_max[ms^-1]'
set title col
rep

set xlabel 'Time, min'
col='stress_0_0[Pa]'
col_min='stress_min[Pa]'
col_max='stress_max[Pa]'
set title col
rep

col='tau_eff_0[Pa]'
col_min='tau_eff_min[Pa]'
col_max='tau_eff_max[Pa]'
col_a='tau_eff_axis[Pa]'
col_t='tau_eff_triple[Pa]'
set title col
p \
f u ($1/60):col w l ti 'Probe 1', \
f u ($1/60):col_min w l ti 'Min', \
f u ($1/60):col_max w l ti 'Max', \
f2 u ($1/60):col_a w l ti 'Axis', \
f2 u ($1/60):col_t w l ti 'Triple'

col='strain_c_0_0[-]'
col_min='strain_c_min[-]'
col_max='strain_c_max[-]'
set title col
p \
f u ($1/60):col w l ti 'Probe 1', \
f u ($1/60):col_min w l ti 'Min', \
f u ($1/60):col_max w l ti 'Max'

col='dot_strain_c_0_0[s^-1]'
col_min='dot_strain_c_min[s^-1]'
col_max='dot_strain_c_max[s^-1]'
set title col
rep
