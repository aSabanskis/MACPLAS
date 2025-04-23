#!/usr/bin/env gnuplot

set term pdfcairo rounded size 15cm,8cm font ',8'

set grid
set key center left Left reverse width -2

set output 'probes-minmax.pdf'
set multiplot layout 2,4

f = 'probes-dislocation-3d.txt'

col='T_0[K]'
col_min='T_min[K]'
col_max='T_max[K]'
set title col noenh
p \
f u ($1/60):col w l ti 'Probe 1', \
f u ($1/60):col_min w l ti 'Min', \
f u ($1/60):col_max w l ti 'Max'

unset key
col='displacement_2_0[m]'
col_min='displacement_min[m]'
col_max='displacement_max[m]'
set title col
rep

col='N_m_0[m^-2]'
col_min='N_m_min[m^-2]'
col_max='N_m_max[m^-2]'
set title col
rep

col='dot_N_m_0[m^-2s^-1]'
col_min='dot_N_m_min[m^-2s^-1]'
col_max='dot_N_m_max[m^-2s^-1]'
set title col
rep

set xlabel 'Time, min'
col='v_0[ms^-1]'
col_min='v_min[ms^-1]'
col_max='v_max[ms^-1]'
set title col
rep

col='stress_0_0[Pa]'
col_min='stress_min[Pa]'
col_max='stress_max[Pa]'
set title col
rep

col='tau_eff_0[Pa]'
col_min='tau_eff_min[Pa]'
col_max='tau_eff_max[Pa]'
set title col
rep

col='strain_c_0_0[-]'
col_min='strain_c_min[-]'
col_max='strain_c_max[-]'
set title col
rep
