#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Script for setting-up parametric crystal model
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

params = {
    "L0": 8,
    "r0": 2,
    "L1": 38,
    "r1": 24,
    "L2": 70,
    "r2": 21,
    "L3": 23,
    "r3": 12,
    "v0": 0,
    "v1": 1,
    "v2": 1,
    "t0": 5,
    "t1": 30,
    "t2": 120,
    "Ta1": 300,
    "Ta2": 940,
    "rho": 5370,
}


# RADIUS
x = [params.get(f"L{x}") for x in range(4)]
y = [params.get(f"r{x}") for x in range(4)]
if x[0] > 0:
    x = np.append([0], x)
    y = np.append([y[0]], y)
x = np.cumsum(x)

fig = plt.figure(figsize=(6, 4))
plt.plot(x, y, "o-")
plt.grid(True)
plt.gca().set_aspect("equal")
plt.xlabel("Length, mm")
plt.ylabel("Radius, mm")
plt.savefig("crystal-shape.png", dpi=150, bbox_inches="tight")

# convert to SI
x = x * 1e-3
y = y * 1e-3
L_max = x[-1]
print(f"L_max={L_max*1e3:g} mm")
np.savetxt("crystal-shape.dat", np.c_[x, y], fmt="%g")
R = interpolate.interp1d(x, y)


# PULL RATE
x = [params.get(f"t{x}") for x in range(3)]
y = [params.get(f"v{x}") for x in range(3)]
if x[0] > 0:
    x = np.append([0], x)
    y = np.append([0], y)
x = np.cumsum(x)

fig = plt.figure(figsize=(6, 4))
plt.plot(x, y, "o-")
plt.grid(True)
plt.xlabel("Time, min")
plt.ylabel("Pull rate, mm/min")
plt.savefig("pull-rate.png", dpi=150, bbox_inches="tight")

# convert to SI
x = x * 60
y = y * 1e-3 / 60
np.savetxt("pull-rate.dat", np.c_[x, y], fmt="%g")
V = interpolate.interp1d(x, y)


# MASS
m_total = 0
rho = params.get("rho")
L_arr, dL = np.linspace(0, L_max, int(L_max / 1e-4), retstep=True)
for L in L_arr:
    m_total += rho * dL * np.pi * R(L) ** 2
print(f"m={m_total*1e3:g} g")
