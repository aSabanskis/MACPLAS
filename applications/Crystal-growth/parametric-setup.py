#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Script for setting-up parametric crystal model
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from os.path import exists

# DEFAULT PARAMETERS
params = {
    "L0": 8,
    "r0": 2,
    "L1": 38,
    "r1": 24,
    "L2": 70,
    "r2": 21,
    "L3": 23,
    "r3": 12,
    "d1": 0,
    "d2": 0,
    "v0": 0,
    "v1": 1,
    "v2": 1,
    "t0": 0,
    "t1": 30,
    "t2": 120,
    "Ta1": 300,
    "Ta2": 937,
    "HTa": 30,
    "mr": 85,
    "R_crucible": 43,
    "H0_crucible": 13,
    "H1_crucible": 58,
    "rho_c": 5370,
    "rho_m": 5534,
}

# READ PARAMETERS
filename = "problem.ini"
if exists(filename):
    print(f"Reading {filename}")
    data = open(filename).readlines()
    for line in data:
        k_v = line.strip().split("=")

        if len(k_v) == 2:
            key = k_v[0].strip()
            val = float(k_v[1])

            if key in params:
                print(f"Setting {key} to {val}")
                params[key] = val
            else:
                print(f"Parameter {key} not supported")
    print("Done.")
else:
    print(f"{filename} does not exist, skipping")


# RADIUS
x = np.array([params.get(f"L{x}") for x in range(4)])
y = np.array([params.get(f"r{x}") for x in range(4)])
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
L_total = x[-1]
L_0 = x[1]
print(f"L_total={L_total*1e3:g} mm")
print(f"L_0={L_0*1e3:g} mm")
np.savetxt("crystal-shape.dat", np.c_[x, y], fmt="%g")
R = interpolate.interp1d(x, y)
# save for later use
Length = x


# PULL RATE
x = np.array([params.get(f"t{x}") for x in range(3)])
y = np.array([params.get(f"v{x}") for x in range(3)])
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
rho_c = params.get("rho_c")
L_arr, dL = np.linspace(0, L_total, int(L_total / 1e-4), retstep=True)
for L in L_arr:
    m_total += rho_c * dL * np.pi * R(L) ** 2
print(f"m_total={m_total*1e3:g} g")

# MAX LENGTH
m_melt = m_total
m_residual = params.get("mr") * 1e-3
L = 0
dL = 1e-4
while True:
    L += dL
    dm = rho_c * dL * np.pi * R(L) ** 2
    m_melt -= dm
    if L >= L_total or m_melt <= m_residual:
        break
L_max = L
m_crystal = m_total - m_melt
print(f"L_max={L_max*1e3:g} mm")
print(f"m_crystal={m_crystal*1e3:g} g")

# CRUCIBLE SHAPE
H0 = params.get("H0_crucible")
H1 = params.get("H1_crucible")
x = np.array([0, H0, H0 + H1])
Rc = params.get("R_crucible")
y = np.array([0, Rc, Rc])

fig = plt.figure(figsize=(6, 4))
plt.plot(x, y, "o-")
plt.grid(True)
plt.gca().set_aspect("equal")
plt.xlabel("Height, mm")
plt.ylabel("Radius, mm")
plt.savefig("crucible-shape.png", dpi=150, bbox_inches="tight")

# convert to SI
x = x * 1e-3
y = y * 1e-3
np.savetxt("crucible-shape.dat", np.c_[x, y], fmt="%g")
H_crucible = y[-1]
print(f"H_crucible={H_crucible*1e3:g} mm")
R_crucible = interpolate.interp1d(x, y)

# MELT LEVEL
rho_m = params.get("rho_m")
h_melt = 0
m = 0
dh = 1e-4
while m < m_total:
    h_melt += dh
    dm = rho_m * dh * np.pi * R_crucible(h_melt) ** 2
    m += dm
print(f"h_melt={h_melt*1e3:g} mm")

m = m_total
h = h_melt
h_arr = []
L_arr, dL = np.linspace(0, L_max, int(L_max / 1e-4), retstep=True)
for L in L_arr:
    dm = rho_c * dL * np.pi * R(L) ** 2
    dh = dm / (rho_m * np.pi * R_crucible(h) ** 2)
    m -= dm
    h -= dh
    h_arr = np.append(h_arr, [h])

np.savetxt("melt-height.dat", np.c_[L_arr, h_arr], fmt="%g")
H_melt = interpolate.interp1d(L_arr, h_arr)

fig = plt.figure(figsize=(6, 4))
plt.plot(L_arr * 1e3, h_arr * 1e3, "-")
plt.grid(True)
plt.xlabel("Length, mm")
plt.ylabel("Melt height, mm")
plt.savefig("melt-height.png", dpi=150, bbox_inches="tight")

zT = H_crucible + params.get("HTa") * 1e-3 - h_melt
Ta1 = params.get("Ta1") + 273
Ta2 = params.get("Ta2") + 273
Tamb = f"z>{zT:g} ? {Ta1} : {Ta2}"
print(f"Tamb={Tamb}")

# MAX TIME
t = 0
dt = 2.0
L = 0
z_top = L_0
while L < L_max:
    t += dt
    dz = V(t) * dt
    z_top += dz
    L = z_top + (h_melt - H_melt(L))
t_max = t
print(f"t_max={t_max:g} s")

# INTERFACE SHAPE
x = Length * 1e3
y = np.array([params.get("d1"), params.get("d2"), 0])
while y.size < x.size:
    y = np.append([0], y)

fig = plt.figure(figsize=(6, 4))
plt.plot(x, y, "o-")
plt.grid(True)
plt.xlabel("Length, mm")
plt.ylabel("Deflection, mm")
plt.savefig("deflection.png", dpi=150, bbox_inches="tight")

# convert to SI
x = x * 1e-3
y = y * 1e-3
deflection = interpolate.interp1d(x, y)

L_arr = np.linspace(0, L_max, int(L_max / 1e-3))
R_arr = np.linspace(0, 1, 11)
with open("interface-shape.dat", "w") as f:
    f.write(f"{R_arr.size} {L_arr.size}\n")
    f.write(f"-1")  # dummy value
    for r in R_arr:
        f.write(f" {r:g}")
    f.write("\n")

    for L in L_arr:
        f.write(f"{L:g}")

        d = deflection(L)
        z0 = H_melt(L) - H_melt(L_0)

        for r in R_arr:
            z = z0 + d * (r ** 2 - 1)
            f.write(f" {z:g}")
        f.write("\n")


# PARAMETER FILE
with open("problem-generated.prm", "w") as f:
    f.write("# Auto-generated file, please do not edit\n")
    f.write("set Crystal radius = crystal-shape.dat\n")
    f.write("set Pull rate = pull-rate.dat\n")
    f.write("set Interface shape = interface-shape.dat\n")
    f.write(f"set Ambient temperature = {Tamb}\n")
    f.write(f"set Max time = {t_max:g}\n")

lines = open("problem.prm").readlines()
include_found = False
with open("problem.prm", "w") as f:
    for line in lines:
        f.write(line)
        if line.startswith("include "):
            include_found = True
    if not include_found:
        f.write("include problem-generated.prm")
