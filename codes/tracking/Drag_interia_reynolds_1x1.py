#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:04:45 2023

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft
from numpy.fft import fft, ifft
import seaborn as sns
from scipy.signal import find_peaks
import math

density = pd.read_excel("/Users/jameslofty/Library/CloudStorage/OneDrive-CardiffUniversity/PhD/Biofilm+settling/particle_density/py_particle_density.xlsx")

#####denisty

PTFE_1x1_p = np.mean(density["PTFE 1x1"])
POM_1x1_p = np.mean(density["POM 1x1"])
PS_1x1_p = np.mean(density["PS 1x1"])


PTFE_1x1_b = np.mean(density["PTFE 1x1_b"])
POM_1x1_b = np.mean(density["POM 1x1_b"])
PS_1x1_b = np.mean(density["PS 1x1_b"])

#%%

PTFE_1x1_p_W = np.array([6.79, 6.15, 25.58])
POM_1x1_p_W = np.array([5.46, 4.08, 17.1])
PS_1x1_p_W = np.array([2.43, 2.06])

PTFE_1x1_b_W = np.array([6.43, 6.07, 27.19])
POM_1x1_b_W = np.array([4.78, 4.12, 16.46])
PS_1x1_b_W = np.array([2.04, 2.05, 4.07])


#%%
PTFE_1x1_p_V = 0.089
POM_1x1_p_V = 0.133
PS_1x1_p_V = 0.108

PTFE_1x1_b_V = 0.114
POM_1x1_b_V = 0.119
PS_1x1_b_V = 0.111



#%%
g = 9.81
pf = 1000
v = 0.000001

L1 = 0.01
L2 = 0.01
L3 = 0.001
#%%

A_1x1_m1 = L2 * L2
A_1x1_m2 = L2 * L3

# d_m_1x1_m1 = math.sqrt((4 * A_1x1_m1) / math.pi)
# d_m_1x1_m2 = math.sqrt((4 * A_1x1_m2) / math.pi)

def Re_p(W, l, v): 
    Re_p = ((W/100) * l) / v
    return Re_p

###################### ReP 2x1 PRISTINE AVG

PTFE_1x1_ReP_p_avg = Re_p(PTFE_1x1_p_W[0], L1, v)
POM_1x1_ReP_p_avg = Re_p(POM_1x1_p_W[0], L1, v)
PS_1x1_ReP_p_avg = Re_p(PS_1x1_p_W[0], L1, v)

###################### ReP 1x1 BIOFOULED AVG

PTFE_1x1_ReP_b_avg = Re_p(PTFE_1x1_b_W[0], L1, v)
POM_1x1_ReP_b_avg = Re_p(POM_1x1_b_W[0], L1, v)
PS_1x1_ReP_b_avg = Re_p(PS_1x1_b_W[0], L1, v)

###################### ReP 1x1 PRISTINE MODE 1 (SLOW)

PTFE_1x1_ReP_p_m1 = Re_p(PTFE_1x1_p_W[1], L1, v)
POM_1x1_ReP_p_m1 = Re_p(POM_1x1_p_W[1], L1, v)
PS_1x1_ReP_p_m1 = Re_p(PS_1x1_p_W[1], L1, v)

###################### ReP 1x1 BIOFOULED MODE 1 (SLOW)

PTFE_1x1_ReP_b_m1 = Re_p(PTFE_1x1_b_W[1], L1, v)
POM_1x1_ReP_b_m1 = Re_p(POM_1x1_b_W[1], L1, v)
PS_1x1_ReP_b_m1 = Re_p(PS_1x1_b_W[1], L1, v)

###################### ReP 1x1 PRISTINE MODE 2 (FAST)

PTFE_1x1_ReP_p_m2 = Re_p(PTFE_1x1_p_W[2], L3, v)
POM_1x1_ReP_p_m2 = Re_p(POM_1x1_p_W[2], L3, v)

###################### ReP 1x1 BIOFOULED MODE 2 (FAST)

PTFE_1x1_ReP_b_m2 = Re_p(PTFE_1x1_b_W[2], L3, v)
POM_1x1_ReP_b_m2 = Re_p(POM_1x1_b_W[2], L3, v)



def cd(pp, pf, V , g, A, W):
    cd = (2 * (pp-pf) * V * g) / (pp * A * W**2)
    return cd

###################### cd 2x1 PRISTINE avg

PTFE_1x1_cd_p_avg = cd(PTFE_1x1_p, pf, (PTFE_1x1_p_V/1e+6), g, A_1x1_m1, (PTFE_1x1_p_W[0]/100))
POM_1x1_cd_p_avg = cd(POM_1x1_p, pf, (POM_1x1_p_V/1e+6), g, A_1x1_m1, (POM_1x1_p_W[0]/100))
PS_1x1_cd_p_avg = cd(PS_1x1_p, pf, (PS_1x1_p_V/1e+6), g, A_1x1_m1, (PS_1x1_p_W[0]/100))

###################### cd 1x1 BIOFOULED avg

PTFE_1x1_cd_b_avg = cd(PTFE_1x1_b, pf, (PTFE_1x1_b_V/1e+6), g, A_1x1_m1, (PTFE_1x1_b_W[0]/100))
POM_1x1_cd_b_avg = cd(POM_1x1_b, pf, (POM_1x1_b_V/1e+6), g, A_1x1_m1, (POM_1x1_b_W[0]/100))
PS_1x1_cd_b_avg = cd(PS_1x1_b, pf, (PS_1x1_b_V/1e+6), g, A_1x1_m1, (PS_1x1_b_W[0]/100))

###################### cd 1x1 PRISTINE MODE 1 (SLOW)

PTFE_1x1_cd_p_m1 = cd(PTFE_1x1_p, pf, (PTFE_1x1_p_V/1e+6), g, A_1x1_m1, (PTFE_1x1_p_W[1]/100))
POM_1x1_cd_p_m1 = cd(POM_1x1_p, pf, (POM_1x1_p_V/1e+6), g, A_1x1_m1, (POM_1x1_p_W[1]/100))
PS_1x1_cd_p_m1 = cd(PS_1x1_p, pf, (PS_1x1_p_V/1e+6), g, A_1x1_m1, (PS_1x1_p_W[1]/100))

###################### cd 1x1 BIOFOULED MODE 1 (SLOW)

PTFE_1x1_cd_b_m1 = cd(PTFE_1x1_b, pf, (PTFE_1x1_b_V/1e+6), g, A_1x1_m1, (PTFE_1x1_b_W[1]/100))
POM_1x1_cd_b_m1 = cd(POM_1x1_b, pf, (POM_1x1_b_V/1e+6), g, A_1x1_m1, (POM_1x1_b_W[1]/100))
PS_1x1_cd_b_m1 = cd(PS_1x1_b, pf, (PS_1x1_b_V/1e+6), g, A_1x1_m1, (PS_1x1_b_W[1]/100))

###################### cd 1x1 PRISTINE MODE 2 (FAST)

PTFE_1x1_cd_p_m2 = cd(PTFE_1x1_p, pf, (PTFE_1x1_p_V/1e+6), g, A_1x1_m2, (PTFE_1x1_p_W[2]/100))
POM_1x1_cd_p_m2 = cd(POM_1x1_p, pf, (POM_1x1_p_V/1e+6), g, A_1x1_m2, (POM_1x1_p_W[2]/100))

###################### cd 1x1 BIOFOULED MODE 2 (FAST)

PTFE_1x1_cd_b_m2 = cd(PTFE_1x1_b, pf, (PTFE_1x1_b_V/1e+6), g, A_1x1_m2, (PTFE_1x1_b_W[2]/100))
POM_1x1_cd_b_m2 = cd(POM_1x1_b, pf, (POM_1x1_b_V/1e+6), g, A_1x1_m2, (POM_1x1_b_W[2]/100))



def I(rho_s, L3, L1, rho_f):
    
    e = rho_s / rho_f
    b = L3/L1
    y = L2 / L1
    
    I = (8 * e * b * (1 + (b**2))) / (3 * math.pi)
    
    return I


PTFE_1x1_I_p_avg = I(PTFE_1x1_p, L3, L1, pf)
POM_1x1_I_p_avg = I(POM_1x1_p, L3, L1, pf)
PS_1x1_I_p_avg = I(PS_1x1_p, L3, L1, pf)

PTFE_1x1_I_b_avg = I(PTFE_1x1_b, L3, L1, pf)
POM_1x1_I_b_avg = I(POM_1x1_b, L3, L1, pf)
PS_1x1_I_b_avg = I(PS_1x1_b, L3, L1, pf)

#%%
import Standard_Drag_Curve as SDC
# -----------------------------------
def compute_Re(ds, vw, mu):
    Re = ds*vw/mu
    return Re
# ----------------------------------

N = 10000
Re = 10**np.linspace(-2, 6, N)
Cd = np.zeros(N)

for i in range(0, N):
    Cd[i] = SDC.StandardDragCurve(Re[i])

plt.figure(figsize=(4, 3))
plt.loglog(Re, Cd, 'k', label = "Drag curve for sphere")

plt.scatter(PTFE_1x1_ReP_p_avg, PTFE_1x1_cd_p_avg, color='darkgrey', marker = "o")
plt.scatter(POM_1x1_ReP_p_avg, POM_1x1_cd_p_avg, color='darkgrey', marker = "s")
plt.scatter(PS_1x1_ReP_p_avg, PS_1x1_cd_p_avg, color='darkgrey', marker = "^")

plt.scatter(PTFE_1x1_ReP_b_avg, PTFE_1x1_cd_b_avg, color='forestgreen', marker = "o")
plt.scatter(POM_1x1_ReP_b_avg, POM_1x1_cd_b_avg, color='forestgreen', marker = "s")
plt.scatter(PS_1x1_ReP_b_avg, PS_1x1_cd_b_avg,color='forestgreen', marker = "^")

plt.scatter(PTFE_1x1_ReP_p_m1, PTFE_1x1_cd_p_m1, color='none', marker = "o", edgecolor = "dimgrey")
plt.scatter(POM_1x1_ReP_p_m1, POM_1x1_cd_p_m1, color='none', marker = "s", edgecolor = "dimgrey")
plt.scatter(PS_1x1_ReP_p_m1, PS_1x1_cd_p_m1, color='none', marker = "^", edgecolor = "dimgrey")

plt.scatter(PTFE_1x1_ReP_b_m1, PTFE_1x1_cd_b_m1, color='none', marker = "o", edgecolor = "darkgreen")
plt.scatter(POM_1x1_ReP_b_m1, POM_1x1_cd_b_m1, color='none', marker = "s", edgecolor = "darkgreen")
plt.scatter(PS_1x1_ReP_b_m1, PS_1x1_cd_b_m1, color='none', marker = "^", edgecolor = "darkgreen")

plt.scatter(PTFE_1x1_ReP_p_m2, PTFE_1x1_cd_p_m2, color='none', marker = "o", edgecolor = "gainsboro")
plt.scatter(POM_1x1_ReP_p_m2, POM_1x1_cd_p_m2, color='none', marker = "s", edgecolor = "gainsboro")

plt.scatter(PTFE_1x1_ReP_b_m2, PTFE_1x1_cd_b_m2, color='none', marker = "o", edgecolor = "lightgreen")
plt.scatter(POM_1x1_ReP_b_m2, POM_1x1_cd_b_m2, color='none', marker = "s", edgecolor = "lightgreen")
sns.despine(top=True, right=True, left=False, bottom=False)

plt.xlim(0.01, 1000000)
plt.ylim(0.001, 100000)
plt.xlabel('$Re_p$ (-)')
plt.ylabel('$C_D$ (-)')
plt.savefig('figures/drag_big.svg', format='svg')




#%%


import Standard_Drag_Curve as SDC
# -----------------------------------
def compute_Re(ds, vw, mu):
    Re = ds*vw/mu
    return Re
# ----------------------------------

N = 10000
Re = 10**np.linspace(-2, 6, N)
Cd = np.zeros(N)

for i in range(0, N):
    Cd[i] = SDC.StandardDragCurve(Re[i])

plt.figure(figsize=(1.5, 1.5))
plt.loglog(Re, Cd, 'k', label = "Drag curve for sphere")

plt.scatter(PTFE_1x1_ReP_p_avg, PTFE_1x1_cd_p_avg, color='darkgrey', marker = "o")
plt.scatter(POM_1x1_ReP_p_avg, POM_1x1_cd_p_avg, color='darkgrey', marker = "s")
plt.scatter(PS_1x1_ReP_p_avg, PS_1x1_cd_p_avg, color='darkgrey', marker = "^")

plt.scatter(PTFE_1x1_ReP_b_avg, PTFE_1x1_cd_b_avg, color='forestgreen', marker = "o")
plt.scatter(POM_1x1_ReP_b_avg, POM_1x1_cd_b_avg, color='forestgreen', marker = "s")
plt.scatter(PS_1x1_ReP_b_avg, PS_1x1_cd_b_avg,color='forestgreen', marker = "^")

plt.scatter(PTFE_1x1_ReP_p_m1, PTFE_1x1_cd_p_m1, color='none', marker = "o", edgecolor = "dimgrey")
plt.scatter(POM_1x1_ReP_p_m1, POM_1x1_cd_p_m1, color='none', marker = "s", edgecolor = "dimgrey")
plt.scatter(PS_1x1_ReP_p_m1, PS_1x1_cd_p_m1, color='none', marker = "^", edgecolor = "dimgrey")

plt.scatter(PTFE_1x1_ReP_b_m1, PTFE_1x1_cd_b_m1, color='none', marker = "o", edgecolor = "darkgreen")
plt.scatter(POM_1x1_ReP_b_m1, POM_1x1_cd_b_m1, color='none', marker = "s", edgecolor = "darkgreen")
plt.scatter(PS_1x1_ReP_b_m1, PS_1x1_cd_b_m1, color='none', marker = "^", edgecolor = "darkgreen")

plt.scatter(PTFE_1x1_ReP_p_m2, PTFE_1x1_cd_p_m2, color='none', marker = "o", edgecolor = "gainsboro")
plt.scatter(POM_1x1_ReP_p_m2, POM_1x1_cd_p_m2, color='none', marker = "s", edgecolor = "gainsboro")

plt.scatter(PTFE_1x1_ReP_b_m2, PTFE_1x1_cd_b_m2, color='none', marker = "o", edgecolor = "lightgreen")
plt.scatter(POM_1x1_ReP_b_m2, POM_1x1_cd_b_m2, color='none', marker = "s", edgecolor = "lightgreen")

plt.xlim(10, 5000)
plt.ylim(0.1, 100)
plt.xlabel('$Re_p$ (-)')
plt.ylabel('$C_D$ (-)')
# plt.legend()
# plt.legend(bbox_to_anchor=(-0.01, -0.2), loc='upper left', borderaxespad=1, ncol=2)
plt.savefig('figures/drag_small.svg', format='svg')


#%%


plt.figure(figsize=(3, 3))

plt.scatter(PTFE_1x1_ReP_p_avg, PTFE_1x1_I_p_avg, color='none', marker = "o", edgecolor = "silver")
plt.scatter(POM_1x1_ReP_p_avg, POM_1x1_I_p_avg, color='none', marker = "s", edgecolor = "silver")
plt.scatter(PS_1x1_ReP_p_avg, PS_1x1_I_p_avg, color='none', marker = "^", edgecolor = "silver")

plt.scatter(PTFE_1x1_ReP_b_avg, PTFE_1x1_I_b_avg, color='none', marker = "o", edgecolor = "lime")
plt.scatter(POM_1x1_ReP_b_avg, POM_1x1_I_b_avg, color='none', marker = "s", edgecolor = "lime")
plt.scatter(PS_1x1_ReP_b_avg, PS_1x1_I_b_avg,color='none', marker = "^", edgecolor = "lime")

# plt.scatter(PTFE_1x1_ReP_p_m1, PTFE_1x1_I_p_m1, color='none', marker = "o", edgecolor = "dimgrey")
# plt.scatter(POM_1x1_ReP_p_m1, POM_1x1_I_p_m1, color='none', marker = "s", edgecolor = "dimgrey")
# plt.scatter(PS_1x1_ReP_p_m1, PS_1x1_I_p_m1, color='none', marker = "^", edgecolor = "dimgrey")

# plt.scatter(PTFE_1x1_ReP_b_m1, PTFE_1x1_I_b_m1, color='none', marker = "o", edgecolor = "darkgreen")
# plt.scatter(POM_1x1_ReP_b_m1, POM_1x1_I_b_m1, color='none', marker = "s", edgecolor = "darkgreen")
# plt.scatter(PS_1x1_ReP_b_m1, PS_1x1_I_b_m1, color='none', marker = "^", edgecolor = "darkgreen")

# plt.scatter(PTFE_1x1_ReP_p_m2, PTFE_1x1_I_p_m2, color='none', marker = "o", edgecolor = "gainsboro")
# plt.scatter(POM_1x1_ReP_p_m2, POM_1x1_I_p_m2, color='none', marker = "s", edgecolor = "gainsboro")
# plt.scatter(PS_1x1_ReP_p_m2, PS_1x1_I_p_m2, color='none', marker = "^", edgecolor = "gainsboro")

# plt.scatter(PTFE_1x1_ReP_b_m2, PTFE_1x1_I_b_m2, color='none', marker = "o", edgecolor = "lightgreen")
# plt.scatter(POM_1x1_ReP_b_m2, POM_1x1_I_b_m2, color='none', marker = "s", edgecolor = "lightgreen")
# plt.scatter(PS_1x1_ReP_b_m2, PS_1x1_I_b_m2, color='none', marker = "^", edgecolor = "lightgreen")

plt.xscale('log')
plt.yscale('log')
plt.xlim(10, 10000)
plt.ylim(0.001, 10)
plt.xlabel('$Re_p$ (-)')
plt.ylabel('$I_*$ (-)')
# plt.legend()
# plt.legend(bbox_to_anchor=(-0.01, -0.2), loc='upper left', borderaxespad=1, ncol=2)
plt.savefig('figures/I_Rep.svg', format='svg')
print("PTFE_1x1_ReP_p_avg:", np.round(PTFE_1x1_ReP_p_avg, 3))
print("POM_1x1_ReP_p_avg:", np.round(POM_1x1_ReP_p_avg, 3))
print("PS_1x1_ReP_p_avg:", np.round(PS_1x1_ReP_p_avg, 3))

print("PTFE_1x1_ReP_b_avg:", np.round(PTFE_1x1_ReP_b_avg, 3))
print("POM_1x1_ReP_b_avg:", np.round(POM_1x1_ReP_b_avg, 3))
print("PS_1x1_ReP_b_avg:", np.round(PS_1x1_ReP_b_avg, 3))

print("PTFE_1x1_I_p_avg:", np.round(PTFE_1x1_I_p_avg, 3))
print("POM_1x1_I_p_avg:", np.round(POM_1x1_I_p_avg, 3))
print("PS_1x1_I_p_avg:", np.round(PS_1x1_I_p_avg, 3))

print("PTFE_1x1_I_b_avg:", np.round(PTFE_1x1_I_b_avg, 3))
print("POM_1x1_I_b_avg:", np.round(POM_1x1_I_b_avg, 3))
print("PS_1x1_I_b_avg:", np.round(PS_1x1_I_b_avg, 3))