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
PTFE_2x1_p = np.mean(density["PTFE 2x1"])
POM_2x1_p = np.mean(density["POM 2x1"])
PS_2x1_p = np.mean(density["PS 2x1"])

PTFE_1x1_p = np.mean(density["PTFE 1x1"])
POM_1x1_p = np.mean(density["POM 1x1"])
PS_1x1_p = np.mean(density["PS 1x1"])

PTFE_50x50_p = np.mean(density["PTFE 50x50"])
POM_50x50_p = np.mean(density["POM 50x50"])
PS_50x50_p = np.mean(density["PS 50x50"])

PTFE_2x1_b = np.mean(density["PTFE 2x1_b"])
POM_2x1_b = np.mean(density["POM 2x1_b"])
PS_2x1_b = np.mean(density["PS 2x1_b"])

PTFE_1x1_b = np.mean(density["PTFE 1x1_b"])
POM_1x1_b = np.mean(density["POM 1x1_b"])
PS_1x1_b = np.mean(density["PS 1x1_b"])

PTFE_50x50_b = np.mean(density["PTFE 50x50_b"])
POM_50x50_b = np.mean(density["POM 50x50_b"])
PS_50x50_b = np.mean(density["PS 50x50_b"])
#%%
#settling velocity
PTFE_2x1_p_W = np.array([9.17, 6.01, 31.09])
POM_2x1_p_W = np.array([6.14, 3.92, 19.22])
PS_2x1_p_W = np.array([2.25, 1.86, 6.54])

PTFE_1x1_p_W = np.array([6.77, 6.11, 25.58])
POM_1x1_p_W = np.array([5.44, 4.03, 17.1])
PS_1x1_p_W = np.array([2.03, 2.05])

PTFE_50x50_p_W = np.array([26.27])
POM_50x50_p_W = np.array([13.64])
PS_50x50_p_W = np.array([4.17])

PTFE_2x1_b_W = np.array([9.2, 5.89, 29.91])
POM_2x1_b_W = np.array([8.19, 3.91, 17.63])
PS_2x1_b_W = np.array([2.2, 1.95, 6.9])

PTFE_1x1_b_W = np.array([6.37, 6.02, 27.19])
POM_1x1_b_W = np.array([4.75, 4.11, 16.47])
PS_1x1_b_W = np.array([2.01, 2.03])

PTFE_50x50_b_W = np.array([26.24])
POM_50x50_b_W =  np.array([13.46])
PS_50x50_b_W = np.array([4.2])


#%%
PTFE_2x1_p_V = 0.175
POM_2x1_p_V = 0.242
PS_2x1_p_V = 0.205

PTFE_1x1_p_V = 0.089
POM_1x1_p_V = 0.133
PS_1x1_p_V = 0.108

PTFE_50x50_p_V = 0.06
POM_50x50_p_V = 0.069
PS_50x50_p_V = 0.06

PTFE_2x1_b_V = 0.236
POM_2x1_b_V = 0.233
PS_2x1_b_V = 0.221

PTFE_1x1_b_V = 0.114
POM_1x1_b_V = 0.119
PS_1x1_b_V = 0.111

PTFE_50x50_b_V = 0.07
POM_50x50_b_V =  0.071
PS_50x50_b_V = 0.06


#%%
g = 9.81
pf = 1000
v = 0.000001

L1 = 0.02
L2 = 0.01
L3 = 0.001
#%%

A_2x1_m1 = L2 * L1
A_2x1_m2 = L2 * L3

A_1x1_m1 = L2 * L2
A_1x1_m2 = L2 * L3

d_m_2x1_m1 = math.sqrt((4 * A_2x1_m1) / math.pi)
d_m_2x1_m2 = math.sqrt((4 * A_2x1_m2) / math.pi)

d_m_1x1_m1 = math.sqrt((4 * A_1x1_m1) / math.pi)
d_m_1x1_m2 = math.sqrt((4 * A_1x1_m2) / math.pi)

def Re_p(W, d_m, v): 
    Re_p = ((W/100) * d_m) / v
    return Re_p

###################### ReP 2x1 PRISTINE AVG

PTFE_2x1_ReP_p_avg = Re_p(PTFE_2x1_p_W[0], d_m_2x1_m1, v)
POM_2x1_ReP_p_avg = Re_p(POM_2x1_p_W[0], d_m_2x1_m1, v)
PS_2x1_ReP_p_avg = Re_p(PS_2x1_p_W[0], d_m_2x1_m1, v)

###################### ReP 2x1 BIOFOULED AVG

PTFE_2x1_ReP_b_avg = Re_p(PTFE_2x1_b_W[0], d_m_2x1_m1, v)
POM_2x1_ReP_b_avg = Re_p(POM_2x1_b_W[0], d_m_2x1_m1, v)
PS_2x1_ReP_b_avg = Re_p(PS_2x1_b_W[0], d_m_2x1_m1, v)

###################### ReP 2x1 PRISTINE MODE 1 (SLOW)

PTFE_2x1_ReP_p_m1 = Re_p(PTFE_2x1_p_W[1], d_m_2x1_m1, v)
POM_2x1_ReP_p_m1 = Re_p(POM_2x1_p_W[1], d_m_2x1_m1, v)
PS_2x1_ReP_p_m1 = Re_p(PS_2x1_p_W[1], d_m_2x1_m1, v)

###################### ReP 2x1 BIOFOULED MODE 1 (SLOW)

PTFE_2x1_ReP_b_m1 = Re_p(PTFE_2x1_b_W[1], d_m_2x1_m1, v)
POM_2x1_ReP_b_m1 = Re_p(POM_2x1_b_W[1], d_m_2x1_m1, v)
PS_2x1_ReP_b_m1 = Re_p(PS_2x1_b_W[1], d_m_2x1_m1, v)

###################### ReP 2x1 PRISTINE MODE 2 (FAST)

PTFE_2x1_ReP_p_m2 = Re_p(PTFE_2x1_p_W[2], d_m_2x1_m2, v)
POM_2x1_ReP_p_m2 = Re_p(POM_2x1_p_W[2], d_m_2x1_m2, v)
PS_2x1_ReP_p_m2 = Re_p(PS_2x1_p_W[2], d_m_2x1_m2, v)

###################### ReP 2x1 BIOFOULED MODE 2 (FAST)

PTFE_2x1_ReP_b_m2 = Re_p(PTFE_2x1_b_W[2], d_m_2x1_m2, v)
POM_2x1_ReP_b_m2 = Re_p(POM_2x1_b_W[2], d_m_2x1_m2, v)
PS_2x1_ReP_b_m2 = Re_p(PS_2x1_b_W[2], d_m_2x1_m2, v)



def cd(pp, pf, V , g, A, W):
    cd = (2 * (pp-pf) * V * g) / (pp * A * W**2)
    return cd

###################### cd 2x1 PRISTINE avg

PTFE_2x1_cd_p_avg = cd(PTFE_2x1_p, pf, (PTFE_2x1_p_V/10000000), g, A_2x1_m1, (PTFE_2x1_p_W[0]/100))
POM_2x1_cd_p_avg = cd(POM_2x1_p, pf, (POM_2x1_p_V/10000000), g, A_2x1_m1, (POM_2x1_p_W[0]/100))
PS_2x1_cd_p_avg = cd(PS_2x1_p, pf, (PS_2x1_p_V/10000000), g, A_2x1_m1, (PS_2x1_p_W[0]/100))

###################### cd 2x1 BIOFOULED avg

PTFE_2x1_cd_b_avg = cd(PTFE_2x1_b, pf, (PTFE_2x1_b_V/10000000), g, A_2x1_m1, (PTFE_2x1_b_W[0]/100))
POM_2x1_cd_b_avg = cd(POM_2x1_b, pf, (POM_2x1_b_V/10000000), g, A_2x1_m1, (POM_2x1_b_W[0]/100))
PS_2x1_cd_b_avg = cd(PS_2x1_b, pf, (PS_2x1_b_V/10000000), g, A_2x1_m1, (PS_2x1_b_W[0]/100))

###################### cd 2x1 PRISTINE MODE 1 (SLOW)

PTFE_2x1_cd_p_m1 = cd(PTFE_2x1_p, pf, (PTFE_2x1_p_V/10000000), g, A_2x1_m1, (PTFE_2x1_p_W[1]/100))
POM_2x1_cd_p_m1 = cd(POM_2x1_p, pf, (POM_2x1_p_V/10000000), g, A_2x1_m1, (POM_2x1_p_W[1]/100))
PS_2x1_cd_p_m1 = cd(PS_2x1_p, pf, (PS_2x1_p_V/10000000), g, A_2x1_m1, (PS_2x1_p_W[1]/100))

###################### cd 2x1 BIOFOULED MODE 1 (SLOW)

PTFE_2x1_cd_b_m1 = cd(PTFE_2x1_b, pf, (PTFE_2x1_b_V/10000000), g, A_2x1_m1, (PTFE_2x1_b_W[1]/100))
POM_2x1_cd_b_m1 = cd(POM_2x1_b, pf, (POM_2x1_b_V/10000000), g, A_2x1_m1, (POM_2x1_b_W[1]/100))
PS_2x1_cd_b_m1 = cd(PS_2x1_b, pf, (PS_2x1_b_V/10000000), g, A_2x1_m1, (PS_2x1_b_W[1]/100))

###################### cd 2x1 PRISTINE MODE 2 (FAST)

PTFE_2x1_cd_p_m2 = cd(PTFE_2x1_p, pf, (PTFE_2x1_p_V/10000000), g, A_2x1_m2, (PTFE_2x1_p_W[2]/100))
POM_2x1_cd_p_m2 = cd(POM_2x1_p, pf, (POM_2x1_p_V/10000000), g, A_2x1_m2, (POM_2x1_p_W[2]/100))
PS_2x1_cd_p_m2 = cd(PS_2x1_p, pf, (PS_2x1_p_V/10000000), g, A_2x1_m2, (PS_2x1_p_W[2]/100))

###################### cd 2x1 BIOFOULED MODE 2 (FAST)

PTFE_2x1_cd_b_m2 = cd(PTFE_2x1_b, pf, (PTFE_2x1_b_V/10000000), g, A_2x1_m2, (PTFE_2x1_b_W[2]/100))
POM_2x1_cd_b_m2 = cd(POM_2x1_b, pf, (POM_2x1_b_V/10000000), g, A_2x1_m2, (POM_2x1_b_W[2]/100))
PS_2x1_cd_b_m2 = cd(PS_2x1_b, pf, (PS_2x1_b_V/10000000), g, A_2x1_m2, (PS_2x1_b_W[2]/100))


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

plt.scatter(PTFE_2x1_ReP_p_avg, PTFE_2x1_cd_p_avg, color='darkgrey', marker = "o")
plt.scatter(POM_2x1_ReP_p_avg, POM_2x1_cd_p_avg, color='darkgrey', marker = "s")
plt.scatter(PS_2x1_ReP_p_avg, PS_2x1_cd_p_avg, color='darkgrey', marker = "^")

plt.scatter(PTFE_2x1_ReP_b_avg, PTFE_2x1_cd_b_avg, color='forestgreen', marker = "o")
plt.scatter(POM_2x1_ReP_b_avg, POM_2x1_cd_b_avg, color='forestgreen', marker = "s")
plt.scatter(PS_2x1_ReP_b_avg, PS_2x1_cd_b_avg,color='forestgreen', marker = "^")

plt.scatter(PTFE_2x1_ReP_p_m1, PTFE_2x1_cd_p_m1, color='none', marker = "o", edgecolor = "dimgrey")
plt.scatter(POM_2x1_ReP_p_m1, POM_2x1_cd_p_m1, color='none', marker = "s", edgecolor = "dimgrey")
plt.scatter(PS_2x1_ReP_p_m1, PS_2x1_cd_p_m1, color='none', marker = "^", edgecolor = "dimgrey")

plt.scatter(PTFE_2x1_ReP_b_m1, PTFE_2x1_cd_b_m1, color='none', marker = "o", edgecolor = "darkgreen")
plt.scatter(POM_2x1_ReP_b_m1, POM_2x1_cd_b_m1, color='none', marker = "s", edgecolor = "darkgreen")
plt.scatter(PS_2x1_ReP_b_m1, PS_2x1_cd_b_m1, color='none', marker = "^", edgecolor = "darkgreen")

plt.scatter(PTFE_2x1_ReP_p_m2, PTFE_2x1_cd_p_m2, color='none', marker = "o", edgecolor = "gainsboro")
plt.scatter(POM_2x1_ReP_p_m2, POM_2x1_cd_p_m2, color='none', marker = "s", edgecolor = "gainsboro")
plt.scatter(PS_2x1_ReP_p_m2, PS_2x1_cd_p_m2, color='none', marker = "^", edgecolor = "gainsboro")

plt.scatter(PTFE_2x1_ReP_b_m2, PTFE_2x1_cd_b_m2, color='none', marker = "o", edgecolor = "lightgreen")
plt.scatter(POM_2x1_ReP_b_m2, POM_2x1_cd_b_m2, color='none', marker = "s", edgecolor = "lightgreen")
plt.scatter(PS_2x1_ReP_b_m2, PS_2x1_cd_b_m2, color='none', marker = "^", edgecolor = "lightgreen")
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

plt.figure(figsize=(3, 2))
plt.loglog(Re, Cd, 'k', label = "Drag curve for sphere")

plt.scatter(PTFE_2x1_ReP_p_avg, PTFE_2x1_cd_p_avg, color='darkgrey', marker = "o")
plt.scatter(POM_2x1_ReP_p_avg, POM_2x1_cd_p_avg, color='darkgrey', marker = "s")
plt.scatter(PS_2x1_ReP_p_avg, PS_2x1_cd_p_avg, color='darkgrey', marker = "^")

plt.scatter(PTFE_2x1_ReP_b_avg, PTFE_2x1_cd_b_avg, color='forestgreen', marker = "o")
plt.scatter(POM_2x1_ReP_b_avg, POM_2x1_cd_b_avg, color='forestgreen', marker = "s")
plt.scatter(PS_2x1_ReP_b_avg, PS_2x1_cd_b_avg,color='forestgreen', marker = "^")

plt.scatter(PTFE_2x1_ReP_p_m1, PTFE_2x1_cd_p_m1, color='none', marker = "o", edgecolor = "dimgrey")
plt.scatter(POM_2x1_ReP_p_m1, POM_2x1_cd_p_m1, color='none', marker = "s", edgecolor = "dimgrey")
plt.scatter(PS_2x1_ReP_p_m1, PS_2x1_cd_p_m1, color='none', marker = "^", edgecolor = "dimgrey")

plt.scatter(PTFE_2x1_ReP_b_m1, PTFE_2x1_cd_b_m1, color='none', marker = "o", edgecolor = "darkgreen")
plt.scatter(POM_2x1_ReP_b_m1, POM_2x1_cd_b_m1, color='none', marker = "s", edgecolor = "darkgreen")
plt.scatter(PS_2x1_ReP_b_m1, PS_2x1_cd_b_m1, color='none', marker = "^", edgecolor = "darkgreen")

plt.scatter(PTFE_2x1_ReP_p_m2, PTFE_2x1_cd_p_m2, color='none', marker = "o", edgecolor = "gainsboro")
plt.scatter(POM_2x1_ReP_p_m2, POM_2x1_cd_p_m2, color='none', marker = "s", edgecolor = "gainsboro")
plt.scatter(PS_2x1_ReP_p_m2, PS_2x1_cd_p_m2, color='none', marker = "^", edgecolor = "gainsboro")

plt.scatter(PTFE_2x1_ReP_b_m2, PTFE_2x1_cd_b_m2, color='none', marker = "o", edgecolor = "lightgreen")
plt.scatter(POM_2x1_ReP_b_m2, POM_2x1_cd_b_m2, color='none', marker = "s", edgecolor = "lightgreen")
plt.scatter(PS_2x1_ReP_b_m2, PS_2x1_cd_b_m2, color='none', marker = "^", edgecolor = "lightgreen")

plt.xlim(100, 10000)
plt.ylim(0.01, 10)
plt.xlabel('$Re_p$ (-)')
plt.ylabel('$C_D$ (-)')
# plt.legend()
# plt.legend(bbox_to_anchor=(-0.01, -0.2), loc='upper left', borderaxespad=1, ncol=2)
plt.savefig('figures/drag_small.svg', format='svg')
