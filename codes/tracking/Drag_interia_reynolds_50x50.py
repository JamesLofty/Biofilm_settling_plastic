#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:17:06 2024

@author: jameslofty
"""

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

PTFE_50x50_p = np.mean(density["PTFE 50x50"])
POM_50x50_p = np.mean(density["POM 50x50"])
PS_50x50_p = np.mean(density["PS 50x50"])

PTFE_50x50_b = np.mean(density["PTFE 50x50_b"])
POM_50x50_b = np.mean(density["POM 50x50_b"])
PS_50x50_b = np.mean(density["PS 50x50_b"])
#%%
#settling velocity

PTFE_50x50_p_W = np.array([26.47])
POM_50x50_p_W = np.array([13.72])
PS_50x50_p_W = np.array([4.17])

PTFE_50x50_b_W = np.array([26.39])
POM_50x50_b_W =  np.array([13.72])
PS_50x50_b_W = np.array([4.19])


#%%

PTFE_50x50_p_V = 0.06
POM_50x50_p_V = 0.069
PS_50x50_p_V = 0.06

PTFE_50x50_b_V = 0.07
POM_50x50_b_V =  0.071
PS_50x50_b_V = 0.06


#%%
g = 9.81
pf = 1000
v = 0.000001

d = 0.005
A = d*d
#%%

def Re_p(W, d_m, v): 
    Re_p = ((W/100) * d_m) / v
    return Re_p

###################### ReP 2x1 PRISTINE AVG

PTFE_50x50_ReP_p_avg = Re_p(PTFE_50x50_p_W[0], d, v)
POM_50x50_ReP_p_avg = Re_p(POM_50x50_p_W[0], d, v)
PS_50x50_ReP_p_avg = Re_p(PS_50x50_p_W[0], d, v)

###################### ReP 1x1 BIOFOULED AVG

PTFE_50x50_ReP_b_avg = Re_p(PTFE_50x50_b_W[0], d, v)
POM_50x50_ReP_b_avg = Re_p(POM_50x50_b_W[0], d, v)
PS_50x50_ReP_b_avg = Re_p(PS_50x50_b_W[0], d, v)

###################### ReP 1x1 PRISTINE MODE 1 (SLOW)


def cd(pp, pf, V , g, A, W):
    cd = (2 * (pp-pf) * V * g) / (pp * A * W**2)
    return cd

###################### cd 2x1 PRISTINE avg

PTFE_50x50_cd_p_avg = cd(PTFE_50x50_p, pf, (PTFE_50x50_p_V/1e+6), g, A, (PTFE_50x50_p_W[0]/100))
POM_50x50_cd_p_avg = cd(POM_50x50_p, pf, (POM_50x50_p_V/1e+6), g, A, (POM_50x50_p_W[0]/100))
PS_50x50_cd_p_avg = cd(PS_50x50_p, pf, (PS_50x50_p_V/1e+6), g, A, (PS_50x50_p_W[0]/100))

###################### cd 1x1 BIOFOULED avg

PTFE_50x50_cd_b_avg = cd(PTFE_50x50_b, pf, (PTFE_50x50_b_V/1e+6), g, A, (PTFE_50x50_b_W[0]/100))
POM_50x50_cd_b_avg = cd(POM_50x50_b, pf, (POM_50x50_b_V/1e+6), g, A, (POM_50x50_b_W[0]/100))
PS_50x50_cd_b_avg = cd(PS_50x50_b, pf, (PS_50x50_b_V/1e+6), g, A, (PS_50x50_b_W[0]/100))

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

plt.scatter(PTFE_50x50_ReP_p_avg, PTFE_50x50_cd_p_avg, color='darkgrey', marker = "o")
plt.scatter(POM_50x50_ReP_p_avg, POM_50x50_cd_p_avg, color='darkgrey', marker = "s")
plt.scatter(PS_50x50_ReP_p_avg, PS_50x50_cd_p_avg, color='darkgrey', marker = "^")

plt.scatter(PTFE_50x50_ReP_b_avg, PTFE_50x50_cd_b_avg, color='forestgreen', marker = "o")
plt.scatter(POM_50x50_ReP_b_avg, POM_50x50_cd_b_avg, color='forestgreen', marker = "s")
plt.scatter(PS_50x50_ReP_b_avg, PS_50x50_cd_b_avg,color='forestgreen', marker = "^")



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

plt.scatter(PTFE_50x50_ReP_p_avg, PTFE_50x50_cd_p_avg, color='darkgrey', marker = "o")
plt.scatter(POM_50x50_ReP_p_avg, POM_50x50_cd_p_avg, color='darkgrey', marker = "s")
plt.scatter(PS_50x50_ReP_p_avg, PS_50x50_cd_p_avg, color='darkgrey', marker = "^")

plt.scatter(PTFE_50x50_ReP_b_avg, PTFE_50x50_cd_b_avg, color='forestgreen', marker = "o")
plt.scatter(POM_50x50_ReP_b_avg, POM_50x50_cd_b_avg, color='forestgreen', marker = "s")
plt.scatter(PS_50x50_ReP_b_avg, PS_50x50_cd_b_avg,color='forestgreen', marker = "^")



plt.xlim(10, 5000)
plt.ylim(0.1, 100)
plt.xlabel('$Re_p$ (-)')
plt.ylabel('$C_D$ (-)')
# plt.legend()
# plt.legend(bbox_to_anchor=(-0.01, -0.2), loc='upper left', borderaxespad=1, ncol=2)
plt.savefig('figures/drag_small.svg', format='svg')


#%%

