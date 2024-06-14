#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:13:53 2024

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from scipy import signal
import math
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.animation import PillowWriter

PTFE_2x1_ReP_p_avg= 1814.0
POM_2x1_ReP_p_avg= 1340.0
PS_2x1_ReP_p_avg= 466.0
PTFE_2x1_ReP_b_avg= 1738.0
POM_2x1_ReP_b_avg= 1270.0
PS_2x1_ReP_b_avg= 458.0

PTFE_1x1_ReP_p_avg= 679.0
POM_1x1_ReP_p_avg= 546.0
PS_1x1_ReP_p_avg= 243.0
PTFE_1x1_ReP_b_avg= 643.0
POM_1x1_ReP_b_avg= 478.0
PS_1x1_ReP_b_avg= 204.0

PTFE_2x1_I_p_avg= 0.084
POM_2x1_I_p_avg= 0.06
PS_2x1_I_p_avg= 0.046
PTFE_2x1_I_b_avg= 0.084
POM_2x1_I_b_avg= 0.061
PS_2x1_I_b_avg= 0.046

PTFE_1x1_I_p_avg= 0.17
POM_1x1_I_p_avg= 0.121
PS_1x1_I_p_avg= 0.093
PTFE_1x1_I_b_avg= 0.172
POM_1x1_I_b_avg= 0.122
PS_1x1_I_b_avg= 0.096

freq_2x1 = [1.73,1.83,1.52,1.67,0.94,0.68]
freq_1x1 = [2.52,2.7,1.91,2.08,0.84,0.89]

st_2x1 = [0.281,0.306,0.386,0.424,0.488,0.34]
st_1x1 = [0.429,0.44,0.472,0.487,0.399,0.438]

amp_2x1 = [0.88,0.98,0.54,0.53,0.4,0.51]
amp_1x1 =[0.44,0.5,0.32,0.34,0.34,0.42]

#%%

# Create figure and axes objects
fig, axs = plt.subplots(1, 2, figsize=(5, 2))

# Plot for St values
axs[0].scatter(PTFE_2x1_ReP_p_avg, st_2x1[0], label='PTFE 2x1 ReP_p', color='none', marker="o", edgecolor="dimgrey")
axs[0].scatter(PTFE_2x1_ReP_b_avg, st_2x1[1], label='PTFE 2x1 ReP_b', color='none', marker="o", edgecolor="green")
axs[0].scatter(POM_2x1_ReP_p_avg, st_2x1[2], label='POM 2x1 ReP_p', color='none', marker="s", edgecolor="dimgrey")
axs[0].scatter(POM_2x1_ReP_b_avg, st_2x1[3], label='POM 2x1 ReP_b', color='none', marker="s", edgecolor="green")
axs[0].scatter(PS_2x1_ReP_p_avg, st_2x1[4], label='PS 2x1 ReP_p', color='none', marker="^", edgecolor="dimgrey")
axs[0].scatter(PS_2x1_ReP_b_avg, st_2x1[5], label='PS 2x1 ReP_b', color='none', marker="^", edgecolor="green")

axs[0].scatter(PTFE_1x1_ReP_p_avg, st_1x1[0], label='PTFE 1x1 ReP_p', color='none', marker="o", edgecolor="silver")
axs[0].scatter(PTFE_1x1_ReP_b_avg, st_1x1[1], label='PTFE 1x1 ReP_b', color='none', marker="o", edgecolor="lime")
axs[0].scatter(POM_1x1_ReP_p_avg, st_1x1[2], label='POM 1x1 ReP_p', color='none', marker="s", edgecolor="silver")
axs[0].scatter(POM_1x1_ReP_b_avg, st_1x1[3], label='POM 1x1 ReP_b', color='none', marker="s", edgecolor="lime")
axs[0].scatter(PS_1x1_ReP_p_avg, st_1x1[4], label='PS 1x1 ReP_p', color='none', marker="^", edgecolor="silver")
axs[0].scatter(PS_1x1_ReP_b_avg, st_1x1[5], label='PS 1x1 ReP_b', color='none', marker="^", edgecolor="lime")

axs[0].set_ylabel("St (-)")
axs[0].set_xlabel("Re$_p$ (-)")

axs[0].set_ylim(0.25, 0.5)
axs[0].set_xlim(0, 2000)
# axs[0].legend()
sns.despine(ax=axs[0], top=True, right=True, left=False, bottom=False)

# Plot for α values
axs[1].scatter(PTFE_2x1_ReP_p_avg, amp_2x1[0], label='PTFE 2x1 ReP_p', color='none', marker="o", edgecolor="dimgrey")
axs[1].scatter(PTFE_2x1_ReP_b_avg, amp_2x1[1], label='PTFE 2x1 ReP_b', color='none', marker="o", edgecolor="green")
axs[1].scatter(POM_2x1_ReP_p_avg, amp_2x1[2], label='POM 2x1 ReP_p', color='none', marker="s", edgecolor="dimgrey")
axs[1].scatter(POM_2x1_ReP_b_avg, amp_2x1[3], label='POM 2x1 ReP_b', color='none', marker="s", edgecolor="green")
axs[1].scatter(PS_2x1_ReP_p_avg, amp_2x1[4], label='PS 2x1 ReP_p', color='none', marker="^", edgecolor="dimgrey")
axs[1].scatter(PS_2x1_ReP_b_avg, amp_2x1[5], label='PS 2x1 ReP_b', color='none', marker="^", edgecolor="green")

axs[1].scatter(PTFE_1x1_ReP_p_avg, amp_1x1[0], label='PTFE 1x1 ReP_p', color='none', marker="o", edgecolor="silver")
axs[1].scatter(PTFE_1x1_ReP_b_avg, amp_1x1[1], label='PTFE 1x1 ReP_b', color='none', marker="o", edgecolor="lime")
axs[1].scatter(POM_1x1_ReP_p_avg, amp_1x1[2], label='POM 1x1 ReP_p', color='none', marker="s", edgecolor="silver")
axs[1].scatter(POM_1x1_ReP_b_avg, amp_1x1[3], label='POM 1x1 ReP_b', color='none', marker="s", edgecolor="lime")
axs[1].scatter(PS_1x1_ReP_p_avg, amp_1x1[4], label='PS 1x1 ReP_p', color='none', marker="^", edgecolor="silver")
axs[1].scatter(PS_1x1_ReP_b_avg, amp_1x1[5], label='PS 1x1 ReP_b', color='none', marker="^", edgecolor="lime")

axs[1].set_xlabel("Re$_p$ (-)")
axs[1].set_ylabel("α (cm)")
axs[1].set_ylim(0, 1.2)
axs[1].set_xlim(0, 2000)
# axs[1].legend()
sns.despine(ax=axs[1], top=True, right=True, left=False, bottom=False)

plt.tight_layout()
plt.savefig("figures/amp_freq_reP.svg", format="svg")


#%%
fig, axs = plt.subplots(1, 2, figsize=(5, 2))

# Plot for St values
axs[0].scatter(PTFE_2x1_I_p_avg, st_2x1[0], label='PTFE 2x1 I_p', color='none', marker="o", edgecolor="dimgrey")
axs[0].scatter(PTFE_2x1_I_b_avg, st_2x1[1], label='PTFE 2x1 I_b', color='none', marker="o", edgecolor="green")
axs[0].scatter(POM_2x1_I_p_avg, st_2x1[2], label='POM 2x1 I_p', color='none', marker="s", edgecolor="dimgrey")
axs[0].scatter(POM_2x1_I_b_avg, st_2x1[3], label='POM 2x1 I_b', color='none', marker="s", edgecolor="green")
axs[0].scatter(PS_2x1_I_p_avg, st_2x1[4], label='PS 2x1 I_p', color='none', marker="^", edgecolor="dimgrey")
axs[0].scatter(PS_2x1_I_b_avg, st_2x1[5], label='PS 2x1 I_b', color='none', marker="^", edgecolor="green")

axs[0].scatter(PTFE_1x1_I_p_avg, st_1x1[0], label='PTFE 1x1 I_p', color='none', marker="o", edgecolor="silver")
axs[0].scatter(PTFE_1x1_I_b_avg, st_1x1[1], label='PTFE 1x1 I_b', color='none', marker="o", edgecolor="lime")
axs[0].scatter(POM_1x1_I_p_avg, st_1x1[2], label='POM 1x1 I_p', color='none', marker="s", edgecolor="silver")
axs[0].scatter(POM_1x1_I_b_avg, st_1x1[3], label='POM 1x1 I_b', color='none', marker="s", edgecolor="lime")
axs[0].scatter(PS_1x1_I_p_avg, st_1x1[4], label='PS 1x1 I_p', color='none', marker="^", edgecolor="silver")
axs[0].scatter(PS_1x1_I_b_avg, st_1x1[5], label='PS 1x1 I_b', color='none', marker="^", edgecolor="lime")

axs[0].set_ylabel("St (-)")
axs[0].set_xlabel("I$_*$ (-)")

axs[0].set_ylim(0.25, 0.5)
axs[0].set_xlim(0, 0.2)
# axs[0].legend()
sns.despine(ax=axs[0], top=True, right=True, left=False, bottom=False)

# Plot for α values
axs[1].scatter(PTFE_2x1_I_p_avg, amp_2x1[0], label='PTFE 2x1 I_p', color='none', marker="o", edgecolor="dimgrey")
axs[1].scatter(PTFE_2x1_I_b_avg, amp_2x1[1], label='PTFE 2x1 I_b', color='none', marker="o", edgecolor="green")
axs[1].scatter(POM_2x1_I_p_avg, amp_2x1[2], label='POM 2x1 I_p', color='none', marker="s", edgecolor="dimgrey")
axs[1].scatter(POM_2x1_I_b_avg, amp_2x1[3], label='POM 2x1 I_b', color='none', marker="s", edgecolor="green")
axs[1].scatter(PS_2x1_I_p_avg, amp_2x1[4], label='PS 2x1 I_p', color='none', marker="^", edgecolor="dimgrey")
axs[1].scatter(PS_2x1_I_b_avg, amp_2x1[5], label='PS 2x1 I_b', color='none', marker="^", edgecolor="green")

axs[1].scatter(PTFE_1x1_I_p_avg, amp_1x1[0], label='PTFE 1x1 I_p', color='none', marker="o", edgecolor="silver")
axs[1].scatter(PTFE_1x1_I_b_avg, amp_1x1[1], label='PTFE 1x1 I_b', color='none', marker="o", edgecolor="lime")
axs[1].scatter(POM_1x1_I_p_avg, amp_1x1[2], label='POM 1x1 I_p', color='none', marker="s", edgecolor="silver")
axs[1].scatter(POM_1x1_I_b_avg, amp_1x1[3], label='POM 1x1 I_b', color='none', marker="s", edgecolor="lime")
axs[1].scatter(PS_1x1_I_p_avg, amp_1x1[4], label='PS 1x1 I_p', color='none', marker="^", edgecolor="silver")
axs[1].scatter(PS_1x1_I_b_avg, amp_1x1[5], label='PS 1x1 I_b', color='none', marker="^", edgecolor="lime")

axs[1].set_xlabel("I$_*$ (-)")
axs[1].set_ylabel("α (cm)")
axs[1].set_ylim(0, 1.2)
axs[1].set_xlim(0, 0.2)
# axs[1].legend()
sns.despine(ax=axs[1], top=True, right=True, left=False, bottom=False)

plt.tight_layout()
plt.savefig("figures/amp_freq_I.svg", format="svg")