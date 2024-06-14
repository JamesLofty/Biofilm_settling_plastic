#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:04:45 2024

@author: jameslofty
"""

import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft
from numpy.fft import fft, ifft
import seaborn as sns
from scipy.signal import find_peaks
import math
import imutils
from matplotlib import gridspec



file_paths = [
"POM_1x1_b1.xlsx",
"POM_1x1_b2.xlsx",
"PS_1x1_b1.xlsx",
"PS_1x1_b2.xlsx",
"PTFE_1x1_b1.xlsx",
"PTFE_1x1_b2.xlsx",
"POM_2x1_b1.xlsx",
"POM_2x1_b2.xlsx",
"PS_2x1_b1.xlsx",
"PS_2x1_b2.xlsx",
"PTFE_2x1_b1.xlsx",
"PTFE_2x1_b2.xlsx",
"POM_50x50.xlsx",
"PTFE_50x50.xlsx",
"PS_50x50.xlsx"
]



# List to store first column values from each file
percent_biofilm = []
biofilm_intensity = []


dataframes = {}

# Read each Excel file, extract the first column, and store the values
for file_path in file_paths:
    df = pd.read_excel(file_path)
    dataframes[file_path] = df  # Saving DataFrame with file path as key

POM_2x1 = pd.concat([dataframes["POM_2x1_b1.xlsx"], dataframes["POM_2x1_b2.xlsx"],])
PTFE_2x1 = pd.concat([dataframes["PTFE_2x1_b1.xlsx"], dataframes["PTFE_2x1_b2.xlsx"]])
PS_2x1 = pd.concat([dataframes["PS_2x1_b1.xlsx"], dataframes["PS_2x1_b2.xlsx"]])

POM_1x1 = pd.concat([dataframes["POM_1x1_b1.xlsx"], dataframes["POM_1x1_b2.xlsx"],])
PTFE_1x1 = pd.concat([dataframes["PTFE_1x1_b1.xlsx"], dataframes["PTFE_1x1_b2.xlsx"]])
PS_1x1 = pd.concat([dataframes["PS_1x1_b1.xlsx"], dataframes["PS_1x1_b2.xlsx"]])

POM_50x50 = pd.concat([dataframes["POM_50x50.xlsx"]])
PTFE_50x50 = pd.concat([dataframes["PTFE_50x50.xlsx"]])
PS_50x50 = pd.concat([dataframes["PS_50x50.xlsx"]])


POM_2x1['size'] = "2x1"
PTFE_2x1['size'] = "2x1"
PS_2x1['size'] = "2x1"

POM_1x1['size'] = "1x1"
PTFE_1x1['size'] = "1x1"
PS_1x1['size'] = "1x1"

POM_50x50['size'] = "50x50"
PTFE_50x50['size'] = "50x50"
PS_50x50['size'] = "50x50"

PS = pd.concat([PS_2x1, PS_1x1, PS_50x50])
POM = pd.concat([POM_2x1, POM_1x1, POM_50x50])
PTFE = pd.concat([PTFE_2x1, PTFE_1x1, PTFE_50x50])

PS["plastic"] = "PS"
POM["plastic"] = "POM"
PTFE["plastic"] = "PTFE"

ALL = pd.concat([PTFE, POM , PS])
# ALL.to_excel("/Users/jameslofty/Library/CloudStorage/OneDrive-CardiffUniversity/PhD/Biofilm+settling/particle_roughness_contact/" + "biofilm_photogrammery_data" + ".xlsx")
#%%

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)

spec = gridspec.GridSpec(ncols=3, nrows=2, wspace=0.4, hspace=0.4)

ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])
ax2 = fig.add_subplot(spec[2])

ax3 = fig.add_subplot(spec[3])
ax4 = fig.add_subplot(spec[4])
ax5 = fig.add_subplot(spec[5])


sns.boxplot(data=ALL[ALL["size"] == "2x1"], x='plastic', y='biofilm_percentages', 
            ax=ax0, color='yellowgreen', showfliers=False)

ax0.set_title('Rectangular plastic')
ax0.set_ylim(0, 100)
ax0.set_xlabel('')

sns.boxplot(data=ALL[ALL["size"] == "2x1"], x='plastic', y='std_intensity_means', 
            ax=ax3, color='yellowgreen', showfliers=False)
ax3.set_title('Rectangular plastic')
ax3.set_ylim(0, 0.25)
ax3.set_xlabel('')

sns.boxplot(data=ALL[ALL["size"] == "1x1"], x='plastic', y='biofilm_percentages', 
            ax=ax1, color='yellowgreen', showfliers=False)
ax1.set_ylabel('')
ax1.set_title('Square plastic')
ax1.set_ylim(0, 100)
ax1.set_xlabel('')

sns.boxplot(data=ALL[ALL["size"] == "1x1"], x='plastic', y='std_intensity_means', 
            ax=ax4, color='yellowgreen', showfliers=False)
ax4.set_ylabel('')
ax4.set_title('Square plastic')
ax4.set_ylim(0, 0.25)
ax4.set_xlabel('')

sns.boxplot(data=ALL[ALL["size"] == "50x50"], x='plastic', y='biofilm_percentages', 
            ax=ax2, color='yellowgreen', showfliers=False)
ax2.set_ylabel('')
ax2.set_title('Spherical plastic')
ax2.set_ylim(0, 100)
ax2.set_xlabel('')

sns.boxplot(data=ALL[ALL["size"] == "50x50"], x='plastic', y='std_intensity_means', 
            ax=ax5, color='yellowgreen', showfliers=False)
ax5.set_ylabel('')
ax5.set_title('Spherical plastic')
ax5.set_ylim(0, 0.25)
ax5.set_xlabel('')

for ax in [ax0, ax1, ax3, ax4]:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    
ax0.set_ylabel('Biofilm area coverage (%)')
ax3.set_ylabel('Biofilm spatial deviation \n(normalised pixel value)')


ax0.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax1.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax3.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax4.set_xticklabels(labels=["PTFE", "POM", "PS"])

plt.tight_layout()
sns.despine(top=True, right=True, left=False, bottom=False)

plt.savefig("biofilm_boxplots.svg", format="svg")

plt.show()