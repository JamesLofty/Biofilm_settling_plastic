#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:20:33 2024

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
import scipy.stats
from itertools import zip_longest
from scipy.stats import mannwhitneyu


# def calculate_amplitude(x_coords, y_coords):
#     # Calculate displacement along each axis    
#     delta_x = max(x_coords) - min(x_coords)
#     delta_y = max(y_coords) - min(y_coords)

#     # Calculate the amplitude using Pythagorean theorem
#     amplitude = np.sqrt(delta_x**2 + delta_y**2) / 2
    
#     return amplitude

def calculate_amplitude(x_coords, y_coords, z_coords):
    # Calculate the mean position
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    # Calculate the displacement from the mean position for each point
    displacements = np.sqrt((x_coords - mean_x)**2 + (y_coords - mean_y)**2)
    # plt.scatter(displacements, z_coords)
    # The amplitude is the maximum displacement
    amplitude = np.mean(displacements)
    # plt.vlines(amplitude, 0, 20)
    return amplitude


#%%

plastic = "PTFE"
size = "_1x1_"

data1_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_p.xlsx")
data2_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos2_p.xlsx")
data3_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos3_p.xlsx")

data2_p["label_mid"] = data2_p["label_mid"] + 100
data3_p["label_mid"] = data3_p["label_mid"] + 1000

data_all_p_null = pd.concat([data1_p, data2_p, data3_p])
data_all_p = data_all_p_null[data_all_p_null["mode"] == 1]
#%%
data1_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_b.xlsx")
data2_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos2_b.xlsx")
data3_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos3_b.xlsx")

data2_b["label_mid"] = data2_b["label_mid"] + 100
data3_b["label_mid"] = data3_b["label_mid"] + 1000

data_all_b_null = pd.concat([data1_b, data2_b, data3_b])
data_all_b = data_all_b_null[data_all_b_null["mode"] == 1]

#%%

x_coords_p = np.array(data_all_p["x_mid"])
y_coords_p = np.array(data_all_p["y_mid"])
z_coords_p = np.array(data_all_p["z_mid"])
z_y_coords_p = np.sqrt(x_coords_p**2 + y_coords_p**2)
wxwy_p = np.array(((data_all_p["wx"] ** 2) + (data_all_p["wy"] ** 2)) ** 0.5)
label_p = data_all_p["label_mid"]



x_coords_b = np.array(data_all_b["x_mid"])
y_coords_b = np.array(data_all_b["y_mid"])
z_coords_b = np.array(data_all_b["z_mid"])
z_y_coords_b = np.sqrt(x_coords_b**2 + y_coords_b**2)
wxwy_b = np.array(((data_all_b["wx"] ** 2) + (data_all_b["wy"] ** 2)) ** 0.5)
label_b = data_all_b["label_mid"]

#%%

amplitudes_p = []
amplitudes_b = []


for i in label_p.unique(): 
    amplitude_p = calculate_amplitude(x_coords_p[label_p == i], y_coords_p[label_p == i], 
                                      z_coords_p[label_p == i])
    
    amplitudes_p.append(amplitude_p)
    

for i in label_b.unique(): 
    amplitude_b = calculate_amplitude(x_coords_b[label_b == i], y_coords_b[label_b == i],
                                      z_coords_b[label_b == i])
    
    amplitudes_b.append(amplitude_b)
    

#%%
binsize = 0.5
min_value = np.min(x_coords_b)
max_value = np.max(x_coords_b)
bin_width = binsize
num_bins = int((max_value - min_value) / bin_width)
bin_edges = np.linspace(min_value, max_value, num_bins + 1)

#%%
plt.figure(figsize = (2,2))
plt.hist(amplitudes_p, bins = bin_edges, color = "grey", histtype='step',alpha = 1, lw=2)
plt.hist(amplitudes_b, bins = bin_edges, color = "yellowgreen", histtype='step',alpha = 1,lw=2)
plt.ylim(0, 30)
plt.xlim(0, 10)
sns.despine(top=True, right=True, left=False, bottom=False)

mean_p_amp = np.mean(amplitudes_p)
mean_b_amp = np.mean(amplitudes_b)

plt.axvline(mean_p_amp, color='dimgrey', linestyle='dashed', linewidth=2, label='Mean P')
plt.axvline(mean_b_amp, color='green', linestyle='dashed', linewidth=2, label='Mean B')


#%%
def mann_whitney_u_test(data1, data2, alpha=0.05):

    statistic, p_value = mannwhitneyu(data1, data2)
    significant = p_value < alpha

    # Output the results
    # print("Mann-Whitney U Test:")
    # print("Statistic:", statistic)
    print("p-value:", p_value)
    print("Is the difference significant?", significant)

    return statistic, p_value, significant

# Example usage:


print("amp")
mann_whitney_u_test(amplitudes_p, amplitudes_b)



print("pristine", round(np.mean(amplitudes_p), 2), round(np.std(amplitudes_p), 2))
print("biofoued", round(np.mean(amplitudes_b), 2), round(np.std(amplitudes_b), 2))


