#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:08:29 2024

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.stats
from scipy.stats import mannwhitneyu

def calculate_net_horizontal_motion(x_coordinates, y_coordinates):
    x_initial = x_coordinates[0]  # Initial x-coordinate
    x_final = x_coordinates[-1]   # Final x-coordinate
    y_initial = y_coordinates[0]  # Initial y-coordinate
    y_final = y_coordinates[-1]   # Final y-coordinate
    return np.sqrt((x_final - x_initial)**2 + (y_final - y_initial)**2)


plastic = "PTFE"
size = "_2x1_"


if size == "_05x05_": 
    data1_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1" + "_p.xlsx")
    data2_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1" + "_p.xlsx")
    data3_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1" + "_p.xlsx")

    data1_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1" + "_b.xlsx")
    data2_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1" + "_b.xlsx")
    data3_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1" + "_b.xlsx")
    lim = -3, 3
    ticks = [-3,-1.5, 0, 1.5 ,3]
else: 

    data1_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos2" + "_p.xlsx")
    data2_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos2" + "_p.xlsx")
    data3_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos2" + "_p.xlsx")
    
    data1_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos2" + "_b.xlsx")
    data2_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos2" + "_b.xlsx")
    data3_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos2" + "_b.xlsx")
    lim = -6,6
    ticks = [-5, 0,5]

data2_p["label_mid"] = data2_p["label_mid"] + 100
data3_p["label_mid"] = data3_p["label_mid"] + 1000

data2_b["label_mid"] = data2_b["label_mid"] + 100
data3_b["label_mid"] = data3_b["label_mid"] + 1000

data_p = pd.concat([data1_p, data2_p, data3_p], ignore_index=True)
data_b = pd.concat([data1_b, data2_b, data3_b], ignore_index=True)

data_p["data"] = "data_p"
data_b["data"] = "data_b"


depth1 = 0
depth2 = 20

data_p = data_p[(data_p['z_mid'] >= depth1) & (data_p['z_mid'] <= depth2)]
data_b = data_b[(data_b['z_mid'] >= depth1) & (data_b['z_mid'] <= depth2)]


#%%
plt.figure(figsize=(2, 2))


x_p = np.array(data_p["x_mid"])
y_p = np.array(data_p["y_mid"])
z_p = np.array(data_p["z_mid"])
wz_p = np.array(data_p["wz"])
wx_p = np.array(data_p["wx"])
label_p = data_p["label_mid"]

for i in label_p.unique(): 

    if (x_p[label_p == i][0] == 0) and (y_p[label_p == i][0] == 0):
        plt.plot(x_p[label_p == i], y_p[label_p == i], alpha=0.3, color="dimgrey")


#%%
max_xy_b = []
largest_list_b = []
last_xy_b = []
last_list_b = []

x_b = np.array(data_b["x_mid"])
y_b = np.array(data_b["y_mid"])
z_b = np.array(data_b["z_mid"])
wz_b = np.array(data_b["wz"])
wx_b = np.array(data_b["wx"])
label_b = data_b["label_mid"]

for i in label_b.unique(): 

    if (x_b[label_b == i][0] == 0) and (y_b[label_b == i][0] == 0):
        plt.plot(x_b[label_b == i], y_b[label_b == i], alpha=0.3, color="yellowgreen")

#%%
# Assuming label_p and label_b are unique identifiers for each particle
label_p = data_p["label_mid"]
x_coords_p = np.array(data_p["x_mid"])
y_coords_p = np.array(data_p["y_mid"])

label_b = data_b["label_mid"]
x_coords_b = np.array(data_b["x_mid"])
y_coords_b = np.array(data_b["y_mid"])

# Calculate net horizontal motion for each particle
net_horizontal_motion_p = [calculate_net_horizontal_motion(x_coords_p[label_p == i], y_coords_p[label_p == i]) for i in label_p.unique()]
net_horizontal_motion_b = [calculate_net_horizontal_motion(x_coords_b[label_b == i], y_coords_b[label_b == i]) for i in label_b.unique()]

# Compute the average horizontal drift
average_horizontal_drift_p = np.mean(net_horizontal_motion_p)
average_horizontal_drift_b = np.mean(net_horizontal_motion_b)

std_horizontal_drift_p = np.std(net_horizontal_motion_p)
std_horizontal_drift_b = np.std(net_horizontal_motion_b)

print("Average Horizontal Drift for 'p' particles: {:.2f}".format(average_horizontal_drift_p))
print("std Horizontal Drift for 'p' particles: {:.2f}".format(std_horizontal_drift_p))


print("Average Horizontal Drift for 'b' particles: {:.2f}".format(average_horizontal_drift_b))
print("std Horizontal Drift for 'b' particles: {:.2f}".format(std_horizontal_drift_b))


# Plot the circles for both datasets
circle_p = plt.Circle((0, 0), average_horizontal_drift_p, color='dimgrey', fill=False, zorder=8, lw=2, alpha = 0.8)
plt.gca().add_patch(circle_p)
circle_p_std = plt.Circle((0, 0), average_horizontal_drift_p - std_horizontal_drift_p, color='dimgrey', ls=":", fill=False, zorder=8, lw=2, alpha = 0.8)
plt.gca().add_patch(circle_p_std)
circle_p_std_alpha = plt.Circle((0, 0), average_horizontal_drift_p + std_horizontal_drift_p, color='dimgrey', ls=":", fill=False, zorder=8, lw=2, alpha = 0.8)
plt.gca().add_patch(circle_p_std_alpha)

circle_b = plt.Circle((0, 0), average_horizontal_drift_b, color='green', fill=False, zorder=8, lw=2, alpha = 0.8)
plt.gca().add_patch(circle_b)
circle_b_std = plt.Circle((0, 0), average_horizontal_drift_b - std_horizontal_drift_b, color='green', ls=":", fill=False, zorder=8, lw=2, alpha = 0.8)
plt.gca().add_patch(circle_b_std)
circle_b_std_alpha = plt.Circle((0, 0), average_horizontal_drift_b + std_horizontal_drift_b, color='green', ls=":", fill=False, zorder=8, lw=2, alpha = 0.8)
plt.gca().add_patch(circle_b_std_alpha)



plt.yticks(ticks)
plt.xticks(ticks)


plt.xlim(lim)
plt.ylim(lim)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig("figures/xy_combined.svg")
plt.show()
    
#%%
bin_edges = np.arange(min(min(net_horizontal_motion_p), min(net_horizontal_motion_p)), max(max(net_horizontal_motion_p), max(net_horizontal_motion_p)) + 2,0.25)

# Plot the histograms
plt.figure(figsize=(2, 1))
plt.hist(net_horizontal_motion_p, bins=bin_edges, color="grey", alpha=0.5, density=True)
plt.hist(net_horizontal_motion_b,  bins=bin_edges, color="yellowgreen", alpha=0.5, density=True)

plt.hist(net_horizontal_motion_p, histtype='step',bins=bin_edges, color="grey", density=True)
plt.hist(net_horizontal_motion_b, histtype='step', bins=bin_edges, color="yellowgreen", density=True)



# Plot the mean lines
plt.vlines(np.mean(net_horizontal_motion_p), 0, 4, color="dimgrey")
plt.vlines(np.mean(net_horizontal_motion_b), 0, 4, color="green")


# Remove top and right spines
sns.despine(top=True, right=True, left=False, bottom=False)

plt.xlim(0,5)
plt.ylim(0,2)
plt.xticks([0, 2.5, 5])

# Display the means and standard deviations
# print(round(np.mean(last_list_p), 3), round(np.std(last_list_p), 3))
# print(round(np.mean(last_list_b), 3), round(np.std(last_list_b), 3))

# Show the plot
plt.savefig("figures/xy_combined_his.svg")


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


print("horiiizzz drift")
mann_whitney_u_test(net_horizontal_motion_p, net_horizontal_motion_b)






