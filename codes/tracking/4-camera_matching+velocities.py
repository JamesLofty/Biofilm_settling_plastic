#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:28:22 2023

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


#%%

plastic = "POM"
size = "1x1"
pos = "pos2"
plastic_type = "b"

cam1_file= "3-labelled_coords/labeled" + plastic +"_"+size+"_"+pos+"_cam1_" + plastic_type + ".xlsx"
cam2_file= "3-labelled_coords/labeled" + plastic +"_"+size+"_"+pos+"_cam2_" + plastic_type + ".xlsx"


#%%
cam1_coords = pd.read_excel(cam1_file)
cam2_coords = pd.read_excel(cam2_file)

merged_data = pd.merge(cam2_coords, cam1_coords, on='tp', suffixes=('_cam2', '_cam1'))

xp_cam1 = np.array(merged_data["xp_cam1"])
yp_cam1 = np.array(merged_data["yp_cam1"])

xp_cam2 = np.array(merged_data["xp_cam2"])
yp_cam2 =  np.array(merged_data["yp_cam2"])

area_cam1 = np.array(merged_data["area_cam1"])
area_cam2 = np.array(merged_data["area_cam2"])

area = (area_cam1 + area_cam2) / 2

######################################################
x = xp_cam2
y = xp_cam1
if len(yp_cam1) == len(yp_cam2):
    z = [(x + y) / 2 for x, y in zip(yp_cam1, yp_cam2)]
######################################################

tp = np.array(merged_data["tp"])
label = np.array(merged_data["label_cam1"])

#%%
"velocities"


t = np.diff(np.array(tp))

wy = np.diff(y) / t
wx = np.diff(x) / t
wz = np.diff(z) / t

#%%
# %matplotlib qt5
# %matplotlib inline

def calculate_midpoints(data):
    midpoints = [(data[i] + data[i+1]) / 2 for i in range(len(data) - 1)]
    return np.array(midpoints)

tp_mid = calculate_midpoints(tp)
label_mid = calculate_midpoints(label)
x_mid = calculate_midpoints(x)
y_mid = calculate_midpoints(y)
z_mid = calculate_midpoints(z)
area = calculate_midpoints(area)


#%%
"""as label_mid now has annoying decimals, we remove the decimals"""
tp_mid = tp_mid[label_mid % 1 == 0]
x_mid = x_mid[label_mid % 1 == 0]
y_mid = y_mid[label_mid % 1 == 0]
z_mid = z_mid[label_mid % 1 == 0]
wy = wy[label_mid % 1 == 0]
wx = wx[label_mid % 1 == 0]
wz = wz[label_mid % 1 == 0]
area_mid = area[label_mid % 1 == 0]

label_mid = label_mid[label_mid % 1 == 0]

#%%
"""lets make sure everything is normalised"""

x_norm = []
y_norm = []
z_norm = []
tp_list_norm = []
label_list_norm = []

for i in np.unique(label_mid):
    x_norm1 = x_mid[label_mid == i] - x_mid[label_mid == i][0:1]
    y_norm1 = y_mid[label_mid == i] - y_mid[label_mid == i][0:1]
    z_norm1 = z_mid[label_mid == i] - np.min(z_mid[label_mid == i])
    tp_norm = tp_mid[label_mid == i] - tp_mid[label_mid == i][0:1]
    
    x_norm.extend(x_norm1)
    y_norm.extend(y_norm1)
    z_norm.extend(z_norm1)
    tp_list_norm.extend(tp_norm)
    
#%%
x_norm = np.array(x_norm)
y_norm = np.array(y_norm)
z_norm = np.array(z_norm)
# label_list_norm = np.array(label_list_norm)


# for i in np.unique(label_mid):
#     fig = plt.figure(figsize=(4,4))
#     ax = fig.add_subplot(projection='3d')
    
#     ax.scatter(x_norm[label_mid==i], y_norm[label_mid==i], z_norm[label_mid==i], 
#                 c =label_mid[label_mid==i], s = 3)
    
#     ax.set_xlim(-10, 10)
#     ax.set_ylim(-10, 10)
#     ax.set_zlim(0, 30)
    
#     ax.invert_zaxis()
#     ax.set_box_aspect(aspect=(2, 2, 4))
#     ax.view_init(5, 45)

x_mid = x_norm
y_mid = y_norm
z_mid = z_norm
tp_mid =  np.array(tp_list_norm)
label_mid = label_mid
area_mid = np.array(area_mid)

#%%

# Set the threshold for filtering based on z-scores
z_score_threshold = 1000
filered_x_mid_list = []
filered_y_mid_list = []
filered_z_mid_list = []
filered_tp_mid_list = []
filered_label_mid_list = []
filered_wz_mid_list = []
filered_wx_mid_list = []
filered_wy_mid_list = []
filtered_area_mid_list = []

for i in np.unique(label_mid):
    z_score_wz = (wz[label_mid == i] - np.mean(wz[label_mid == i])) / np.std(wz[label_mid == i])
    z_score_wx = (wx[label_mid == i] - np.mean(wx[label_mid == i])) / np.std(wx[label_mid == i])
    z_score_wy = (wy[label_mid == i] - np.mean(wy[label_mid == i])) / np.std(wy[label_mid == i])

    z_score_condition = (abs(z_score_wz)  <= abs(z_score_threshold))
    # z_score_condition = (abs(z_score_wx) <= abs(z_score_threshold)).all() and (abs(z_score_wy) <= abs(z_score_threshold)).all()

    combined_condition = z_score_condition

    filered_x_mid_list.append(x_mid[label_mid == i][combined_condition])
    filered_y_mid_list.append(y_mid[label_mid == i][combined_condition])
    filered_z_mid_list.append(z_mid[label_mid == i][combined_condition])
    filered_tp_mid_list.append(tp_mid[label_mid == i][combined_condition])
    filered_label_mid_list.append(label_mid[label_mid == i][combined_condition])
    filered_wz_mid_list.append(wz[label_mid == i][combined_condition])
    filered_wx_mid_list.append(wx[label_mid == i][combined_condition])
    filered_wy_mid_list.append(wy[label_mid == i][combined_condition])
    filtered_area_mid_list.append(area_mid[label_mid == i][combined_condition])
    
x_mid = np.concatenate(filered_x_mid_list)
y_mid = np.concatenate(filered_y_mid_list)
z_mid = np.concatenate(filered_z_mid_list)
tp_mid = np.concatenate(filered_tp_mid_list)
label_mid = np.concatenate(filered_label_mid_list)
wx = np.concatenate(filered_wx_mid_list)
wy = np.concatenate(filered_wy_mid_list)
wz= np.concatenate(filered_wz_mid_list)
area = np.concatenate(filtered_area_mid_list)

#%%
    
#%%
# for i in np.unique(label_mid):

#     fig = plt.figure(figsize=(8,5))
    
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.plot(x_mid[label_mid==i], y_mid[label_mid==i], z_mid[label_mid==i])
#     ax1.set_xlabel('x (cm)')
#     ax1.set_ylabel('y (cm)')
#     ax1.set_zlabel('z (cm)')
#     ax1.set_xlim(-15,15)
#     ax1.set_ylim(-15,15)
#     ax1.set_zlim(0, 30)
#     ax1.dist = 9
#     ax1.invert_zaxis()
#     ax1.set_box_aspect(aspect=(1, 1, 2))
#     ax1.set_title(f'tracjectory {i}')
    
#     ax2 = fig.add_subplot(132)
#     # ax2.plot(xp_cam1[label == i], yp_cam1[label == i])
#     ax2.scatter(x_mid[label_mid==i], z_mid[label_mid==i], c=wz[label_mid == i])
#     ax2.set_xlabel('x (cm)')
#     ax2.set_ylabel('z (cm)')
#     ax2.set_ylim(30, 0)
#     ax2.set_xlim(-15,15)
#     sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)

#     ax3 = fig.add_subplot(133)
#     # ax3.plot(xp_cam2[label == i], yp_cam2[label == i])
#     scatter = ax3.scatter(y_mid[label_mid==i], z_mid[label_mid==i], c=wz[label_mid == i])
#     ax3.set_xlabel('y (cm)')
#     ax3.set_ylabel('z (cm)')
#     ax3.set_ylim(30, 0)
#     ax3.set_xlim(-15,15)
#     sns.despine(ax=ax3, top=True, right=True, left=False, bottom=False)
#     colorbar = plt.colorbar(scatter, ax=ax3)
#     colorbar.set_label('z Velocity (cm/s)')  # Set a label for the color bar
    
#     plt.tight_layout()

# %%
#
    
#%%
"""3d plot"""
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection='3d')

ax.scatter(x_mid[label_mid == 2], y_mid[label_mid == 2], z_mid[label_mid == 2], 
           c = label_mid[label_mid == 2], s = 2)


horizontal_y_values = [5, 10, 15, 20, 25]

# for y in horizontal_y_values:
#     ax.plot([0, 20], [20, 0], [y, y], color='red', linestyle='--')

ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_zlabel('z (cm)')

# ax.set_xlim(-15, 15)
# ax.set_ylim(-15, 15)
# ax.set_zlim(0, 30)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 30)

ax.invert_zaxis()
ax.set_box_aspect(aspect=(2, 2, 4))
ax.view_init(5, 45)

# plt.savefig('3dplot.svg', format='svg')

#%%


#%%
"vertical velocity"
plt.figure(figsize=(5,5))
plt.scatter(wy, z_mid, c = label_mid)
# plt.scatter(w_cam2, yp_cam2[1:])
plt.gca().invert_yaxis()

# plt.xlim(-10, 25)
# plt.ylim(45, 10)

plt.axvline(x = np.mean(wy), color='blue', linestyle='--', label='Y = 15')
plt.axvline(x = np.percentile(wy, 1), color='blue', linestyle='--', label='Y = 15')
plt.axvline(x = np.percentile(wy, 99), color='blue', linestyle='--', label='Y = 15')


plt.axhline(y=5, color='blue', linestyle='--', label='Y = 15')
plt.axhline(y=10, color='blue', linestyle='--', label='Y = 15')
plt.axhline(y=15, color='blue', linestyle='--', label='Y = 20')
plt.axhline(y=20, color='blue', linestyle='--', label='Y = 25')
plt.axhline(y=25, color='blue', linestyle='--', label='Y = 30')
plt.axhline(y=30, color='blue', linestyle='--', label='Y = 35')

sns.despine(top=True, right=True, left=False, bottom=False)

plt.xlabel("z Velocity (cm/s)")
plt.ylabel("Depth (cm")

#%%

condition1 = (z_mid > 0) & (z_mid < 5)
condition2 = (z_mid > 5) & (z_mid < 10)
condition3 = (z_mid > 10) & (z_mid < 15)
condition4 = (z_mid > 15) & (z_mid < 20)
condition5 = (z_mid > 20) & (z_mid < 25)

fig, axes = plt.subplots(5, 1, figsize=(5, 9))

# Define your conditions
conditions = [condition1, condition2, condition3, condition4, condition5]
depth_ranges = ["0-5 depth", "5-10 depth", "10-15 depth", "15-20 depth", "20-25 depth"]

min_value = np.min(wz)
max_value = np.max(wz)
bin_width = 1
num_bins = int((max_value - min_value) / bin_width)
bin_edges = np.linspace(min_value, max_value, num_bins + 1)

# Plot histograms for each condition on separate subplots
for i, condition in enumerate(conditions):
    axes[i].hist(wz[condition], bins = bin_edges, color = "steelblue")
    
    # sns.kdeplot(wz_cam1[condition], color="steelblue", ax=axes[i])
    # sns.kdeplot(wz_cam2[condition], color="steelblue", ax=axes[i])


    axes[i].set_title(depth_ranges[i])
    # axes[i].set_xlim(-10, 15) 
    # axes[i].set_ylim(0, 60) 
    
    avg = np.mean(wz[condition])
    med = np.median(wz[condition])
    percentile_10 = np.percentile(wz[condition], 10)
    percentile_90 = np.percentile(wz[condition], 90)

    axes[i].axvline(avg, color='r', linestyle='dashed', linewidth=2, label='Average')
    # axes[i].axvline(med, color='g', linestyle='dashed', linewidth=2, label='Average')

    # axes[i].axvline(percentile_10, color='b', linestyle='dashed', linewidth=2, label='10th Percentile')
    # axes[i].axvline(percentile_90, color='b', linestyle='dashed', linewidth=2, label='90th Percentile')
    
    axes[i].set_xlabel("y velocity (cm/s)")
    axes[i].set_ylabel("Frequency")
    
    sns.despine(ax=axes[i], top=True, right=True, left=False, bottom=False)

    plt.tight_layout()
    
    



#%%

# abs_x = np.sqrt(xp_list_mid1**2 + xp_list_mid2**2)

# results = pd.DataFrame(list(zip(wy, wx, yp_list_mid1, yp_list_mid2, xp_list_mid1,  xp_list_mid2, abs_x, label_mid, tp_mid)),
#                        columns=['wy', "wx", 'yp_list_mid1','yp_list_mid2','xp_list_mid1', "xp_list_mid2", "abs_x", "label", "tp_mid"])

results = pd.DataFrame(list(zip(wz,
                                wx,
                                wy, 
                                x_mid, 
                                y_mid,
                                z_mid,
                                label_mid, 
                                tp_mid)), 
                       columns= ["wz", "wx", "wy", "x_mid", "y_mid", "z_mid", "label_mid", "tp_mid"])
                                

    
if "pos1" in cam1_file:
    pos = "pos1_"
if "pos2" in cam1_file:
    pos = "pos2_"
if "pos3" in cam1_file:
    pos = "pos3_"

if "2x1" in cam1_file:
    size = "2x1_"
if "1x1" in cam1_file:
    size = "1x1_"
if "05x05" in cam1_file:
    size = "05x05_"
    
if "PTFE" in cam1_file:
    plastic = "PTFE_"
if "POM" in cam1_file:
    plastic = "POM_"
if "PS" in cam1_file:
    plastic = "PS_"
if "PA" in cam1_file:
    plastic = "PA_"

results.to_excel("4 - velocities/" + "velocities" + plastic + size + pos + plastic_type + ".xlsx")
print("all data saved wooo")



