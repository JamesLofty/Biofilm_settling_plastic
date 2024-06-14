#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 11:01:31 2024

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
from scipy.interpolate import make_interp_spline

# data = pd.read_excel("5 - modes/modePTFE_2x1_pos2_p.xlsx")
# label_=19
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePOM_2x1_pos2_p.xlsx")
# label_=9
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePS_2x1_pos2_p.xlsx")
# label_=16
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePTFE_2x1_pos1_p.xlsx")
# label_=30
# c ="red"
# ls = "-"

# data = pd.read_excel("5 - modes/modePTFE_2x1_pos3_p.xlsx")
# label_=29
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePTFE_2x1_pos3_b.xlsx")
# label_=28
# c ="green"
# ls = "-"

# data = pd.read_excel("5 - modes/modePTFE_2x1_pos3_b.xlsx")
# label_=4
# c ="green"
# ls = "-"

# data = pd.read_excel("5 - modes/modePTFE_1x1_pos2_b.xlsx")
# label_=17
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePOM_1x1_pos2_p.xlsx")
# label_=25
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePS_1x1_pos2_p.xlsx")
# label_=21
# c ="steelblue"
# ls = "-"


# data = pd.read_excel("5 - modes/modePTFE_05x05_pos1_p.xlsx")
# label_=13
# c ="orange"
# ls = "-"

# data = pd.read_excel("5 - modes/modePOM_1x1_pos1_p.xlsx")
# label_=29
# c ="steelblue"
# ls = "-"

# data = pd.read_excel("5 - modes/modePOM_1x1_pos1_p.xlsx")
# label_=30
# c ="steelblue"
# ls = "-"

data = pd.read_excel("5 - modes/modePTFE_2x1_pos1_b.xlsx")
label_=2
c ="steelblue"
ls = "-"

x = data["x_mid"]
y = data["y_mid"]
z = data["z_mid"]
wz = data["wz"]
wx = data["wx"]
wy = data["wy"]

wxwyx = ((wx**2) + (wy**2)) ** 0.5
label = data["label_mid"]
mode = data["mode"]

color_map = {1: 'steelblue', 2: 'orange', 3: 'green', 0: "red"}


#%%

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(projection='3d')

ax.plot(x[label == label_ ], y[label == label_ ], z[label == label_ ])
ax.scatter(x[label == label_ ], y[label == label_ ], z[label == label_ ],
                     c=wz[label == label_ ], s = 10)

ax.plot(x[label == label_ ], y[label == label_ ], c = "steelblue", zdir='z', lw = 1, zs=26, alpha = 0.4)
# ax.plot(x[label == label_ ], z[label == label_ ], c = "steelblue", zdir='y', lw = 1, zs=8, alpha = 0.4)
# ax.plot(y[label == label_ ], z[label == label_ ], c = "steelblue", zdir='x', lw = 1, zs=8, alpha = 0.4)

# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
ax.set_zlim(0, 25)

ax.set_xticks([-3, 0, 3])
ax.set_yticks([-3, 0, 3])

ax.grid(False)
ax.invert_zaxis()
ax.set_box_aspect(aspect=(0.6, 0.6, 2))
ax.view_init(10, 225)

# Adding color bar
mappable = cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(wz[label == label_])
# mappable.set_clim(vmin=0, vmax=35)  # Set your desired color bar limits

color_bar = plt.colorbar(mappable, ax=ax)
# color_bar.ax.set_aspect(0.1) 

plt.savefig("figures/3d tracj.svg", format="svg")

#%%
y_1 = wz[label == label_].values
x_1 = z[label == label_].values

# Data for the second subplot
y_2 = wxwyx[label == label_].values
x_2 = z[label == label_].values

# Define figure and axes
fig, axs = plt.subplots(1, 2, figsize=(4, 1))

# First subplot
axs[0].plot(x_1, y_1, color=c, linestyle='-')
axs[0].set_ylim(-10, 40)
axs[0].set_yticks([-10, 0, 20, 40])
axs[0].set_xlim(0, 20)
sns.despine(ax=axs[0], top=True, right=True, left=False, bottom=False)

# Second subplot
axs[1].plot(x_2, y_2, color=c, linestyle='-')
axs[1].set_ylim(0, 40)
axs[1].set_yticks([0, 20, 40])
axs[1].set_xlim(0, 20)
sns.despine(ax=axs[1], top=True, right=True, left=False, bottom=False)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig("figures/subplots.svg", format="svg")

# Show the plot
plt.show()

    
#%%

# for i in label.unique():
#     fig = plt.figure(figsize=(4,3))
#     ax = fig.add_subplot(projection='3d')
    
#     ax.plot(x[label == i ], y[label == i ], z[label == i ])
#     ax.scatter(x[label == i ], y[label == i ], z[label == i ],
#                           c=wz[label == i ],s = 10)
    
    
#     ax.set_xlabel('x (cm)')
#     ax.set_ylabel('y (cm)')
#     ax.set_zlabel('z (cm)')
    
#     ax.set_xticks([-3, 0, 3])
#     ax.set_yticks([-3, 0, 3])
#     ax.set_zlim(0, 25)
#     ax.set_title(f'Label {i}')  # Set the title as the label number

    
#     ax.grid(False)
#     ax.invert_zaxis()
#     ax.set_box_aspect(aspect=(0.6, 0.6, 2))
#     ax.view_init(10, 225)
    

#     # Adding color bar
#     mappable = cm.ScalarMappable(cmap=plt.cm.viridis)
#     mappable.set_array(wz[label == i])
#     color_bar = plt.colorbar(mappable, ax=ax)
# %%


# plt.figure(figsize=(4,0.5))

# y_1 = wz[label == label_].values  # Convert to numpy array with .values
# x_1 = z[label == label_].values   # Convert to numpy array with .values

# # Sort x_ and y_ based on the depth values
# # sorted_indices = np.argsort(x_)
# # x_p = x_[sorted_indices]
# # y_p = y_[sorted_indices]

# # x_smooth = np.linspace(x_p.min(), x_p.max(), 100)

# # spl = make_interp_spline(x_p, y_p)
# # y_smooth = spl(x_smooth)

# # plt.plot(x_smooth, y_smooth, color = c)
# plt.plot(x_1, y_1, color = c, linestyle = ls)
# # plt.scatter(x_, y_, c = wz[label == label_ ], s = 5, zorder = 10)

# plt.ylim(-5, 40)
# plt.yticks([0, 20, 40])
# plt.xlim(0, 20)
# # plt.ylabel("$w_vi$ (cm/s)")
# # plt.xlabel("Distance traveled (cm)")
# sns.despine(top=True, right=True, left=False, bottom=False)

# plt.savefig("figures/w_vi.svg", format="svg")
