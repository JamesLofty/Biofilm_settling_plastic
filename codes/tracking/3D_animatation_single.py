#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:04:26 2024

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


# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
#           '#1a55FF', '#FF751a', '#3ca02c', '#b62728', '#aa67bd', '#cb564b', '#e277c2', '#4f7f7f', '#dbcd22', '#f7becf',
#           '#cfb22c', '#47becf', '#4a55FF', '#FF455a', '#3cf02c', '#f72728', '#aa97bd', '#5b564b', '#e27c62', '#477a7f', 
#           '#dbc22f', '#f7aecf', '#cb122c', '#17fecf', '#4a55FF']

# def generate_color_map(num_shades):
#     cmap = plt.cm.get_cmap('Greens', num_shades)
#     hex_colors = [plt.cm.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
#     return hex_colors
# num_shades = 37
# red_colors = generate_color_map(num_shades)


# def generate_color_map(num_shades):
#     cmap = plt.cm.get_cmap('Greys', num_shades)
#     hex_colors = [plt.cm.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
#     return hex_colors
# num_shades = 37
# green_colors = generate_color_map(num_shades)

# # Create a ListedColormap
# custom_cmap = mcolors.ListedColormap(colors)
# reds = mcolors.ListedColormap(red_colors)
# greens = mcolors.ListedColormap(green_colors)

#%%
# Read data

# Read data
%matplotlib qt5
# 
# Read data

data1 = pd.read_excel("5 - modes/modePTFE_2x1_pos2_p.xlsx")
label_1 = 19
data1_avg = np.mean(data1["wz"][data1["label_mid"]==label_1])

data2 = pd.read_excel("5 - modes/modePTFE_2x1_pos3_b.xlsx")
label_2 = 28
data2_avg = np.mean(data2["wz"][data2["label_mid"]==label_2])

data3 = pd.read_excel("5 - modes/modePTFE_2x1_pos3_p.xlsx")
label_3 = 29
data3_avg = np.mean(data3["wz"][data3["label_mid"]==label_3])


data5 = pd.read_excel("5 - modes/modePTFE_2x1_pos1_b.xlsx")
label_5 = 2
data5_avg = np.mean(data5["wz"][data5["label_mid"]==label_5])


data4 = pd.read_excel("5 - modes/modePTFE_2x1_pos1_p.xlsx")
label_4 = 30
data4_avg = np.mean(data4["wz"][data4["label_mid"]==label_4]) - 5


# Extracting data for each label
x_1 = data1["x_mid"][data1["label_mid"] == label_1]
y_1 = data1["y_mid"][data1["label_mid"] == label_1]
z_1 = data1["z_mid"][data1["label_mid"] == label_1]

x_2 = data2["x_mid"][data2["label_mid"] == label_2]
y_2 = data2["y_mid"][data2["label_mid"] == label_2]
z_2 = data2["z_mid"][data2["label_mid"] == label_2]

x_3 = data3["x_mid"][data3["label_mid"] == label_3]
y_3 = data3["y_mid"][data3["label_mid"] == label_3]
z_3 = data3["z_mid"][data3["label_mid"] == label_3]

x_4 = data4["x_mid"][data4["label_mid"] == label_4]
y_4 = data4["y_mid"][data4["label_mid"] == label_4]
z_4 = data4["z_mid"][data4["label_mid"] == label_4]

x_5 = data5["x_mid"][data5["label_mid"] == label_5]
y_5 = data5["y_mid"][data5["label_mid"] == label_5]
z_5 = data5["z_mid"][data5["label_mid"] == label_5]

# Define colors for each subplot
colors = ['r', 'g', 'b', 'm', 'orange']

# Initialize the figure and subplots with bigger size
fig, axs = plt.subplots(1, 5, figsize = (30,30), subplot_kw={'projection': '3d'})

# Function to update the scatter plot data for each subplot
def update_scatter(num, scatter, line, x, y, z):
    scatter._offsets3d = (x[:num], y[:num], z[:num])
    if num > 1:
        line.set_data(x[:num], y[:num])
        line.set_3d_properties(z[:num])
    return scatter, line

# Create scatter and line objects for each subplot
scatters = []
lines = []
for ax, x, y, z, color in zip(axs, [x_1, x_2, x_3, x_4, x_5], [y_1, y_2, y_3, y_4, y_5], [z_1, z_2, z_3, z_4, z_5], colors):
    ax.scatter(0, 0, 0, color='b', s=0.1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 25)
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_yticks([-10,-5,0,5,10])

    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("$y$ (cm)")
    ax.set_zlabel("$z$ (cm)")
    ax.invert_zaxis()
    ax.set_box_aspect(aspect=(3, 3, 5))
    ax.view_init(5, 45)
    scatter = ax.scatter([], [], [], color=color, s=4)
    line, = ax.plot([], [], [], color=color, linewidth=2)
    scatters.append(scatter)
    lines.append(line)

# Apply tight layout
plt.tight_layout()

# Function to update all subplots
def update_all(num):
    artists = []
    for scatter, line, x, y, z in zip(scatters, lines, [x_1, x_2, x_3, x_4, x_5], [y_1, y_2, y_3, y_4, y_5], [z_1, z_2, z_3, z_4, z_5]):
        n = num % len(x)
        artist = update_scatter(n, scatter, line, x, y, z)
        artists.append(artist)
    return artists

# Creating the Animation object
ani = animation.FuncAnimation(fig, update_all, frames=None, interval=50, repeat=True)

plt.show()


#%%

%matplotlib inline
_1 = pd.read_excel("5 - modes/modePTFE_2x1_pos1_b.xlsx")
_2 = pd.read_excel("5 - modes/modePTFE_2x1_pos2_b.xlsx")
_3 = pd.read_excel("5 - modes/modePTFE_2x1_pos3_b.xlsx")

_2["label_mid"] = _2["label_mid"] + 100
_3["label_mid"] = _3["label_mid"] + 1000

dataaaa = pd.concat([_1, _2, _3])


average_velocity = dataaaa.groupby('label_mid')['wz'].mean()
average = np.mean(average_velocity)


# colors = ['r', 'g', 'b', 'orange', 'm']
lw1 = 2
lw2 = 4
plt.figure(figsize = (6,4))
plt.hist(average_velocity, density = True, bins = 20, alpha =0.7)
plt.axvline(data1_avg, color='r', linestyle='-', lw = lw1)
plt.axvline(data2_avg, color='g', linestyle='-', lw = lw1)
plt.axvline(data3_avg, color='b', linestyle='-', lw = lw1)
plt.axvline(data4_avg, color='m', linestyle='-', lw = lw1)
plt.axvline(data5_avg, color='orange', linestyle='-', lw = lw1)
plt.axvline(average, color='black', linestyle = '--', lw = lw2)
sns.despine( top=True, right=True, left=False, bottom=False)
# plt.yscale('log')

plt.xlabel("Vertical velocity (cm/s)")
plt.ylabel("Frequency")



#%%
# data = pd.read_excel("5 - modes/modePTFE_2x1_pos3_b.xlsx")
# label_ = 28

# x = data["x_mid"]
# y = data["y_mid"]
# z = data["z_mid"]
# wz = data["wz"]
# label = data["label_mid"]
# mode = data["mode"]

# %matplotlib qt5

# what = label_

# # chaotic = 28 pos 3

# x_1 = x[label == what]
# y_1= y[label == what]
# z_1 = z[label == what]

# # Attaching 3D axis to the figure
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection="3d")

# # Plotting the initial point
# ax.scatter(0, 0, 0, color='b', s=0.1)

# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)

# ax.set_xticks([-10,-5, 0,5,10])
# ax.set_yticks([-10,-5,0,5,10])

# ax.set_zlim(0, 25)

# ax.set_xlabel("x (cm)")
# ax.set_ylabel("y (cm)")
# ax.set_zlabel("z (cm)")

# # Create a scatter plot
# scatter = ax.scatter([], [], [], color='b', s=4)

# # Create a line plot
# line, = ax.plot([], [], [], color='b', linewidth = 2)

# # Setting the axes properties
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_zlim(0, 25)

# ax.invert_zaxis()
# ax.set_box_aspect(aspect=(2, 2, 4))
# ax.view_init(5, 45)

# # Function to update the scatter plot data
# def update_scatter(num, scatter, line, x, y, z):
#     scatter._offsets3d = (x[:num], y[:num], z[:num])
#     if num > 1:
#         line.set_data(x[:num], y[:num])
#         line.set_3d_properties(z[:num])
#     return scatter, line

# # Creating the Animation object
# ani = animation.FuncAnimation(
#     fig, update_scatter, len(x_1), fargs=(scatter, line, x_1, y_1, z_1), interval=50, repeat = True)

# plt.show()