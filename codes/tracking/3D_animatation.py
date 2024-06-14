#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:09:41 2024

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


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
          '#1a55FF', '#FF751a', '#3ca02c', '#b62728', '#aa67bd', '#cb564b', '#e277c2', '#4f7f7f', '#dbcd22', '#f7becf',
          '#cfb22c', '#47becf', '#4a55FF', '#FF455a', '#3cf02c', '#f72728', '#aa97bd', '#5b564b', '#e27c62', '#477a7f', 
          '#dbc22f', '#f7aecf', '#cb122c', '#17fecf', '#4a55FF']

def generate_color_map(num_shades):
    cmap = plt.cm.get_cmap('Greens', num_shades)
    hex_colors = [plt.cm.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    return hex_colors
num_shades = 37
red_colors = generate_color_map(num_shades)


def generate_color_map(num_shades):
    cmap = plt.cm.get_cmap('Greys', num_shades)
    hex_colors = [plt.cm.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    return hex_colors
num_shades = 37
green_colors = generate_color_map(num_shades)

# Create a ListedColormap
custom_cmap = mcolors.ListedColormap(colors)
reds = mcolors.ListedColormap(red_colors)
greens = mcolors.ListedColormap(green_colors)

#%%

%matplotlib qt5

data1 = pd.read_excel("5 - modes/modePS_2x1_pos2_p.xlsx")

# %matplotlib qt5
x_1 = data1["x_mid"]
y_1= data1["y_mid"]
z_1 = data1["z_mid"]
label1 = data1["label_mid"]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

# Plotting the initial point
ax.scatter(0, 0, 0, color='b', s=0)

# Create a scatter plot
scatter = ax.scatter([], [], [], color='b', s=0)

# Setting the axes properties
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.set_xticks([-10,-5, 0,5,10])
ax.set_yticks([-10,-5,0,5,10])

ax.set_zlim(0, 25)

ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")



ax.invert_zaxis()
ax.set_box_aspect(aspect=(5, 5, 10))
ax.view_init(5, 45)

# Define a color map with a unique color for each label
color_map1 = custom_cmap

# Function to update the scatter plot data
def update_scatter1(num, scatter, lines, x, y, z):
    # Clear all lines
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    
    # Clear the scatter plot
    scatter._offsets3d = ([], [], [])
    
    # Iterate over unique labels
    for i, label_value in enumerate(label1.unique()):
        # Get indices where label equals label_value
        indices = np.where(label1 == label_value)[0]
        # Get the available data for this label up to num
        available_data = min(num + 1, len(indices))
        # Plot corresponding (x, y, z) points with a unique color
        scatter._offsets3d = (np.append(scatter._offsets3d[0], x[indices[:available_data]]),
                              np.append(scatter._offsets3d[1], y[indices[:available_data]]),
                              np.append(scatter._offsets3d[2], z[indices[:available_data]]))
        scatter.set_color(color_map1(i))  # Set scatter color
        # Plot trajectory lines with a unique color
        lines[i].set_data(x[indices[:available_data]], y[indices[:available_data]])
        lines[i].set_3d_properties(z[indices[:available_data]])
        lines[i].set_color(color_map1(i))  # Set line color
        lines[i].set_linewidth(2)
    return scatter, *lines


# Initialize lines for each label
lines1 = [ax.plot([], [], [], color='k')[0] for _ in range(len(label1.unique()))]



# Creating the Animation object
ani1 = animation.FuncAnimation(
    fig, update_scatter1, len(x_1), fargs=(scatter, lines1, x_1, y_1, z_1), interval=50, repeat=True)





# Display the animation
plt.show()


#%%
# %matplotlib qt5


# data1 = pd.read_excel("5 - modes/modePTFE_2x1_pos2_p.xlsx")
# data2 = pd.read_excel("5 - modes/modePTFE_2x1_pos2_b.xlsx")

# # %matplotlib qt5
# x_1 = data1["x_mid"]
# y_1= data1["y_mid"]
# z_1 = data1["z_mid"]
# label1 = data1["label_mid"]

# x_2 = data2["x_mid"]
# y_2= data2["y_mid"]
# z_2 = data2["z_mid"]
# label2 = data2["label_mid"]


# # Attaching 3D axis to the figure
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection="3d")

# # Plotting the initial point
# ax.scatter(0, 0, 0, color='b', s=0)

# # Create a scatter plot
# scatter = ax.scatter([], [], [], color='b', s=0)

# # Setting the axes properties
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)

# ax.set_xticks([-10,-5, 0,5,10])
# ax.set_yticks([-10,-5,0,5,10])

# ax.set_zlim(0, 25)

# ax.set_xlabel("x (cm)")
# ax.set_ylabel("y (cm)")
# ax.set_zlabel("z (cm)")



# ax.invert_zaxis()
# ax.set_box_aspect(aspect=(5, 5, 10))
# ax.view_init(5, 45)

# # Define a color map with a unique color for each label
# color_map1 = reds
# color_map2 = greens

# # Function to update the scatter plot data
# def update_scatter1(num, scatter, lines, x, y, z):
#     # Clear all lines
#     for line in lines:
#         line.set_data([], [])
#         line.set_3d_properties([])
    
#     # Clear the scatter plot
#     scatter._offsets3d = ([], [], [])
    
#     # Iterate over unique labels
#     for i, label_value in enumerate(label1.unique()):
#         # Get indices where label equals label_value
#         indices = np.where(label1 == label_value)[0]
#         # Get the available data for this label up to num
#         available_data = min(num + 1, len(indices))
#         # Plot corresponding (x, y, z) points with a unique color
#         scatter._offsets3d = (np.append(scatter._offsets3d[0], x[indices[:available_data]]),
#                               np.append(scatter._offsets3d[1], y[indices[:available_data]]),
#                               np.append(scatter._offsets3d[2], z[indices[:available_data]]))
#         scatter.set_color(color_map1(i))  # Set scatter color
#         # Plot trajectory lines with a unique color
#         lines[i].set_data(x[indices[:available_data]], y[indices[:available_data]])
#         lines[i].set_3d_properties(z[indices[:available_data]])
#         lines[i].set_color(color_map1(i))  # Set line color
#         lines[i].set_linewidth(2)
#     return scatter, *lines

# def update_scatter2(num, scatter, lines, x, y, z):
#     # Clear all lines
#     for line in lines:
#         line.set_data([], [])
#         line.set_3d_properties([])
    
#     # Clear the scatter plot
#     scatter._offsets3d = ([], [], [])
    
#     # Iterate over unique labels
#     for i, label_value in enumerate(label2.unique()):
#         # Get indices where label equals label_value
#         indices = np.where(label2 == label_value)[0]
#         # Get the available data for this label up to num
#         available_data = min(num + 1, len(indices))
#         # Plot corresponding (x, y, z) points with a unique color
#         scatter._offsets3d = (np.append(scatter._offsets3d[0], x[indices[:available_data]]),
#                               np.append(scatter._offsets3d[1], y[indices[:available_data]]),
#                               np.append(scatter._offsets3d[2], z[indices[:available_data]]))
#         scatter.set_color(color_map2(i))  # Set scatter color
#         # Plot trajectory lines with a unique color
#         lines[i].set_data(x[indices[:available_data]], y[indices[:available_data]])
#         lines[i].set_3d_properties(z[indices[:available_data]])
#         lines[i].set_color(color_map2(i))  # Set line color
#         lines[i].set_linewidth(2)
#     return scatter, *lines

# # Initialize lines for each label
# lines1 = [ax.plot([], [], [], color='k')[0] for _ in range(len(label1.unique()))]

# lines2 = [ax.plot([], [], [], color='k')[0] for _ in range(len(label2.unique()))]


# # Creating the Animation object
# ani1 = animation.FuncAnimation(
#     fig, update_scatter1, len(x_1), fargs=(scatter, lines1, x_1, y_1, z_1), interval=50, repeat=True)


# ani2 = animation.FuncAnimation(
#     fig, update_scatter2, len(x_2), fargs=(scatter, lines2, x_2, y_2, z_2), interval=50, repeat=True)




# # Display the animation
# plt.show()



#%%
# data = pd.read_excel("5 - modes/modePS_2x1_pos2_p.xlsx")
data = pd.read_excel("5 - modes/modePTFE_2x1_pos2_p.xlsx")
label_=19

x = data["x_mid"]
y = data["y_mid"]
z = data["z_mid"]
wz = data["wz"]
label = data["label_mid"]
mode = data["mode"]


%matplotlib qt5

what = label

# chaotic = 28 pos 3

x_1 = x[label == what]
y_1= y[label == what]
z_1 = z[label == what]

# Attaching 3D axis to the figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

# Plotting the initial point
ax.scatter(0, 0, 0, color='b', s=0.1)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.set_xticks([-10,-5, 0,5,10])
ax.set_yticks([-10,-5,0,5,10])

ax.set_zlim(0, 25)

ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")

# Create a scatter plot
scatter = ax.scatter([], [], [], color='b', s=4)

# Create a line plot
line, = ax.plot([], [], [], color='b', linewidth = 2)

# Setting the axes properties
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 25)

ax.invert_zaxis()
ax.set_box_aspect(aspect=(2, 2, 4))
ax.view_init(5, 45)

# Function to update the scatter plot data
def update_scatter(num, scatter, line, x, y, z):
    scatter._offsets3d = (x[:num], y[:num], z[:num])
    if num > 1:
        line.set_data(x[:num], y[:num])
        line.set_3d_properties(z[:num])
    return scatter, line

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_scatter, len(x_1), fargs=(scatter, line, x_1, y_1, z_1), interval=50, repeat = True)

plt.show()