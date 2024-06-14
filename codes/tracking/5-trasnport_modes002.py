
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:43:23 2023

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plastic = "POM"
size = "05x05"
pos = "pos1"
plastic_type = "b"

file1 = "4 - velocities/velocities" + plastic + "_" + size + "_" + pos + "_" + plastic_type + ".xlsx"
file2 = "modes_update/" + plastic + "_" + size + "_" + pos + "_" + plastic_type + ".xlsx"

data = pd.read_excel(file1)
pos = pd.read_excel(file2)


    #%%
# Function to assign modes based on coordinates
def assign_mode(x, start, end):
    if x < start:
        return 2
    elif start <= x <= end:
        return 0
    else:
        return 1

data['mode'] = data.apply(lambda row: assign_mode(row['z_mid'], pos.at[row['label_mid'], 'start'], pos.at[row['label_mid'], 'end']), axis=1)

bad = pos[pos['keep'] == 'n']['label']
bad = np.array(bad)

# if "PTFE" in file1:

#     def assign_mode_chaotic(x, start, end):
#         if x < start:
#             return 2
#         elif start <= x <= end:
#             return 0
#         else:
#             return 1
    
#     data['mode_chaotic'] = data.apply(lambda row: assign_mode_chaotic(row['z_mid'], pos.at[row['label_mid'], 'start2'], pos.at[row['label_mid'], 'end2']), axis=1)



#%%

for label in data['label_mid'].unique(): 
    label_data = data[data['label_mid'] == label]
    fig, ax = plt.subplots(1, 3, figsize=(10, 6)) 

    # Create a list of unique modes for coloring within this label
    modes = data['mode'].unique()

    for mode in modes:
        mode_data = label_data[label_data['mode'] == mode]
        color = 'steelblue' if mode == 1 else 'red' if mode == 0 else 'orange'

        if label in bad:
            
            ax[0].scatter(mode_data['wx'], mode_data['z_mid'], label=f'Mode {mode}', color="black")
            ax[1].scatter(mode_data['x_mid'], mode_data['z_mid'], label=f'Mode {mode}', color="black")
            ax[2].scatter(mode_data['y_mid'], mode_data['z_mid'], label=f'Mode {mode}', color="black")

        else:
            ax[0].scatter(mode_data['wx'], mode_data['z_mid'], label=f'Mode {mode}', color=color)
            ax[1].scatter(mode_data['x_mid'], mode_data['z_mid'], label=f'Mode {mode}', color=color)
            ax[2].scatter(mode_data['y_mid'], mode_data['z_mid'], label=f'Mode {mode}', color=color)

        
    # Add labels and legend for the current label
    ax[0].set_xlabel('z velocity (cm/s)')
    ax[0].set_ylabel('z (cm)')
    ax[0].set_ylim(30,0)      
    ax[0].set_xlim(-5,45)
    ax[0].grid(True)    
    
    ax[1].set_xlabel('x (cm)')
    ax[1].set_ylabel('z (cm)')
    ax[1].set_ylim(30,0)
    ax[1].set_xlim(-10,10)
    ax[1].grid(True)
    
    ax[2].set_xlabel('y (cm)')
    ax[2].set_ylabel('z (cm)')
    ax[2].set_ylim(30,0)
    ax[2].set_xlim(-10,10) 
    ax[2].grid(True)
    
    
    plt.title(f'Scatter Plot for Label {label}')
    plt.legend()

    # Show the plot for the current label
    plt.show()


# %%

# for i in data['label_mid'].unique():
#     plt.figure(figsize=(5, 5)) 
#     plt.scatter(data["wz"][data["label_mid"]==i][data["mode"]==2], 
#                 data['z_mid'][data["label_mid"]==i][data["mode"]==2], alpha=0.7, c = "orange", label = "Mode 2")
#     plt.scatter(data["wz"][data["label_mid"]==i][data["mode"]==1], 
#                 data['z_mid'][data["label_mid"]==i][data["mode"]==1], alpha=0.7, c = "steelblue", label = "Mode 1")
#     plt.ylim(30, 0)
#     plt.xlim(-10, 40)
#     plt.xlabel("z velocity (cm/s)")
#     plt.title(f'Scatter Plot for Label {i}')
#     plt.ylabel("Depth (cm)")
#     plt.legend()

#%%

data = data[~data['label_mid'].isin(bad)]

#%%%

plt.figure(figsize=(5, 5)) 

for i in data['label_mid'].unique():
    plt.scatter(data["wx"][data["label_mid"]==i][data["mode"]==2], 
                data['z_mid'][data["label_mid"]==i][data["mode"]==2], alpha=0.7, c = "orange", label = "Mode 2")
    plt.scatter(data["wx"][data["label_mid"]==i][data["mode"]==1], 
                data['z_mid'][data["label_mid"]==i][data["mode"]==1], alpha=0.7, c = "steelblue", label = "Mode 1")
    # plt.scatter(data["wz"][data["label_mid"]==i][data["mode"]==0], 
    #             data['z_mid'][data["label_mid"]==i][data["mode"]==0], alpha=0.7, c = "red", label = "Mode 1")
    plt.ylim(30, 0)
    plt.xlim(-10, 60)
    plt.xlabel("z velocity (cm/s)")
    plt.ylabel("Depth (cm)")
    
# plt.figure(figsize=(5, 5)) 
# for i in data['label_mid'].unique():
#     plt.plot(data['wx'], data['y_mid'])
#     plt.plot(data['wy'], data['y_mid'])

#%%
plt.figure()
plt.hist(data["wx"][data["mode"]==1])
plt.hist(data["wx"][data["mode"]==2])

print(np.mean(data["wx"]))

#%%
data.to_excel("5 - modes/" + "mode" + file1[25:])
print("done")

#%%





#%%

# df = pd.DataFrame(label_data_all)


# # Get unique labels in the data
# unique_labels = df['label'].unique()

# # Plotting for each label
# for label in unique_labels:
#     label_data = df[df['label'] == label]
#     fig, ax = plt.subplots(1, 3, figsize=(7, 6)) 

#     # Create a list of unique modes for coloring within this label
#     modes = label_data['mode'].unique()

#     for mode in modes:
#         mode_data = label_data[label_data['mode'] == mode]
#         # plt.scatter(mode_data['xp_list_mid'], mode_data['yp_list_mid'], label=f'Mode {mode}')
#         ax[0].scatter(mode_data['wy'], mode_data['y_mid'], label=f'Mode {mode}')
#         ax[1].scatter(mode_data['xp_list_mid1'], mode_data['y_mid'], label=f'Mode {mode}')
#         ax[2].scatter(mode_data['xp_list_mid2'], mode_data['yp_list_mid2'], label=f'Mode {mode}')

        
#     # Add labels and legend for the current label
#     ax[0].set_xlabel('yv velocity (cm/s)')
#     ax[0].set_ylabel('yp_list_mid')
#     ax[0].set_ylim(30,0)      
#     ax[0].set_xlim(-5,50)      
    
#     ax[1].set_xlabel('xp_list_mid')
#     ax[1].set_ylabel('yp_list_mid')
#     ax[1].set_ylim(30,0)
#     ax[1].set_xlim(-10,10) 
    
#     ax[2].set_xlabel('xp_list_mid')
#     ax[2].set_ylabel('yp_list_mid')
#     ax[2].set_ylim(30,0)
#     ax[2].set_xlim(-10,10) 
    
    
#     plt.title(f'Scatter Plot for Label {label}')
#     plt.legend()

#     # Show the plot for the current label
#     plt.show()
    
#%%

# for i in data['label'].unique():
#     plt.figure()
#     plt.plot(label_data_d1["wy"], 2.5)
#     plt.ylim(30, 0)

#%%
# min_value = np.min(label_data_d1["wy"])
# max_value = np.max(label_data_d1["wy"])
# bin_width = 1
# num_bins = int((max_value - min_value) / bin_width)
# bin_edges = np.linspace(min_value, max_value, num_bins + 1)

# fig, axs = plt.subplots(5, 1, figsize=(4, 8))
# axs[0].hist(label_data_d1["wy"][label_data_d1["mode"]==1])
# axs[0].hist(label_data_d1["wy"][label_data_d1["mode"]==2])

# axs[1].hist(label_data_d2["wy"][label_data_d2["mode"]==1])
# axs[1].hist(label_data_d2["wy"][label_data_d2["mode"]==2])

# axs[2].hist(label_data_d3["wy"][label_data_d3["mode"]==1])
# axs[2].hist(label_data_d3["wy"][label_data_d3["mode"]==2])

# axs[3].hist(label_data_d4["wy"][label_data_d4["mode"]==1])
# axs[3].hist(label_data_d4["wy"][label_data_d4["mode"]==2])

# axs[4].hist(label_data_d5["wy"][label_data_d5["mode"]==1])
# axs[4].hist(label_data_d5["wy"][label_data_d5["mode"]==2])


#%%






# fig, axes = plt.subplots(5, 1, figsize=(5, 9))

# # Define your data and labels
# depth_ranges = ["0-5 depth", "5-10 depth", "10-15 depth", "15-20 depth", "20-25 depth"]

# data = [label_data_d1, label_data_d2, label_data_d3, label_data_d4, label_data_d5]
# colors = ['b', 'g']
# modes = [1, 2]




# plt.figure
# plt.hist(label_data_d3["wy"][label_data_d3["mode"]==2])

# for i in range(5):
#     ax = axes[i]
    
#     for mode, color in zip(modes, colors):
#         # sns.kdeplot(data[i]["wy"][data[i]["mode"] == mode], c=color, ax=ax)
#         ax.hist(data[i]["wy"][data[i]["mode"] == mode], alpha = 0.5, bins = bin_edges, color=color)
    
#         mode_data = data[i]["wy"][data[i]["mode"] == mode]
#         average = np.mean(mode_data)

#         ax.axvline(average, color=color, linestyle='--', label=f'Avg Mode {mode}: {average:.2f}')
    
#     ax.set_xlabel('y velocity (cm/s)')
#     ax.set_ylabel('Frequncy')
#     ax.set_xlim(-7.5, 40)
#     ax.set_title(depth_ranges[i])
#     sns.despine(top=True, right=True, left=False, bottom=False)

#     if i == 1:
#             ax.legend(["Mode 1", "Mode 2"])

# # Adjust spacing between subplots
# plt.tight_layout()

# # Show the plot
# plt.show()






