#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:57:24 2023

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft
from numpy.fft import fft, ifft
import seaborn as sns
from matplotlib import gridspec
import scipy.stats
from scipy.stats import mannwhitneyu

def calculate_percentage_change(old_value, new_value):
    percentage_change = ((new_value - old_value) / abs(old_value)) * 100
    return percentage_change

#%%

density = pd.read_excel("py_particle_density.xlsx")
weight = pd.read_excel("py_particle_weight.xlsx")

melted_density = pd.melt(density)
melted_weight = pd.melt(weight)


def new_column(value):
    if 'b' in value:
        return 'b'
    else:
        return 'p'

# Apply the function to create a new column 'new_column'
melted_density['plastic'] = melted_density['variable'].apply(new_column)
melted_density['variable'] = melted_density['variable'].str.replace('_b', '')
melted_density_non = melted_density[~melted_density['variable'].str.contains('50x50')]


melted_weight['plastic'] = melted_weight['variable'].apply(new_column)
melted_weight['variable'] = melted_weight['variable'].str.replace('_b', '')
melted_weight_non = melted_weight[~melted_weight['variable'].str.contains('50x50')]


custom_palette = {'p': 'dimgrey', 'b': 'yellowgreen'}



print(round(np.mean(density["PTFE 50x50_b"]), 2), round(np.std(density["PTFE 50x50_b"]), 2))

diff_2x1 = calculate_percentage_change(np.mean(density["PTFE 2x1"]), np.mean(density["PTFE 2x1_b"]))
diff_1x1 = calculate_percentage_change(np.mean(density["PTFE 1x1"]), np.mean(density["PTFE 1x1_b"]))
diff_50x50 = calculate_percentage_change(np.mean(density["PTFE 50x50"]), np.mean(density["PTFE 50x50_b"]))

# print(np.mean([diff_2x1, diff_1x1, diff_50x50 ]), np.std([diff_2x1, diff_1x1, diff_50x50 ]))




#%%
fig = plt.figure()
fig.set_figheight(7)
fig.set_figwidth(4)

spec = gridspec.GridSpec(ncols=2, nrows=3,
                         width_ratios=[2, 1], wspace=0.6,
                         hspace=0.8)


ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])
ax2 = fig.add_subplot(spec[2])
ax3 = fig.add_subplot(spec[3])
ax4 = fig.add_subplot(spec[4])
ax5 = fig.add_subplot(spec[5])

filtered_data_1 = melted_density_non[melted_density_non['variable'].str.contains('PTFE')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.6,
            linewidth=1,
            ax=ax0)
ax0.set_ylim(1950, 2050)

filtered_data_1 = melted_density[melted_density['variable'].str.contains('PTFE 50x50')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.6,
            linewidth=1,
            ax=ax1)
ax1.set_ylim(2100, 2200)


filtered_data_1 = melted_density_non[melted_density_non['variable'].str.contains('POM')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.6,
            linewidth=1,
            ax=ax2)
ax2.set_ylim(1380, 1480)


filtered_data_1 = melted_density[melted_density['variable'].str.contains('POM 50x50')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.6,
            linewidth=1,
            ax=ax3)
ax3.set_ylim(1300, 1400)


filtered_data_1 = melted_density_non[melted_density_non['variable'].str.contains('PS')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.6,
            linewidth=1,
            ax=ax4)
ax4.set_ylim(1050, 1200)


filtered_data_1 = melted_density[melted_density['variable'].str.contains('PS 50x50')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.6,
            linewidth=1,
            ax=ax5)
ax5.set_ylim(1000, 1100)


sns.despine(top=True, right=True, left=False, bottom=False)

for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

ax0.set_xticks(ticks=[0, 1], labels=["Rectangular", "Square"])
ax1.set_xticks(ticks=[0], labels=["Spherical"])

ax2.set_xticks(ticks=[0, 1], labels=["Rectangular", "Square"])
ax3.set_xticks(ticks=[0], labels=["Spherical"])

ax4.set_xticks(ticks=[0, 1], labels=["Rectangular", "Square"])
ax5.set_xticks(ticks=[0], labels=["Spherical"])

    
ax2.set_ylabel("Density (kg/m$^3$)")

row_titles = ['PTFE', 'POM', "PS"]
for i, title in enumerate(row_titles):
    fig.text(0.5, 0.93 - i * 0.30, title, ha='center', va='center', fontsize=12)

plt.savefig("density.svg", format="svg")



#%%


# fig = plt.figure()
# fig.set_figheight(5)
# fig.set_figwidth(3.5)

# spec = gridspec.GridSpec(ncols=2, nrows=3,
#                          width_ratios=[2, 1], wspace=0.6,
#                          hspace=0.8)


# ax0 = fig.add_subplot(spec[0])
# ax1 = fig.add_subplot(spec[1])
# ax2 = fig.add_subplot(spec[2])
# ax3 = fig.add_subplot(spec[3])
# ax4 = fig.add_subplot(spec[4])
# ax5 = fig.add_subplot(spec[5])

# filtered_data_1 = melted_weight_non[melted_weight_non['variable'].str.contains('PTFE')]
# sns.boxplot(x=filtered_data_1['variable'], 
#             y=filtered_data_1['value'], 
#             hue=filtered_data_1['plastic'],
#             showfliers=False,
#             palette=custom_palette,
#             width=0.5,
#             ax=ax0)
# ax0.set_ylim(0.13, 0.6)

# filtered_data_1 = melted_weight[melted_weight['variable'].str.contains('PTFE 50x50')]
# sns.boxplot(x=filtered_data_1['variable'], 
#             y=filtered_data_1['value'], 
#             hue=filtered_data_1['plastic'],
#             showfliers=False,
#             palette=custom_palette,
#             width=.5,
#             ax=ax1)
# ax1.set_ylim(0.14, 0.144)


# filtered_data_1 = melted_weight_non[melted_weight_non['variable'].str.contains('POM')]
# sns.boxplot(x=filtered_data_1['variable'], 
#             y=filtered_data_1['value'], 
#             hue=filtered_data_1['plastic'],
#             showfliers=False,
#             palette=custom_palette,
#             width=.5,
#             ax=ax2)
# ax2.set_ylim(0.13, 0.4)


# filtered_data_1 = melted_weight[melted_weight['variable'].str.contains('POM 50x50')]
# sns.boxplot(x=filtered_data_1['variable'], 
#             y=filtered_data_1['value'], 
#             hue=filtered_data_1['plastic'],
#             showfliers=False,
#             palette=custom_palette,
#             width=.5,
#             ax=ax3)
# ax3.set_ylim(0.09, 0.1)


# filtered_data_1 = melted_weight_non[melted_weight_non['variable'].str.contains('PS')]
# sns.boxplot(x=filtered_data_1['variable'], 
#             y=filtered_data_1['value'], 
#             hue=filtered_data_1['plastic'],
#             showfliers=False,
#             palette=custom_palette,
#             width=.5,
#             ax=ax4)
# ax4.set_ylim(0.1, 0.3)



# filtered_data_1 = melted_weight[melted_weight['variable'].str.contains('PS 50x50')]
# sns.boxplot(x=filtered_data_1['variable'], 
#             y=filtered_data_1['value'], 
#             hue=filtered_data_1['plastic'],
#             showfliers=False,
#             palette=custom_palette,
#             width=.5,
#             ax=ax5)
# ax5.set_ylim(0.055, 0.066)



# sns.despine(top=True, right=True, left=False, bottom=False)

# for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
#     ax.get_legend().remove()
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.set_xticklabels([])


# ax0.set_xticks(ticks=[0, 1], labels=["Rectangular", "Square"])
# ax1.set_xticks(ticks=[0], labels=["Spherical"])

# ax2.set_xticks(ticks=[0, 1], labels=["Rectangular", "Square"])
# ax3.set_xticks(ticks=[0], labels=["Spherical"])

# ax4.set_xticks(ticks=[0, 1], labels=["Rectangular", "Square"])
# ax5.set_xticks(ticks=[0], labels=["Spherical"])


# ax2.set_ylabel("Weight (g)")

# row_titles = ['PTFE', 'POM', "PS"]
# for i, title in enumerate(row_titles):
#     fig.text(0.5, 0.93 - i * 0.30, title, ha='center', va='center', fontsize=12)



# plt.savefig("weight.svg", format="svg")

# #%%

# Index(['PTFE 2x1', 'PTFE 2x1_b', 'PTFE 1x1', 'PTFE 1x1_b', 'PTFE 50x50',
#        'PTFE 50x50_b', 'PS 2x1', 'PS 2x1_b', 'PS 1x1', 'PS 1x1_b', 'PS 50x50',
#        'PS 50x50_b', 'POM 2x1', 'POM 2x1_b', 'POM 1x1', 'POM 1x1_b',
#        'POM 50x50', 'POM 50x50_b'],
#       dtype='object')

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


test1 = density["PTFE 1x1"]
test2 = density["PTFE 1x1_b"]

mann_whitney_u_test(test1, test2)

plt.figure()
plt.hist(test1)
plt.hist(test2)

plt.figure()
plt.boxplot(test1, positions=[1])
plt.boxplot(test2, positions=[2])


