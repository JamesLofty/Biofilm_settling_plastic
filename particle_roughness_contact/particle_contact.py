
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


contact = pd.read_excel("particle_contact_py.xlsx")

melted_contact = pd.melt(contact)

def new_column(value):
    if 'b' in value:
        return 'b'
    else:
        return 'p'

# Apply the function to create a new column 'new_column'
melted_contact['plastic'] = melted_contact['variable'].apply(new_column)
melted_contact['variable'] = melted_contact['variable'].str.replace('_b', '')
melted_contact_non = melted_contact[~melted_contact['variable'].str.contains('50x50')]

#%%
roughness = pd.read_excel("particle_roughness_py.xlsx")

melted_roughness = pd.melt(roughness)

def new_column(value):
    if 'b' in value:
        return 'b'
    else:
        return 'p'

# Apply the function to create a new column 'new_column'
melted_roughness['plastic'] = melted_roughness['variable'].apply(new_column)
melted_roughness['variable'] = melted_roughness['variable'].str.replace('_b', '')
melted_roughness_non = melted_roughness[~melted_roughness['variable'].str.contains('50x50')]

custom_palette = {'p': 'dimgrey', 'b': 'yellowgreen'}

# print(np.mean(contact ["PTFE 2x1"]))

#%%
fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(7)


spec = gridspec.GridSpec(ncols=3, nrows=2, wspace=0.6, hspace=0.6)

ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])

ax2 = fig.add_subplot(spec[3])
ax3 = fig.add_subplot(spec[4])
ax4 = fig.add_subplot(spec[5])


filtered_data_1 = melted_contact_non[melted_contact_non['variable'].str.contains('2x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax0)
ax0.legend_.remove()  # Hide legend
ax0.set_ylim(0, 100)



filtered_data_1 = melted_contact[melted_contact['variable'].str.contains('1x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax1)
ax1.legend_.remove()  # Hide legend
ax1.set_ylim(0, 100)


filtered_data_1 = melted_roughness_non[melted_roughness_non['variable'].str.contains('2x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax2)
ax2.legend_.remove()  # Hide legend
ax2.set_ylim(0, 200)



filtered_data_1 = melted_roughness[melted_roughness['variable'].str.contains('1x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax3)
ax3.legend_.remove()  # Hide legend
ax3.set_ylim(0, 200)


filtered_data_1 = melted_roughness[melted_roughness['variable'].str.contains('50x50')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax4)
ax4.legend_.remove()  # Hide legend
ax4.set_ylim(0, 200)


sns.despine(top=True, right=True, left=False, bottom=False)

for ax in [ax0, ax1, ax2, ax3, ax4]:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

ax0.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax1.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax2.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax3.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax4.set_xticklabels(labels=["PTFE", "POM", "PS"])

ax0.set_ylabel("Contact angle (°)")
ax2.set_ylabel("Areal average roughness $Sa$ (μm)")

ax0.set_title("Rectangular plastic")
ax1.set_title("Square plastic")

ax2.set_title("Rectangular plastic")
ax3.set_title("Square plastic")
ax4.set_title("Spherical plastic")

# row_titles = ['Rectangular plastic', 'Square plastic']
# for i, title in enumerate(row_titles):
#     fig.text(0.5, 0.93 - i * 0.48, title, ha='center', va='center', fontsize=12)

plt.savefig("contact.svg", format="svg")



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

# def two_sample_t_test(data1, data2, alpha=0.05):
#     # Perform a two-sample t-test
#     t_statistic, p_value = scipy.stats.ttest_ind(data1, data2)
    
#     # Print the results
#     print(f'T-statistic: {t_statistic}')
#     print(f'P-value: {p_value}')

#     # Compare p-value with alpha to make a decision
#     if p_value < alpha:
#         print("Reject the null hypothesis. There is a significant difference.")
#     else:
#         print("Fail to reject the null hypothesis. There is no significant difference.")


# test1 = density["PS 50x50"]
# test2 = density["PS 50x50_b"]

# two_sample_t_test(test1, test2)

# plt.figure()
# plt.hist(test1)
# plt.hist(test2)

# plt.figure()
# plt.boxplot(test1, positions=[1])
# plt.boxplot(test2, positions=[2])


