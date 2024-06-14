
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

# print(round(np.mean(contact["PTFE 50x50_b"]), 2), round(np.std(contact["PTFE 50x50   _b"]), 2))




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

print(round(np.mean(roughness["PTFE 2x1"]), 2), round(np.std(roughness["PTFE 2x1"]), 2))

#%%
spat_dev = pd.read_excel("/Users/jameslofty/Library/CloudStorage/OneDrive-CardiffUniversity/PhD/Biofilm+settling/biofilms/particle_biofilm_deviation.xlsx")

melted_spat_dev = pd.melt(spat_dev)

def new_column(value):
    if 'b' in value:
        return 'b'
    else:
        return 'p'

# Apply the function to create a new column 'new_column'
melted_spat_dev['plastic'] = melted_spat_dev['variable'].apply(new_column)
melted_spat_dev['variable'] = melted_spat_dev['variable'].str.replace('_b', '')
melted_spat_dev_non = melted_spat_dev[~melted_spat_dev['variable'].str.contains('50x50')]


#%%
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(6)


spec = gridspec.GridSpec(ncols=3, nrows=3, wspace=0.4, hspace=0.4)

ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])
ax2 = fig.add_subplot(spec[2])

ax3 = fig.add_subplot(spec[3])
ax4 = fig.add_subplot(spec[4])

ax6 = fig.add_subplot(spec[6])
ax7 = fig.add_subplot(spec[7])
ax8 = fig.add_subplot(spec[8])


# ax9 = fig.add_subplot(spec[9])
# ax10 = fig.add_subplot(spec[10])

# ax1 = fig.add_subplot(spec[1])
# ax2 = fig.add_subplot(spec[2])

filtered_data_1 = melted_roughness_non[melted_roughness_non['variable'].str.contains('2x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax0)
ax0.legend_.remove()  # Hide legend
ax0.set_ylim(0, 200)



filtered_data_1 = melted_roughness[melted_roughness['variable'].str.contains('1x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax1)
ax1.legend_.remove()  # Hide legend
ax1.set_ylim(0, 200)


filtered_data_1 = melted_roughness[melted_roughness['variable'].str.contains('50x50')]
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


filtered_data_1 = melted_contact_non[melted_contact_non['variable'].str.contains('2x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax3)
ax3.legend_.remove()  # Hide legend
ax3.set_ylim(20, 100)



filtered_data_1 = melted_contact[melted_contact['variable'].str.contains('1x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax4)
ax4.legend_.remove()  # Hide legend
ax4.set_ylim(20, 100)


filtered_data_1 = melted_spat_dev[melted_spat_dev['variable'].str.contains('2x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax6)
ax6.legend_.remove()  # Hide legend
ax6.set_ylim(0, 0.25)

filtered_data_1 = melted_spat_dev[melted_spat_dev['variable'].str.contains('1x1')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax7)
ax7.legend_.remove()  # Hide legend
ax7.set_ylim(0, 0.25)

filtered_data_1 = melted_spat_dev[melted_spat_dev['variable'].str.contains('50x50')]
sns.boxplot(x=filtered_data_1['variable'], 
            y=filtered_data_1['value'], 
            hue=filtered_data_1['plastic'],
            showfliers=False,
            palette=custom_palette,
            width=.8,
            linewidth=1,
            ax=ax8)
ax8.legend_.remove()  # Hide legend
ax8.set_ylim(0, 0.25)


# sns.boxplot(data=ALL[ALL["size"] == "2x1"], 
#             x='plastic', 
#             y='biofilm_percentages', 
#             ax=ax6, 
#             color='yellowgreen', 
#             showfliers=False)

# sns.boxplot(data=ALL[ALL["size"] == "1x1"], 
#             x='plastic', 
#             y='biofilm_percentages', 
#             ax=ax7, 
#             color='yellowgreen', 
#             showfliers=False)

# sns.boxplot(data=ALL[ALL["size"] == "2x1"], 
#             x='plastic', 
#             y='std_intensity_means', 
#             ax=ax9, 
#             color='yellowgreen', 
#             showfliers=False)

# sns.boxplot(data=ALL[ALL["size"] == "1x1"], 
#             x='plastic', 
#             y='std_intensity_means', 
#             ax=ax10, 
#             color='yellowgreen', 
#             showfliers=False)




sns.despine(top=True, right=True, left=False, bottom=False)

for ax in [ax0, ax1, ax2, ax3, ax4, ax6, ax7, ax8]:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

ax0.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax1.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax2.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax3.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax4.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax6.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax7.set_xticklabels(labels=["PTFE", "POM", "PS"])
ax8.set_xticklabels(labels=["PTFE", "POM", "PS"])



ax0.set_ylabel("Areal average roughness\n $Sa$ (μm)")
ax3.set_ylabel("Contact angle (°)")
ax6.set_ylabel("Biofilm spatial deviation \n(normalised pixel value)")


ax0.set_title("Rectangular plastic")
ax1.set_title("Square plastic")
ax2.set_title("Spherical plastic")

ax3.set_title("Rectangular plastic")
ax4.set_title("Square plastic")

ax6.set_title("Rectangular plastic")
ax7.set_title("Square plastic")

# row_titles = ['Rectangular plastic', 'Square plastic']
# for i, title in enumerate(row_titles):
#     fig.text(0.5, 0.93 - i * 0.48, title, ha='center', va='center', fontsize=12)

plt.savefig("contact+roughnes.svg", format="svg")

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

mann_whitney_u_test

test1 = contact["PTFE 1x1"].dropna()
test2 = contact["PTFE 1x1_b"].dropna()

mann_whitney_u_test(test1, test2)

plt.figure()
plt.hist(test1)
plt.hist(test2)

plt.figure()
plt.boxplot(test1, positions=[1])
plt.boxplot(test2, positions=[2])


