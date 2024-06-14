#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:20:21 2024

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.ticker import ScalarFormatter
from collections import Counter
import scipy.stats
from scipy.stats import mannwhitneyu


plastic = "PS"
size = "_05x05_"
x_lim = (0,30)
binsize = 2.1

data1_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_p.xlsx")

data_all_p = data1_p
data_all_p["wxwy"] = ((data_all_p["wx"] ** 2) + (data_all_p["wy"] ** 2)) ** 0.5

data_m1_p = data_all_p[data_all_p["mode"] == 1]#!!!!!!!!


m1_wx_p = data_m1_p["wxwy"]

m1_z_p= data_m1_p["z_mid"]


#%%
data1_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_b.xlsx")

data_all_b = data1_b
data_all_b["wxwy"] = ((data_all_b["wx"] ** 2) + (data_all_b["wy"] ** 2)) ** 0.5

data_m1_b = data_all_b[data_all_b["mode"] == 1]#!!!!!!!!


m1_wx_b = data_m1_b["wxwy"]

m1_z_b= data_m1_b["z_mid"]



depth1 = 5
depth2 = 20

#%%

###############FILTERINF DATA BETWEEN 10 and 20 TO GET A mean THAT IS INDEPENDENT OF DROP RELEASE‹‹‹‹‹‹

data_all_b_filtered = data_all_b[(data_all_b['z_mid'] >= depth1) & (data_all_b['z_mid'] <= depth2)]
data_all_p_filtered = data_all_p[(data_all_p['z_mid'] >= depth1) & (data_all_p['z_mid'] <= depth2)]

#%%%
##### find the mean for each label for each mode

means_mode_b = data_all_b_filtered.groupby(['label_mid', 'mode'])["wxwy"].mean().reset_index()
wx_b_totalmean = means_mode_b.groupby('label_mid')['wxwy'].mean()
m1_wx_b_ = means_mode_b[means_mode_b["mode"] == 1]
m2_wx_b_ = means_mode_b[means_mode_b["mode"] == 2]
m3_wx_b_ = means_mode_b[means_mode_b["mode"] == 3]

means_mode_p = data_all_p_filtered.groupby(['label_mid', 'mode'])["wxwy"].mean().reset_index()
wx_p_totalmean = means_mode_p.groupby('label_mid')['wxwy'].mean()
m1_wx_p_ = means_mode_p[means_mode_p["mode"] == 1]
m2_wx_p_ = means_mode_p[means_mode_p["mode"] == 2]
m3_wx_p_ = means_mode_p[means_mode_p["mode"] == 3]

#%%
##### find the mid coordinate for each label for each mode and velocity for hariy plot

def calculate_average(group):
    bin_width = 1 # Adjust the bin width as needed
    bins = pd.cut(group["z_mid"], bins=range(int(group["z_mid"].min()), int(group["z_mid"].max()) + bin_width, bin_width))
    return group.groupby(bins)["wxwy"].mean()

result_p = data_all_p.groupby("label_mid").apply(calculate_average).reset_index()
result_p = result_p[(result_p['wxwy'] >= 0) & (result_p['wxwy'] <= 30)]
mid_p = result_p["z_mid"].apply(lambda x: x.mid) - 1
result_p["mid"] = mid_p

result_b = data_all_b.groupby("label_mid").apply(calculate_average).reset_index()
result_b = result_b[(result_b['wxwy'] >= 0) & (result_b['wxwy'] <= 30)]
mid_b = result_b["z_mid"].apply(lambda x: x.mid) - 1
result_b["mid"] = mid_b
#%%

######################hariplot

plt.figure(figsize=(2.5,2.5))
for i in result_p['label_mid'].unique():
    plt.plot(result_p["wxwy"][result_p["label_mid"] == i], mid_p[result_p["label_mid"] == i], c = "grey", alpha = 0.4)
    
for i in result_b['label_mid'].unique():
    plt.plot(result_b["wxwy"][result_b["label_mid"] == i], mid_b[result_b["label_mid"] == i], c = "yellowgreen", alpha = 0.4)
    plt.ylim(20, 0)
    plt.xlim(x_lim)
    
plt.axhline(5, 0, 40, color = "red", linestyle = ":", alpha = 0.5, zorder = 10)
plt.axhline(19.7, 0, 40, color = "red", linestyle = ":", alpha = 0.5, zorder = 10)


plt.axvline(x=np.mean(m1_wx_p_["wxwy"]), color='grey', linestyle=':')
plt.axvline(x=np.mean(m1_wx_b_["wxwy"]), color='darkgreen', linestyle=':')

# plt.axvline(x=np.mean(m2_wx_p_["wxwy"]), color='grey', linestyle='--')
# plt.axvline(x=np.mean(m2_wx_b_["wxwy"]), color='darkgreen', linestyle='--')

# plt.axvline(x=np.mean(m3_wx_p_["wxwy"]), color='grey', linestyle=('-.'))
# plt.axvline(x=np.mean(m3_wx_b_["wxwy"]), color='darkgreen', linestyle=('-.'))

# plt.axvline(x=np.mean(wx_p_totalmean), color='grey', linestyle='-')
# plt.axvline(x=np.mean(wx_b_totalmean), color='darkgreen', linestyle='-')

# print(np.mean(m1_wx_p_["wxwy"]))
# print(np.mean(m1_wx_b_["wxwy"]))


plt.xlabel('$w_h$ (cm/s)')
plt.ylabel('Depth (cm)')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig("figures/hairplot_h.svg", format="svg")

#%%

#for each label, this calulates the average velocity of when a particle is in mode 1 or 2 or 3. m
#multiple labels will have multiple modes

data_m1_p_filtered = data_m1_p[(data_m1_p['z_mid'] >= depth1) & (data_m1_p['z_mid'] <= depth2)]


data_m1_b_filtered = data_m1_b[(data_m1_b['z_mid'] >= depth1) & (data_m1_b['z_mid'] <= depth2)]

#%%

hist_mode_m1_p = data_m1_p_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()


hist_mode_m1_b = data_m1_b_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()


min_value = np.min(data1_b["wx"])
max_value = np.max(data1_b["wx"])
bin_width = binsize
num_bins = int((max_value - min_value) / bin_width)
bin_edges = np.linspace(min_value, max_value, num_bins + 1)

#%%
plt.figure(figsize=(2.65, 2))

plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.hist(hist_mode_m1_p["wxwy"], histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='black', fc='none')

plt.xlim(x_lim)
plt.yscale('log')
plt.ylim(0.7, 100)
plt.yticks([1, 100])  # Set y-axis ticks to 0.1 and 100
plt.gca().axes.xaxis.set_ticklabels([])
sns.despine(top=True, right=True, left=False, bottom=False)

# Plotting the second histogram
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.hist(hist_mode_m1_b["wxwy"], histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='darkgreen', fc='none')
plt.yscale('log')
plt.xlim(x_lim)
plt.ylim(0.7, 100)
plt.yticks([1, 100])  # Set y-axis ticks to 0.1 and 100
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xlabel("$w_h$ (cm/s)")

# plt.subplots_adjust(vspace=10)  
plt.tight_layout(h_pad=3.0)  # Increase the vertical space between subplots
plt.savefig("figures/histo_combined_h.svg", format="svg")
plt.show()
#%%

plt.figure(figsize=(2, 5))

plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.boxplot([wx_b_totalmean.dropna(),wx_p_totalmean.dropna()], 
            widths=0.5, patch_artist=True,
            boxprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5),
            meanprops=dict(color='red', linewidth=1.5))

# Optionally, you can set labels for each boxplot
plt.xticks([1, 2], ['biofouled', 'pristine'])

plt.show()



#%%

print("mode1 prestine",round(np.mean(m1_wx_p_["wxwy"]), 2), round(np.std(m1_wx_p_["wxwy"]), 2))
print("mode1 biofouled", round(np.mean(m1_wx_b_["wxwy"]), 2), round(np.std(m1_wx_b_["wxwy"]), 2))

print("mode2 pristine", round(np.mean(m2_wx_p_["wxwy"]), 2), round(np.std(m2_wx_p_["wxwy"]), 2))
print("mode2 biofouled", round(np.mean(m2_wx_b_["wxwy"]), 2), round(np.std(m2_wx_b_["wxwy"]), 2))

# print("mode3 pristine", round(np.mean(m2_wx_p), 2), round(np.std(m2_wx_p), 2))
print("mode3 biofouled", round(np.mean(m3_wx_b_["wxwy"]), 2), round(np.std(m3_wx_b_["wxwy"]), 2))
print("mode3 pristine", round(np.mean(m3_wx_p_["wxwy"]), 2), round(np.std(m3_wx_p_["wxwy"]), 2))


print("total mean pristine", round(np.mean(wx_p_totalmean), 2), round(np.std(wx_p_totalmean), 2))
print("total mean biofouled", round(np.mean(wx_b_totalmean), 2), round(np.std(wx_b_totalmean), 2))


#%%

# %%%% of particles in each mode at 10 cm depth

def calculate_mode_percentages(data):
    data = data[data['mode'] != 0]
    filtered_df = data[(data['z_mid'] >= 4) & (data['z_mid'] <= 6)]
    
    percent_mode = []
    
    for i in filtered_df["label_mid"].unique(): 
        mode = np.average(filtered_df["mode"][filtered_df["label_mid"] == i])
        rounded_mode = round(mode)
        percent_mode.append(rounded_mode)
        # print(i, rounded_mode)
    
    mode_counts = Counter(percent_mode)
    total_particles = len(percent_mode)
    for mode, count in mode_counts.items():
        percentage = (count / total_particles) * 100
        print(f'Mode {mode}: {percentage:.0f}%')

# Example usage with two sets of data

print("biofouled:",)
calculate_mode_percentages(data_all_b)
print("\n")

print("pristine:",)
calculate_mode_percentages(data_all_p)
print("\n")



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

mann_whitney_u_test(wx_p_totalmean.dropna(), wx_b_totalmean.dropna())


plt.figure()
plt.hist(wx_p_totalmean.dropna(), bins = 10,alpha = 0.2, color = "b")
plt.hist(wx_b_totalmean.dropna(), bins = 10,alpha = 0.2, color = "g")
plt.axvline(x=np.mean(np.mean(wx_p_totalmean)), color='b', linestyle='-')
plt.axvline(x=np.mean(np.mean(wx_b_totalmean)), color='g', linestyle='-')

# wx_p_totalmean.to_excel("essemabled_average_settling/" + "w_h_p_" + plastic + size + ".xlsx")
# wx_b_totalmean.to_excel("essemabled_average_settling/" + "w_h_b_" + plastic + size + ".xlsx")

# hist_mode_p = pd.concat([hist_mode_m1_p, hist_mode_m2_p, hist_mode_m3_p])
# hist_mode_b = pd.concat([hist_mode_m1_b, hist_mode_m2_b, hist_mode_m3_b])

# hist_mode_p.to_excel("essemabled_average_settling/" + "w_h_p_mode" + plastic + size + ".xlsx")
# hist_mode_b.to_excel("essemabled_average_settling/" + "w_h_b_mode" + plastic + size + ".xlsx")