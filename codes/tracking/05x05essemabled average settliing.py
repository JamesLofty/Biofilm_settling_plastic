#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:38:02 2024

@author: jameslofty
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:01:08 2024

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
x_lim = (0,40)
binsize = 2.1


# data1_p = pd.read_excel("5 - modes/modePS_1x1_pos1_p.xlsx")
# data2_p = pd.read_excel("5 - modes/modePS_1x1_pos2_p.xlsx")
# data3_p = pd.read_excel("5 - modes/modePS_1x1_pos3_p.xlsx")

data1_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_p.xlsx")


data_all_p = data1_p[data1_p['mode'] != 0]


data_m1_p = data_all_p[data_all_p["mode"] == 1]#!!!!!!!!

m1_wz_p = data_m1_p["wz"]

m1_z_p= data_m1_p["z_mid"]

#%%
data1_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_b.xlsx")

data_all_b = data1_b[data1_b['mode'] != 0]


data_m1_b = data_all_b[data_all_b["mode"] == 1]#!!!!!!!!


m1_wz_b = data_m1_b["wz"]


m1_z_b= data_m1_b["z_mid"]

depth1 = 5
depth2 = 20

#%%

###############FILTERINF DATA BETWEEN 10 and 20 TO GET A median THAT IS INDEPENDENT OF DROP RELEASE‹‹‹‹‹‹

data_all_b_filtered = data_all_b[(data_all_b['z_mid'] >= depth1) & (data_all_b['z_mid'] <= depth2)]
data_all_p_filtered = data_all_p[(data_all_p['z_mid'] >= depth1) & (data_all_p['z_mid'] <= depth2)]

#%%%
##### find the median for each label for each mode

medians_mode_b = data_all_b_filtered.groupby(['label_mid', 'mode'])['wz'].median().reset_index()
wz_b_totalmedian = medians_mode_b.groupby('label_mid')['wz'].median()
m1_wz_b_ = medians_mode_b[medians_mode_b["mode"] == 1]
m2_wz_b_ = medians_mode_b[medians_mode_b["mode"] == 2]
m3_wz_b_ = medians_mode_b[medians_mode_b["mode"] == 3]

medians_mode_p = data_all_p_filtered.groupby(['label_mid', 'mode'])['wz'].median().reset_index()
wz_p_totalmedian = medians_mode_p.groupby('label_mid')['wz'].median()
m1_wz_p_ = medians_mode_p[medians_mode_p["mode"] == 1]
m2_wz_p_ = medians_mode_p[medians_mode_p["mode"] == 2]


#%%
##### find the mid coordinate for each label for each mode and velocity for hari plot

def calculate_average(group):
    bin_width = 1 # Adjust the bin width as needed
    bins = pd.cut(group["z_mid"], bins=range(int(group["z_mid"].min()), int(group["z_mid"].max()) + bin_width, bin_width))
    return group.groupby(bins)["wz"].median()

result_p = data_all_p.groupby("label_mid").apply(calculate_average).reset_index()
mid_p = result_p["z_mid"].apply(lambda x: x.mid)
mid_p = result_p["z_mid"].apply(lambda x: x.mid) - 1.5
result_p["mid"] = mid_p

result_b = data_all_b.groupby("label_mid").apply(calculate_average).reset_index()
mid_b = result_b["z_mid"].apply(lambda x: x.mid)
mid_b = result_b["z_mid"].apply(lambda x: x.mid) - 1.5
result_b["mid"] = mid_b
#%%

######################hariplot

plt.figure(figsize=(2.5,2.5))
for i in result_p['label_mid'].unique():
    plt.plot(result_p["wz"][result_p["label_mid"] == i], mid_p[result_p["label_mid"] == i], c = "grey", alpha = 0.4)
    
for i in result_b['label_mid'].unique():
    plt.plot(result_b["wz"][result_b["label_mid"] == i], mid_b[result_b["label_mid"] == i], c = "yellowgreen", alpha = 0.4)
    plt.ylim(20, 0)
    plt.xlim(x_lim)
    
plt.axhline(5, 0, 40, color = "red", linestyle = ":", alpha = 0.5)
plt.axhline(19.7, 0, 40, color = "red", linestyle = ":", alpha = 0.5)


plt.axvline(x=np.median(m1_wz_p_["wz"]), color='grey', linestyle=':')
plt.axvline(x=np.median(m1_wz_b_["wz"]), color='darkgreen', linestyle=':')

plt.axvline(x=np.median(m2_wz_p_["wz"]), color='grey', linestyle='--')
plt.axvline(x=np.median(m2_wz_b_["wz"]), color='darkgreen', linestyle='--')

plt.axvline(x=np.median(m3_wz_b_["wz"]), color='darkgreen', linestyle=('-.'))

plt.axvline(x=np.median(wz_p_totalmedian), color='grey', linestyle='-')
plt.axvline(x=np.median(wz_b_totalmedian), color='darkgreen', linestyle='-')

plt.xlabel('$w$ (cm/s)')
plt.ylabel('Depth (cm)')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig("figures/hairplot.svg", format="svg")

#%%

#for each label, this calulates the average velocity of when a particle is in mode 1 or 2 or 3. m
#multiple labels will have multiple modes

data_m1_p_filtered = data_m1_p[(data_m1_p['z_mid'] >= depth1) & (data_m1_p['z_mid'] <= depth2)]

data_m1_b_filtered = data_m1_b[(data_m1_b['z_mid'] >= depth1) & (data_m1_b['z_mid'] <= depth2)]


hist_mode_m1_p = data_m1_p_filtered.groupby('label_mid').agg({'wz': 'median', 'mode': 'first'}).reset_index()

hist_mode_m1_b = data_m1_b_filtered.groupby('label_mid').agg({'wz': 'median', 'mode': 'first'}).reset_index()


min_value = np.min(data1_b["wz"])
max_value = np.max(data1_b["wz"])
bin_width = binsize
num_bins = int((max_value - min_value) / bin_width)
bin_edges = np.linspace(min_value, max_value, num_bins + 1)

#%%
plt.figure(figsize=(2.65, 3))

plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.hist(hist_mode_m1_p["wz"], density=(True),bins=bin_edges, lw=1.5, alpha = 0.2, color = 'dimgray')
plt.hist(hist_mode_m1_p["wz"], histtype='step', density=(True),bins=bin_edges, lw=1.5, alpha=1, ec='dimgray', fc='none')

plt.axvline(x=np.mean(m1_wz_p_["wz"]), color='grey', linestyle='-')

plt.xlim(x_lim)
plt.ylim(0, 0.5)
plt.gca().axes.xaxis.set_ticklabels([])
sns.despine(top=True, right=True, left=False, bottom=False)

# Plotting the second histogram
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.hist(hist_mode_m1_b["wz"], density=(True),bins=bin_edges, lw=1.5, alpha = 0.2, color = 'darkgreen')
plt.hist(hist_mode_m1_b["wz"], histtype='step', density=(True),bins=bin_edges, lw=1.5, alpha=1, ec='darkgreen', fc='none')

plt.axvline(x=np.mean(m1_wz_b_["wz"]), color='darkgreen', linestyle='-')

plt.xlim(x_lim)
plt.ylim(0, 0.5)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xlabel("$w$ (cm/s)")

# plt.subplots_adjust(vspace=10)  
plt.tight_layout(h_pad=3.0)  # Increase the vertical space between subplots
plt.savefig("figures/histo_combined.svg", format="svg")
plt.show()

#%%

print("mode1 prestine",round(np.median(m1_wz_p_["wz"]), 2), round(np.std(m1_wz_p_["wz"]), 2))
print("mode1 biofouled", round(np.median(m1_wz_b_["wz"]), 2), round(np.std(m1_wz_b_["wz"]), 2))

print("mode2 pristine", round(np.median(m2_wz_p_["wz"]), 2), round(np.std(m2_wz_p_["wz"]), 2))
print("mode2 biofouled", round(np.median(m2_wz_b_["wz"]), 2), round(np.std(m2_wz_b_["wz"]), 2))

# print("mode3 pristine", round(np.median(m2_wz_p), 2), round(np.std(m2_wz_p), 2))
print("mode3 biofouled", round(np.median(m3_wz_b_["wz"]), 2), round(np.std(m3_wz_b_["wz"]), 2))

print("total median pristine", round(np.median(wz_p_totalmedian), 2), round(np.std(wz_p_totalmedian), 2))
print("total median biofouled", round(np.median(wz_b_totalmedian), 2), round(np.std(wz_b_totalmedian), 2))


#%%
# data = data_all_p
# data = data[data['mode'] != 0]

# filtered_df = data[(data['z_mid'] >= 9) & (data['z_mid'] <= 11)]

# percent_mode = []

# for i in filtered_df["label_mid"].unique(): 
#     mode = np.average(filtered_df["mode"][filtered_df["label_mid"] == i])
#     rounded_mode = round(mode)
#     percent_mode.append(rounded_mode)
#     # print(i, rounded_mode)
    
# mode_counts = Counter(percent_mode)
# total_particles = len(percent_mode)
# for mode, count in mode_counts.items():
#     percentage = (count / total_particles) * 100
#     print(f'Mode {mode}: {percentage:.0f}%')


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

mann_whitney_u_test(wz_p_totalmedian.dropna(), wz_b_totalmedian.dropna())

plt.figure()
plt.hist(wz_p_totalmedian.dropna(), bins = 10,alpha = 0.2, color = "b")
plt.hist(wz_b_totalmedian.dropna(), bins = 10,alpha = 0.2, color = "g")
plt.axvline(x=np.median(np.median(wz_p_totalmedian)), color='b', linestyle='-')
plt.axvline(x=np.median(np.median(wz_b_totalmedian)), color='g', linestyle='-')
