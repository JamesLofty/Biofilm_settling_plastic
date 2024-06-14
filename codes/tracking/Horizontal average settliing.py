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
size = "_1x1_"
x_lim = (0,30)
binsize = 2.1

# data1_p = pd.read_excel("5 - modes/modePS_1x1_pos1_p.xlsx")
# data2_p = pd.read_excel("5 - modes/modePS_1x1_pos2_p.xlsx")
# data3_p = pd.read_excel("5 - modes/modePS_1x1_pos3_p.xlsx")

data1_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_p.xlsx")
data2_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos2_p.xlsx")
data3_p = pd.read_excel("5 - modes/mode" + plastic + size + "pos3_p.xlsx")

data2_p["label_mid"] = data2_p["label_mid"] + 100
data3_p["label_mid"] = data3_p["label_mid"] + 1000

data_all_p = pd.concat([data1_p, data2_p, data3_p])
data_all_p["wxwy"] = ((data_all_p["wx"] ** 2) + (data_all_p["wy"] ** 2)) ** 0.5
# data_all_p = data_all_p[data_all_p["mode"] == 2]


data_m1_p = data_all_p[data_all_p["mode"] == 1]#!!!!!!!!
data_m2_p = data_all_p[data_all_p["mode"] == 2]
data_m3_p = data_all_p[data_all_p["mode"] == 3]

m1_wx_p = data_m1_p["wxwy"]
m2_wx_p = data_m2_p["wxwy"]
m3_wx_p = data_m3_p["wxwy"]

m1_z_p= data_m1_p["z_mid"]
m2_z_p = data_m2_p["z_mid"]
m3_z_p = data_m3_p["z_mid"]

#%%
data1_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos1_b.xlsx")
data2_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos2_b.xlsx")
data3_b = pd.read_excel("5 - modes/mode" + plastic + size + "pos3_b.xlsx")

data2_b["label_mid"] = data2_b["label_mid"] + 100
data3_b["label_mid"] = data3_b["label_mid"] + 1000

data_all_b = pd.concat([data1_b, data2_b, data3_b])
data_all_b["wxwy"] = ((data_all_b["wx"] ** 2) + (data_all_b["wy"] ** 2)) ** 0.5
# data_all_b = data_all_b[data_all_b["mode"] == 2]


data_m1_b = data_all_b[data_all_b["mode"] == 1]#!!!!!!!!
data_m2_b = data_all_b[data_all_b["mode"] == 2]
data_m3_b = data_all_b[data_all_b["mode"] == 3]

m1_wx_b = data_m1_b["wxwy"]
m2_wx_b = data_m2_b["wxwy"]
m3_wx_b = data_m3_b["wxwy"]

m1_z_b= data_m1_b["z_mid"]
m2_z_b = data_m2_b["z_mid"]

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
    bin_width = 2 # Adjust the bin width as needed
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


plt.axvline(x=np.mean(m1_wx_p_["wxwy"]), color='black', linestyle=':')
plt.axvline(x=np.mean(m1_wx_b_["wxwy"]), color='darkolivegreen', linestyle=':')

plt.axvline(x=np.mean(m2_wx_p_["wxwy"]), color='silver', linestyle='--')
plt.axvline(x=np.mean(m2_wx_b_["wxwy"]), color='lightgreen', linestyle='--')

plt.axvline(x=np.mean(m3_wx_p_["wxwy"]), color='dimgrey', linestyle=('-.'))
plt.axvline(x=np.mean(m3_wx_b_["wxwy"]), color='mediumseagreen', linestyle=('-.'))

plt.axvline(x=np.mean(wx_p_totalmean), color='grey', linestyle='-')
plt.axvline(x=np.mean(wx_b_totalmean), color='darkgreen', linestyle='-')

# print(np.mean(m1_wx_p_["wxwy"]))
# print(np.mean(m1_wx_b_["wxwy"]))


plt.xlabel('$V_i$ (cm/s)')
plt.ylabel('Depth (cm)')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig("figures/hairplot_h.svg", format="svg")

#%%

#for each label, this calulates the average velocity of when a particle is in mode 1 or 2 or 3. m
#multiple labels will have multiple modes

data_m1_p_filtered = data_m1_p[(data_m1_p['z_mid'] >= depth1) & (data_m1_p['z_mid'] <= depth2)]
data_m2_p_filtered = data_m2_p[(data_m2_p['z_mid'] >= depth1) & (data_m2_p['z_mid'] <= depth2)]
data_m3_p_filtered = data_m3_p[(data_m3_p['z_mid'] >= depth1) & (data_m3_p['z_mid'] <= depth2)]


data_m1_b_filtered = data_m1_b[(data_m1_b['z_mid'] >= depth1) & (data_m1_b['z_mid'] <= depth2)]
data_m2_b_filtered = data_m2_b[(data_m2_b['z_mid'] >= depth1) & (data_m2_b['z_mid'] <= depth2)]
data_m3_b_filtered = data_m3_b[(data_m3_b['z_mid'] >= depth1) & (data_m3_b['z_mid'] <= depth2)]

#%%

hist_mode_m1_p = data_m1_p_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()
hist_mode_m2_p = data_m2_p_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()
hist_mode_m3_p = data_m3_p_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()


hist_mode_m1_b = data_m1_b_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()
hist_mode_m2_b = data_m2_b_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()
hist_mode_m3_b = data_m3_b_filtered.groupby('label_mid').agg({"wxwy": 'mean', 'mode': 'first'}).reset_index()


min_value = np.min(data1_b["wx"])
max_value = np.max(data1_b["wx"])
bin_width = binsize
num_bins = int((max_value - min_value) / bin_width)
bin_edges = np.linspace(min_value, max_value, num_bins + 1)

#%%
plt.figure(figsize=(2.65, 3))

plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.hist(hist_mode_m1_p["wxwy"],  density=(True),histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='black', fc='none')
plt.hist(hist_mode_m1_p["wxwy"],  density=(True),bins=bin_edges, lw=1.5, alpha=0.2, color='black')
plt.hist(hist_mode_m2_p["wxwy"],  density=(True),histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='silver', fc='none')
plt.hist(hist_mode_m2_p["wxwy"],  density=(True),bins=bin_edges, lw=1.5, alpha=0.2, color='silver')
plt.hist(hist_mode_m3_p["wxwy"],  density=(True),histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='dimgrey', fc='none')
plt.hist(hist_mode_m3_p["wxwy"],  density=(True),bins=bin_edges, lw=1.5, alpha=0.2, color='dimgrey')

plt.xlim(x_lim)
# plt.yscale('log')
plt.ylim(0, 0.5)
# plt.yticks([1, 100])  # Set y-axis ticks to 0.1 and 100
plt.gca().axes.xaxis.set_ticklabels([])
sns.despine(top=True, right=True, left=False, bottom=False)

plt.axvline(x=np.mean(m1_wx_p_["wxwy"]), color='black', linestyle=':')
plt.axvline(x=np.mean(m2_wx_p_["wxwy"]), color='silver', linestyle='--')
plt.axvline(x=np.mean(m3_wx_p_["wxwy"]), color='dimgrey', linestyle=('-.'))
plt.axvline(x=np.mean(wx_p_totalmean), color='grey', linestyle='-')


# Plotting the second histogram
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.hist(hist_mode_m1_b["wxwy"],  density=(True),histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='darkolivegreen')
plt.hist(hist_mode_m1_b["wxwy"],  density=(True), bins=bin_edges, lw=1.5, alpha=0.2, color='darkolivegreen')
plt.hist(hist_mode_m2_b["wxwy"],  density=(True),histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='lightgreen')
plt.hist(hist_mode_m2_b["wxwy"],  density=(True), bins=bin_edges, lw=1.5, alpha=0.2, color='lightgreen')
plt.hist(hist_mode_m3_b["wxwy"],  density=(True),histtype='step', bins=bin_edges, lw=1.5, alpha=1, ec='mediumseagreen')
plt.hist(hist_mode_m3_b["wxwy"],  density=(True), bins=bin_edges, lw=1.5, alpha=0.2, color='mediumseagreen')
# plt.yscale('log')
plt.xlim(x_lim)
plt.ylim(0, 0.5)
# plt.yticks([1, 100])  # Set y-axis ticks to 0.1 and 100
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xlabel("$V$ (cm/s)")

plt.axvline(x=np.mean(m1_wx_b_["wxwy"]), color='darkolivegreen', linestyle=':')
plt.axvline(x=np.mean(m2_wx_b_["wxwy"]), color='lightgreen', linestyle='--')
plt.axvline(x=np.mean(m3_wx_b_["wxwy"]), color='mediumseagreen', linestyle=('-.'))
plt.axvline(x=np.mean(wx_b_totalmean), color='darkgreen', linestyle='-')

# plt.subplots_adjust(vspace=10)  
plt.tight_layout(h_pad=4.0)  # Increase the vertical space between subplots
plt.savefig("figures/histo_combined_h.svg", format="svg")


#%%



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
mann_whitney_u_test(m1_wx_p_["wxwy"].dropna(), m1_wx_b_["wxwy"].dropna())
# mann_whitney_u_test(m2_wx_p_["wxwy"].dropna(), m2_wx_b_["wxwy"].dropna())
# mann_whitney_u_test(m3_wz_p_["wz"].dropna(), m3_wz_b_["wz"].dropna())



plt.hist(wx_p_totalmean.dropna(),  density=(True),histtype='step', bins=10, lw=1.5, alpha=1, ec='black', fc='none')
plt.hist(wx_p_totalmean.dropna(),  density=(True),bins=10, lw=1.5, alpha=0.2, color='black')

plt.hist(wx_b_totalmean.dropna(),  density=(True),histtype='step', bins=10, lw=1.5, alpha=1, ec='green', fc='none')
plt.hist(wx_b_totalmean.dropna(),  density=(True),bins=10, lw=1.5, alpha=0.2, color='green')
plt.axvline(x=np.mean(np.mean(wx_p_totalmean)), color='dimgrey', linestyle='-')
plt.axvline(x=np.mean(np.mean(wx_b_totalmean)), color='green', linestyle='-')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xlim(0, 5)





#%%
data = [wx_p_totalmean.dropna(), wx_b_totalmean.dropna()]


df = pd.DataFrame({
    'value': wx_p_totalmean.dropna().tolist() + wx_b_totalmean.dropna().tolist(),
    'category': ['Pristine'] * wx_p_totalmean.dropna().shape[0] + ['Biofouled'] * wx_b_totalmean.dropna().shape[0]
})

plt.figure(figsize=(1, 2))
sns.boxplot(x='category', y='value', data=df, palette=['lightgrey', 'yellowgreen'], showfliers=False)

sns.despine(top=True, right=True, left=False, bottom=False)
plt.ylim(0, 20)
plt.savefig("figures/boxplot_h.svg", format="svg")

#%%

plt.figure(figsize=(3, 3))

# Concatenating the data into a single DataFrame with categories
data = pd.concat([wx_p_totalmean.rename('Value'), wx_b_totalmean.rename('Value')],
                  axis=0,
                  keys=['Pristine', 'Biofouled'],
                  names=['Category'])


# Creating the split violin plot
sns.violinplot(data=data.reset_index(), x='Category', y='Value', hue='Category',
                palette=['lightgrey', 'yellowgreen'], split=True, inner="quart")

plt.ylim(0, 30)
plt.savefig("figures/violin_h.svg", format="svg")


# # Displaying the plot
# plt.show()

# plt.figure(figsize=(2, 2))

# # Concatenating the data into a single DataFrame with categories
# data = pd.concat([wx_p_totalmean.rename('Value'), wx_b_totalmean.rename('Value')],
#                   axis=0,
#                   keys=['Pristine', 'Biofouled'],
#                   names=['Category'])


# # Creating the split violin plot
# sns.violinplot(data=data.reset_index(), x='Category', y='Value', hue='Category',
#                 palette=['lightgrey', 'yellowgreen'], split=True, inner="quart")


# plt.hist(wx_p_totalmean.dropna(), bins = 10,alpha = 0.2, color = "b")
# plt.hist(wx_b_totalmean.dropna(), bins = 10,alpha = 0.2, color = "g")
# plt.axvline(x=np.mean(np.mean(wx_p_totalmean)), color='b', linestyle='-')
# plt.axvline(x=np.mean(np.mean(wx_b_totalmean)), color='g', linestyle='-')

# wx_p_totalmean.to_excel("essemabled_average_settling/" + "w_h_p_" + plastic + size + ".xlsx")
# wx_b_totalmean.to_excel("essemabled_average_settling/" + "w_h_b_" + plastic + size + ".xlsx")

# hist_mode_p = pd.concat([hist_mode_m1_p, hist_mode_m2_p, hist_mode_m3_p])
# hist_mode_b = pd.concat([hist_mode_m1_b, hist_mode_m2_b, hist_mode_m3_b])

# hist_mode_p.to_excel("essemabled_average_settling/" + "w_h_p_mode" + plastic + size + ".xlsx")
# hist_mode_b.to_excel("essemabled_average_settling/" + "w_h_b_mode" + plastic + size + ".xlsx")