#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:42:15 2023

@author: jameslofty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft
from numpy.fft import fft, ifft
import seaborn as sns
from scipy.signal import find_peaks
import math
import scipy.stats
from itertools import zip_longest
from scipy.stats import mannwhitneyu



def find_top_peaks(data, num_peaks=1, prominence_threshold=0.005):
    peaks, _ = find_peaks(data, prominence=prominence_threshold)
    top_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:num_peaks]
    return top_peaks

file1 = "5 - modes/modePTFE_2x1_pos2_b.xlsx"
file2 = "5 - modes/modePTFE_2x1_pos2_p.xlsx"


data2_b = pd.read_excel(file1)

data2_p = pd.read_excel(file2)

data_b = data2_b
data_b["_b"] = "_b"

data_p = data2_p
data_p["_p"] = "_p"

files = [data_p, data_b]

plt.figure(figsize=(2,2))

top_peaks_val = []
w = []

peak_freq_b_1 = []
peak_freq_p_1 = []

peak_freq_b_2 = []
peak_freq_p_2 = []

# w_b = []
# w_p = []

for file in files:
    data = file
    
    data = data[data["mode"]==1]
    
    X = np.sqrt(data["x_mid"]**2 + data["y_mid"]**2)
    Y = np.sqrt(data["x_mid"]**2 + data["y_mid"]**2)

    
        
    #     data["x_mid"] / 100)
    # Y = np.array(data["y_mid"] / 100)
    
    label = data["label_mid"]
    tp = data["tp_mid"]
    wz = data["wz"]

    """FFT for both cameras"""
    FFT_freq_x = []
    FFT_X_x = []

    FFT_freq_y = []
    FFT_X_y = []


    for i in label.unique():

        t = tp[label == i]
        x =  X[label == i]
        wz = data["wz"][label == i]
        
        sr = 30
        
        X1 = fft(x)
        N = len(X1)
        n = np.arange(N)
        T = N/sr
        freq1 = n/T 
        
        mask = np.logical_and(freq1 >= 0, freq1 <= 15)
        X1 = X1[mask]
        freq1 = freq1[mask]

        FFT_X_x.append(np.abs(X1))
        
        if "_b" in data.columns:
            plt.plot(freq1, np.abs(X1), 'yellowgreen', alpha = 0.2, zorder = 6)
            
            #find top peaks
            top_peaks = find_top_peaks(X1, num_peaks=1, prominence_threshold=0.005)
            top_peaks_num = freq1[top_peaks]
            peak_freq_b_1.append(top_peaks_num)
            
            w_b = data.groupby(['label_mid'])['wz'].mean().reset_index()
            
        else:
            plt.plot(freq1, np.abs(X1), 'grey', alpha = 0.2)
            
            #find top peaks
            top_peaks = find_top_peaks(X1, num_peaks=1, prominence_threshold=0.005)
            top_peaks_num = freq1[top_peaks]
            peak_freq_p_1.append(top_peaks_num)
            
            w_p = data.groupby(['label_mid'])['wz'].mean().reset_index()

        
    for i in label.unique():
   
        t = tp[label == i]
        y = Y[label == i]
        
        sr = 30
        
        X2 = fft(y)
        N = len(X2)
        n = np.arange(N)
        T = N/sr
        freq2 = n/T 
        
        mask = np.logical_and(freq2 >= 0, freq2 <= 15)
        X2 = X2[mask]
        freq2 = freq2[mask]

        FFT_freq_y.append(freq2)
        FFT_X_y.append(np.abs(X2))
    
        
        if "_b" in data.columns:
            plt.plot(freq2, np.abs(X2), 'yellowgreen', alpha = 0.2, zorder = 6)
            
            #find top peaks
            top_peaks = find_top_peaks(X2, num_peaks=1, prominence_threshold=0.005)
            top_peaks_num = freq2[top_peaks]
            peak_freq_b_2.append(top_peaks_num)

        else:
            plt.plot(freq2, np.abs(X2), 'grey', alpha = 0.2)
            
            #find top peaks
            top_peaks = find_top_peaks(X2, num_peaks=1, prominence_threshold=0.005)
            top_peaks_num = freq2[top_peaks]
            peak_freq_p_2.append(top_peaks_num)

        plt.xlabel('Freq (Hz)')
        plt.ylabel('|FFT Amplitude ($r$)|')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(0.2, 10)
        plt.ylim(1, 100)
        sns.despine(top=True, right=True, left=False, bottom=False)
        # plt.legend()
        

    FFT_freq1 = []
    FFT_freq2 = []
    
    FFT_freq1.extend(FFT_freq_x)
    FFT_freq2.extend(FFT_freq_y)
    
    FFT_X1 = []
    FFT_X2 = []
    
    FFT_X1.extend(FFT_X_x)
    FFT_X2.extend(FFT_X_y)
    
        
    """find ensemble average"""
    
    all_FFT_freq = pd.concat([pd.Series(FFT_freq1), pd.Series(FFT_freq2)], ignore_index=True)
    
    max_length_freq = max(len(arr) for arr in all_FFT_freq)
    padded_arrays_freq = np.full((len(all_FFT_freq), max_length_freq), np.nan)
    for i, arr in enumerate(all_FFT_freq):
        padded_arrays_freq[i, :len(arr)] = arr
    
    FFT_freq_avg = np.nanmean(padded_arrays_freq, axis=0)
    
    all_FFT_X = pd.concat([pd.Series(FFT_X_x), pd.Series(FFT_X_y)], ignore_index=True)
    
    max_length_X = max(len(arr) for arr in all_FFT_X)
    padded_arrays_X = np.full((len(all_FFT_X), max_length_X), np.nan)
    for i, arr in enumerate(all_FFT_X):
        padded_arrays_X[i, :len(arr)] = arr
    
    # Calculate ensemble average for FFT X
    FFT_X_avg = np.nanmean(padded_arrays_X, axis=0)

    
    if "_p" in data.columns:
        plt.plot(FFT_freq_avg, np.abs(FFT_X_avg), 'dimgrey', lw = 2.5, zorder = 9)

    else:
        plt.plot(FFT_freq_avg, np.abs(FFT_X_avg), 'green', lw = 2.5,  zorder = 9)

    plt.savefig("figures/FFT_python.svg", format="svg")
    

    """Strouhal number"""
d = 1

peak_freq_p_1 = np.concatenate(peak_freq_p_1).ravel()
peak_freq_p_2 = np.concatenate(peak_freq_p_2).ravel()
freq_p = np.concatenate([peak_freq_p_1, peak_freq_p_2])
print("freq_p", round(np.mean(freq_p), 2), round(np.std(freq_p), 2))

w_p = np.array(w_p["wz"])

peak_freq_b_1 = np.concatenate(peak_freq_b_1).ravel()
peak_freq_b_2 = np.concatenate(peak_freq_b_2).ravel()
freq_b = np.concatenate([peak_freq_b_1, peak_freq_b_2])
print("freq_b", round(np.mean(freq_b), 2), round(np.std(freq_b), 2))

# plt.vlines(np.mean(freq_p), 0, 100, zorder = 11, color = "dimgrey", ls = "dashed", alpha = 1)
# plt.vlines(np.mean(freq_b), 0, 100, zorder = 11, color = "green", ls = "dashed", alpha = 1)


w_b = np.array(w_b["wz"])


st_b_1 = (peak_freq_b_1 * d) / w_b
st_b_2 = (peak_freq_b_2 * d) / w_b

# [1:]
st_p_1 = (peak_freq_p_1 * d) / w_p
st_p_2 = (peak_freq_p_2 * d) / w_p

st_p = np.concatenate((st_p_1, st_p_2))
st_b = np.concatenate((st_b_1, st_b_2))

st_p_mean = np.mean(st_p)
st_b_mean = np.mean(st_b)

st_p_std = np.std(st_p)
st_b_std = np.std(st_b)

print("st_p", round(st_p_mean, 3), round(st_p_std, 3))
print("st_b", round(st_b_mean, 3),  round(st_b_std, 3))
    
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


print("freq")
mann_whitney_u_test(freq_b, freq_p)
print("str")
mann_whitney_u_test(st_b, st_p)

#%%

# plt.figure(figsize = (2,2))
# plt.hist(freq_b, bins = 10, color = "grey", histtype='step',alpha = 1, lw=2)
# plt.hist(freq_p, bins = 10, color = "yellowgreen", histtype='step',alpha = 1,lw=2)
# plt.ylim(0, 60)
# # plt.xlim(0, 5)
# sns.despine(top=True, right=True, left=False, bottom=False)

# mean_p = np.mean(freq_p)
# mean_b = np.mean(freq_b)

# plt.axvline(mean_p, color='dimgrey', linestyle='dashed', linewidth=2, label='Mean P')
# plt.axvline(mean_b, color='green', linestyle='dashed', linewidth=2, label='Mean B')


#%%
# plt.figure()
# plt.hist(freq_b, color = "green", alpha = 0.2)
# plt.hist(freq_p, color = "grey", alpha = 0.2)

# plt.figure()
# plt.hist(st_b, color = "green", alpha = 0.2)
# plt.hist(st_p, color = "grey", alpha = 0.2)

zipped_data = zip_longest(freq_b, freq_p, st_b, st_p)

results = pd.DataFrame(zipped_data, columns=['freq_b', 'freq_p', 'st_b', 'st_p'])



if "PS" in file1:
    results.to_excel("freq+strou_numbers/" + file1[14:20] + ".xlsx" )
if "POM" in file1:
    results.to_excel("freq+strou_numbers/" + file1[14:21] + ".xlsx" )
if "PTFE" in file1:
    results.to_excel("freq+strou_numbers/" + file1[14:22] + ".xlsx" )
print("yeeeii")

