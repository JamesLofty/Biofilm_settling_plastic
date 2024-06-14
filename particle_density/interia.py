#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:27:35 2024

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

H = 0.01
W = 0.01


Ix = (1/12) * (H**3) * W
print(Ix*100000000)

Iy = (1/12) * (W**3) * H
print(Iy*100000000)

J = ((W*H) / 12) * ((W**2) + (H**2))
print(J*10000000)

A = H*W
dm = math.sqrt((4*A) / math.pi)
print(dm)

t = 0.001

# I_di = (math.pi * 0.001 * 1950) / (64 * 1000 * dm)
# I_di = (math.pi / 64) * (1950 / 1000) * (t / dm)


# print(I_di)

I_d2 = (math.pi / 64) * (1950 / 1000) * (t / W)
print(I_d2)