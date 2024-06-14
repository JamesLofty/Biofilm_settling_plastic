#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:43:58 2024

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


_1 = pd.read_excel("5 - modes/modePTFE_2x1_pos1_p.xlsx")
_2 = pd.read_excel("5 - modes/modePTFE_2x1_pos2_p.xlsx")
_3 = pd.read_excel("5 - modes/modePTFE_2x1_pos3_p.xlsx")

_2["label_mid"] = _2["label_mid"] + 100
_3["label_mid"] = _3["label_mid"] + 1000

dataaaa = pd.concat([_1, _2, _3])

x = dataaaa["x_mid"]
y = dataaaa["y_mid"]
z = dataaaa["z_mid"]
label = dataaaa["label_mid"]

fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot(projection='3d')

for i in label.unique(): 
    ax.plot(x[label == i], y[label == i], z[label == i], lw = 0.8)
    ax.scatter(x[label == i ], y[label == i], z[label == i], s = 0.8)
    
ax.set_zlim(0, 25)

ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xticks([-5, 0, 5])
ax.set_yticks([-5, 0, 5])

ax.grid(False)
ax.invert_zaxis()
ax.set_box_aspect(aspect=(0.8,0.8, 2))
ax.view_init(10, 225)
plt.savefig("figures/3d tracj_all.svg", format="svg")
