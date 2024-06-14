# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 08:54:13 2023

@author: Lofty
"""

#%%
# Pro libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import os
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs


#%%
# Our functions:
    
def order_the_labels(tp_list, xp_list, yp_list, area_list, aspect_list, labels_all_gmm):
    """
    This function takes labelled data, in a random order, and renumbers them
    following the stream of tp.
    """    
    
    labels_pos = labels_all_gmm[labels_all_gmm>=0]
    tp_pos = tp_list[labels_all_gmm>=0]
    xp_pos = xp_list[labels_all_gmm>=0]
    yp_pos = yp_list[labels_all_gmm>=0]
    area_pos = area_list[labels_all_gmm>=0]
    aspect_pos = aspect_list[labels_all_gmm>=0]
    
    # labels_list = np.arange(500)
    labels_new = []
    # labels_assigned = []
    labels_assigned = np.asarray([], dtype=int)
    
    lbl_k_new = 0
    for i in range(0,len(labels_pos)):
        if labels_pos[i] in labels_assigned:
            
            # I retrieve which is the position within "labels_assigned", = 23
            # Then the new label is: labels.nea.append(23)
            # print("This particle already existed...")
            
            location = labels_assigned == labels_pos[i]
            # the corresponding new label of that old label is: labels_assigned[position][0]
            position = np.argmax(location)        
            labels_new.append(position)
    
        else:
            labels_new.append(lbl_k_new)
            # Keep track:
            # labels_assigned.append(labels_aa[i])
            labels_assigned = np.append(labels_assigned, labels_pos[i])
            # For the next one...
            lbl_k_new = lbl_k_new+1
            
    labels_new = np.asarray(labels_new)            
    
    return tp_pos, xp_pos, yp_pos, area_pos, aspect_pos, labels_new

#%%

file = "2-merged_coords/mergedPS_2x1_pos2_cam2_p.xlsx"

df = pd.read_excel(file)

tp_list = np.asarray(df["tp"])
xp_list = np.asarray(df["xp"])
yp_list = np.asarray(df["yp"])
area_list = np.asarray(df["area"])
aspect_list = np.asarray(df["aspect"])

#%%

plt.figure()
plt.scatter(tp_list, yp_list)

Delta_yp = np.gradient(yp_list)
Delta_tp = np.gradient(tp_list)

# We calculate the angle of our data
theta = np.arctan2(np.median(Delta_yp), np.median(Delta_tp))

# We rotate against the angle.
theta = -theta

# Rotation matrix
R = np.zeros([2, 2])
R[0, 0] = np.cos(theta)
R[0, 1] = -np.sin(theta)
R[1, 0] = np.sin(theta)
R[1, 1] = np.cos(theta)

# Create vector to be rotated
vec_tp_xp = np.asarray([tp_list, yp_list])

# product of two arrays (rotation matrix and data vector)
ptxstar = np.dot(R, vec_tp_xp)

# re define tp and xp
tpx = ptxstar[1, :]
ypx = ptxstar[0, :]

plt.figure()
plt.scatter(tpx, ypx)

#%%

# aa = np.column_stack((tpx, ypx))


# clustering = OPTICS(min_samples=20, xi=.05, min_cluster_size=.05)
# clustering.fit(aa)


# plt.scatter(tpx, ypx, color=colors[clustering.labels_].tolist(), s=50)


#%%%
X = np.zeros([len(yp_list), 1])
X[:, 0] = tpx

# gaussian mixture model clustering
gmm = GMM(n_components=31, covariance_type='full', random_state=8).fit(X)
labels_all_gmm = gmm.predict(X)

plt.figure(figsize=(4,7))
plt.scatter(yp_list, 
            tp_list, 
            c=labels_all_gmm)
plt.gca().invert_yaxis()

plt.figure(figsize=(3,6))
plt.scatter(xp_list, 
            yp_list, 
            c=labels_all_gmm, 
            s = 3)
plt.gca().invert_yaxis()

#%%
"""
Here we take the unodered labels and order them by tp starting at 1
"""

tp_list_ordered, xp_list_ordered, yp_list_ordered, area_list_ordered, aspect_list_ordered, labels_all_gmm_ordered = \
    order_the_labels(tp_list, xp_list, yp_list, area_list, aspect_list, labels_all_gmm)
    

    
#%%

# for i in np.unique(labels_all_gmm_ordered):
#     plt.figure(figsize=(3,6))
#     plt.scatter(xp_list_ordered[labels_all_gmm_ordered==i], 
#                 yp_list_ordered[labels_all_gmm_ordered==i], 
#                 c=labels_all_gmm_ordered[labels_all_gmm_ordered==i], 
#                 s = 3)
#     plt.gca().invert_yaxis()
#     plt.legend()
    


results = pd.DataFrame(list(zip(tp_list_ordered, xp_list_ordered, yp_list_ordered, area_list_ordered, aspect_list_ordered, labels_all_gmm_ordered )),
                       columns=['tp', 'xp', 'yp', "area", "aspect", "label"])
if "cam1" in file:
    camera = "cam1"

if "cam2" in file:
    camera = "cam2"

results.to_excel("3-labelled_coords/" + "labeled" + file[22:])
# results.to_excel("3-labelled_coords/" + "labeled" + "cam1.xlsx")
