#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:38:09 2023

@author: jameslofty
"""

#%%
# Pro libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy import stats

#%%
# def Dist2(p1, p2):
#     return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

# def fuse(points, d):
#     ret = []
#     d2 = d * d
#     n = len(points)
#     taken = [False] * n
#     for i in range(n):
#         if not taken[i]:
#             count = 1
#             point = [points[i][0], points[i][1]]
#             taken[i] = True
#             for j in range(i+1, n):
#                 if Dist2(points[i], points[j]) < d2:
#                     point[0] += points[j][0]
#                     point[1] += points[j][1]
#                     count+=1
#                     taken[j] = True
#             point[0] /= count
#             point[1] /= count
#             ret.append((point[0], point[1]))
#     return ret


#%%



file = "1-tracking_coords/PS_2x1_pos2_cam2_p.xlsx"


df = pd.read_excel(file)

if "cam1" in file:
    camera = "cam1"

if "cam2" in file:
    camera = "cam2"

transform_px_cam1 = 45
transform_px_cam2 = 43
FPS = 30

if camera == "cam1":
    tp_list = np.asarray(df["tp"])/30
    xp_list = np.asarray(df["xp"])/transform_px_cam1
    yp_list = np.asarray(df["yp"])/transform_px_cam1
    area_list = np.asarray(df["area"])/transform_px_cam1
    aspect_list = np.asarray(df["aspect"])
    # pix_intensity_list = np.asarray(df["avg_intensity"])

if camera == "cam2":
    tp_list = np.asarray(df["tp"])/30
    xp_list = np.asarray(df["xp"])/transform_px_cam2
    yp_list = np.asarray(df["yp"])/transform_px_cam2
    area_list = np.asarray(df["area"])/transform_px_cam2
    aspect_list = np.asarray(df["aspect"])
    # pix_intensity_list = np.asarray(df["avg_intensity"])


plt.figure(figsize=(6,6))
plt.scatter(xp_list, yp_list, c = tp_list, s = 3)
plt.gca().invert_yaxis()

plt.figure(figsize=(6,6))
plt.scatter(tp_list, yp_list, c = tp_list, s = 3)
plt.xlabel("tp")
plt.ylabel("z")

plt.figure(figsize=(6,6))
plt.scatter(area_list, xp_list, c = tp_list, s = 3)
plt.xlabel("area")
plt.ylabel("x")

print("data length = ", len(xp_list))

y_diff = np.diff(yp_list)
t_diff = np.diff(tp_list)

yv = y_diff/t_diff

#%%
"""trim?"""

#%%
####filtering. if there is two coordinates in the frame, chooose the one between a range,
#### if not, choose the coordinate with the largest area



data_dict = {}
for frame, x, y, area, aspect in zip(tp_list, xp_list, yp_list, area_list, aspect_list):
    if frame not in data_dict:
        data_dict[frame] = []

    data_dict[frame].append((x, y, area, aspect))

filtered_data = []
for frame, coordinates in data_dict.items():
    filtered_coordinates = []
    for x, y, area, aspect in coordinates:
        if 12<= x <= 18 and 20 <= area <= 200:
            filtered_coordinates.append((x, y, area, aspect))

#     if filtered_coordinates:
#         max_intensity = min(filtered_coordinates, key=lambda item: item[4])
#         filtered_data.append((frame, max_intensity[0], max_intensity[1], max_intensity[2], max_intensity[3]))

# filtered_frame_numbers, filtered_x_coordinates, filtered_y_coordinates, filtered_area_list, filtered_aspect_list = zip(*filtered_data)


    if filtered_coordinates:
        max_area = max(filtered_coordinates, key=lambda item: item[2])
        filtered_data.append((frame, max_area[0], max_area[1], max_area[2], max_area[3]))

filtered_frame_numbers, filtered_x_coordinates, filtered_y_coordinates, filtered_area_list, filtered_aspect_list = zip(*filtered_data)


#%%
plt.figure(figsize=(6,6))
plt.scatter(filtered_frame_numbers, filtered_y_coordinates, c = filtered_frame_numbers, s = 3)
plt.xlabel("tp")
plt.ylabel("z")

plt.figure(figsize=(6,6))
plt.scatter(filtered_area_list, filtered_x_coordinates, c = filtered_area_list, s = 3)
plt.xlabel("area")
plt.ylabel("x")
plt.gca().invert_yaxis()

plt.figure(figsize=(6,6))
plt.scatter(filtered_x_coordinates, filtered_y_coordinates, c = filtered_frame_numbers, s = 3)
plt.xlabel("x")
plt.ylabel("z")
plt.gca().invert_yaxis()



print("data length = ", len(filtered_x_coordinates))


#%%
tp_cleaned = filtered_frame_numbers
xp_cleaned = filtered_x_coordinates
yp_cleaned = filtered_y_coordinates
area_cleaned = filtered_area_list
aspect_cleaned = filtered_aspect_list




#%%
results = pd.DataFrame(list(zip(tp_cleaned, xp_cleaned, yp_cleaned,
                                area_cleaned, aspect_cleaned)),
                      columns=['tp', 'xp', 'yp', "area", "aspect"])


# # folder = "3 - coordinates_cleaned"
# # filename = 

results.to_excel("2-merged_coords/" + "merged" + file[18:])
# results.to_excel("2-merged_coords/" + "merged" + "cam2.xlsx")


        