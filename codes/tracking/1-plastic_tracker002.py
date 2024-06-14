
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:43:39 2023

@author: jameslofty
"""
#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
#%%
folder = "1-tracking_coords/"

# videos = "/Volumes/Seagate Basic/biofilm_settling/videos/"
# plastic = "PS"
# size = "2x1"
# pos = "pos1"
# camera = "Cam1"

file = "/Volumes/Seagate Basic/biofilm_settling/videos/PS_2x1_pos2_cam2_bio.mp4"

# bs = cv2.createBackgroundSubtractorMOG2(0,cv2.THRESH_BINARY,1)
bs = cv2.createBackgroundSubtractorKNN()

#%%

if "cam1" in file:
    camera = "cam1_"
if "cam2" in file:
    camera = "cam2_"
    
if "pos1" in file:
    pos = "pos1_"
if "pos2" in file:
    pos = "pos2_"
if "pos3" in file:
    pos = "pos3_"

if "2x1" in file:
    size = "2x1_"
if "1x1" in file:
    size = "1x1_"
if "50x50" in file:
    size = "05x05_"
    
if "PTFE" in file:
    plastic = "PTFE_"
if "POM" in file:
    plastic = "POM_"
if "PS" in file:
    plastic = "PS_"
if "PA" in file:
    plastic = "PA_"
    
if "bio" in file:
    plastic_type = "b"
if "pris" in file:
    plastic_type = "p"

     
filename = plastic + size + pos + camera + plastic_type

cap = cv2.VideoCapture(file)
t_end =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("t_end = ", t_end)

previous_frame = None
tp_list = []
xp_list = []
yp_list = []
area_list = []
detections = []
aspect_list = []
avg_pixel_intensity = []

frame_ID = 0

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

while(cap.isOpened()):
    ret, frame = cap.read()
    # cv2.normalize(frame, frame, 80, 170, cv2.NORM_MINMAX)
    cv2.normalize(frame, frame, 50, 160, cv2.NORM_MINMAX)
    frame_ID += 1
    print(frame_ID)
    
    if ret == False:
        results = pd.DataFrame(list(zip(tp_list, xp_list, yp_list, area_list, aspect_list)),
                              columns=['tp', 'xp', 'yp', "area", "aspect"])
        results.to_excel(folder + filename + ".xlsx")
        print("all data saved !!!")
    
    if "cam1" in file: 
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        camera_matrix= np.load("calibration/camera_matrix_cam1.npy")
        dist_coeffs = np.load("calibration/distortion_coeffs_cam1.npy")
        
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        framesize = np.shape(frame)
        roi = np.zeros(framesize[:2], dtype = 'uint8')
        roi = cv2.rectangle(roi, (150, 550), (950, 1620), (255,0,0), -1)  
        frame_roi = cv2.bitwise_and(frame, frame, mask = roi)
       
    elif "cam2" in file: 
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        camera_matrix= np.load("calibration/camera_matrix_cam2.npy")
        dist_coeffs = np.load("calibration/distortion_coeffs_cam2.npy")

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        framesize = np.shape(frame)
        roi = np.zeros(framesize[:2], dtype = 'uint8')
        roi = cv2.rectangle(roi, (150, 550), (950, 1620), (255,0,0), -1)  
        frame_roi = cv2.bitwise_and(frame, frame, mask = roi)
        
    
    # # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # hsv_filtered_frame = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        
    frame_clahe = clahe.apply(frame_gray)

    # frame_eqil = cv2.equalizeHist(frame_roi)
    # 
    # frame_blur = cv2.GaussianBlur(src=frame_clahe, ksize=(5, 5), sigmaX=0)

    # # _, frame_thresholded = cv2.threshold(frame_blur, lower_bound, upper_bound, cv2.THRESH_BINARY)

    prepared_frame = frame_clahe
    
    fgmask = bs.apply(prepared_frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    dilate_kernel = np.ones((25, 25))
    dilute_frame = cv2.dilate(fgmask, dilate_kernel, iterations=1)

    # erosion_kernel = np.ones((5, 5))
    # erosion_frame = cv2.erode(dilute_frame, erosion_kernel, iterations=1)

    # Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=dilute_frame, thresh=2, maxval=255, type=cv2.THRESH_BINARY)[1]

    # Find contours
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    

#%%
#filtering

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 <= area <= 6000:
            (x, y, w, h) = cv2.boundingRect(contour)
    
            # Calculate the average color intensity within the contour
            average_color = np.mean(frame_roi[y:y+h, x:x+w])
    
            white_threshold =0
    
            # If the average color intensity is above the threshold, consider it as a white object
            if average_color > white_threshold:
                detections.append([x, y, w, h])
                aspect = float(w) / h
    
                tp = frame_ID
                xp = np.mean(contour[:, 0], axis=0)[0]
                yp = np.mean(contour[:, 0], axis=0)[1]
                area = cv2.contourArea(contour)
    
                tp_list.append(tp)
                xp_list.append(xp)
                yp_list.append(yp)
                area_list.append(area)
                aspect_list.append(aspect)
    
                cv2.rectangle(img=frame_roi, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)

#%%
    # prepare = cv2.resize(prepared_frame, (500, 900))
    # # grey = cv2.resize(frame_thresholded, (500, 900))
    # thresh_frame = cv2.resize(thresh_frame, (500, 900))
    # frame_roi = cv2.resize(frame_roi, (500, 900))
    # frame = cv2.resize(frame, (500, 900))

    # cv2.imshow('prepare', prepare)
    # # cv2.imshow('gray', grey)
    # cv2.imshow('thres', thresh_frame)
    # cv2.imshow('Image', frame_roi)
    # # cv2.imshow('Image', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

#%%
plt.figure(figsize=(10,5))
plt.scatter(tp_list, yp_list, c = area_list)
plt.gca().invert_yaxis()
plt.xlabel("tp")
plt.ylabel("zp")

plt.figure(figsize=(5,10))
scatter = plt.scatter(xp_list, yp_list, c=area_list, s = 3)
plt.gca().invert_yaxis()
plt.xlabel("xp")
plt.ylabel("zp")
colorbar = plt.colorbar(scatter)

plt.figure(figsize=(5,10))
plt.scatter(xp_list, yp_list, s = 3)
plt.gca().invert_yaxis()
plt.xlabel("xp")
plt.ylabel("zp")

y = np.diff(np.array(yp_list))
t = np.diff(np.array(tp_list))

w = y/t

plt.figure(figsize=(5,10))

plt.scatter(w, np.array(yp_list)[1:], s = 3)
plt.gca().invert_yaxis()
plt.xlim(-10,100)
plt.xlabel("w (pixel/frame)")
plt.ylabel("zp (pixels)")

plt.figure()
plt.scatter(tp_list, xp_list, c = area_list)
plt.gca().invert_yaxis()
plt.xlabel("tp")
plt.ylabel("yp")




