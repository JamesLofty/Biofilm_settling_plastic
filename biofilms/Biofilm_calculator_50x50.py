

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:18:59 2024

@author: jameslofty
"""
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft
from numpy.fft import fft, ifft
import seaborn as sns
from scipy.signal import find_peaks
import math
import imutils

def calculate_area(image, threshold):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([120, 50, 0])
    upper_purple = np.array([200, 255, 255])

    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    result = cv2.bitwise_and(image, image, mask=mask_purple)

    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(result_gray, threshold, 255, cv2.THRESH_BINARY)
    
    area_purple = cv2.countNonZero(binary_mask)
    area_non_purple = np.sum(binary_mask == 0)

    return area_purple, area_non_purple, binary_mask

def straighten_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

def plot_cross_section_intensity(image):
    # Get the shape of the image
    height, width = image.shape[:2]

    # Calculate the x and y cross-sections
    x_cross_section = np.mean(image, axis=0)  # Along columns
    y_cross_section = np.mean(image, axis=1)  # Along rows
    
    x_nonzero = x_cross_section[x_cross_section != 0]
    y_nonzero = y_cross_section[y_cross_section != 0]

    # Plot x cross-section
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_nonzero, color='blue')
    plt.title('X Cross-Section Intensity')
    plt.xlabel('Distance')
    plt.ylabel('Pixel Intensity')
    plt.ylim(0, 255)  # Adjust the y-axis limit if needed

    # Plot y cross-section
    plt.subplot(1, 2, 2)
    plt.plot(y_nonzero, color='red')
    plt.title('Y Cross-Section Intensity')
    plt.xlabel('Distance')
    plt.ylabel('Pixel Intensity')
    plt.ylim(0, 255)  # Adjust the y-axis limit if needed

    plt.tight_layout()
    plt.show()
    
    # Calculate mean and standard deviation
    mean_x_intensity = np.mean(x_nonzero)
    std_x_intensity = np.std(x_nonzero)
    mean_y_intensity = np.mean(y_nonzero)
    std_y_intensity = np.std(y_nonzero)
    
    return mean_x_intensity, std_x_intensity, mean_y_intensity, std_y_intensity
    
def plot_intensity_colormap(image):
    normalized_image = image / 255.0
    
    # Apply colormap
    inverted_image = 1 - normalized_image    
    
    # Display the inverted colormap image
    plt.imshow(inverted_image, cmap='viridis_r', vmin=0, vmax=1)  # Adjust vmin and vmax to the desired range
    # plt.title('Inverted Intensity Colormap')
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Adjust the ticks as needed
    plt.show()

#%%
# spectrum = np.zeros((50, 1, 3), dtype=np.uint8)

# for i in range(50):
#     spectrum[i, 0, :] = [int(lower_purple[0] + i * (upper_purple[0] - lower_purple[0]) / 50), upper_purple[1], upper_purple[2]]

# # Convert the spectrum to RGB for display
# spectrum_rgb = cv2.cvtColor(spectrum, cv2.COLOR_HSV2RGB)

# # Display the color spectrum
# plt.imshow(np.reshape(spectrum_rgb, (50, 1, 3)))
# plt.title('Purple Color Spectrum')
# plt.axis('off')
# plt.show()


#%%

file = "images/PTFE_50x50.jpg"
image = cv2.imread(file)

image = cv2.convertScaleAbs(image, alpha = 1, beta = 1)

plt.imshow(image)
plt.title('image')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
thresh_frame = cv2.threshold(src=blurred, thresh=10, maxval=255, type=cv2.THRESH_BINARY)[1]

plt.imshow(thresh_frame)
plt.show()

# dilate_kernel = np.ones((30, 30))
# dilute_frame = cv2.dilate(thresh_frame, dilate_kernel, iterations=1)

# erosion_kernel = np.ones((10, 10))
# erosion_frame = cv2.erode(thresh_frame, erosion_kernel, iterations=1)

contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()

biofilm_percentages = []
biofilm_intensities = []

for contour in contours:
    area = cv2.contourArea(contour)
    print(area)
    if 1000000 <= area <= 80000000:
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 5)  # Draw only if area is larger than 10000

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours (Thresholded)')
plt.show()

plt.imshow(contour_image)
plt.title('Image with Contours (Area > 10000)')
plt.show()

biofilm_percentages = []
biofilm_intensities = []
x_mean_intensities = []
y_mean_intensities = []
x_std_intensities = []
y_std_intensities = []
std = []
#%%
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 100000 <= area <= 8000000:
        # Create an empty mask
        mask = np.zeros_like(gray)

        (x, y, w, h) = cv2.boundingRect(contour)
        center = (int(x + w / 2), int(y + h / 2))
        radius = min(w, h) // 2
        cv2.circle(mask, center, radius, (255), thickness=cv2.FILLED)

        # Apply the circular mask to the image
        cropped_image1 = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop the circular region
        _, thresh = cv2.threshold(cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = cropped_image1[y:y + h, x:x + w]
        
        plt.imshow(cropped_image1)
        plt.title('cropped whole image')
        plt.show()

        plt.imshow(cropped_image)
        plt.title('cropped_image')
        plt.show()

        # plt.imshow(mask, cmap='gray')
        # plt.title('Original Image with Contours')
        # plt.show()
#%%
        
        # rotated_image = straighten_image(cropped_image)
        rotated_image = cropped_image

        # plt.imshow(rotated_image, cmap='gray')
        # plt.title('roated image')
        # plt.show()


#%% 
        #% of biofilm
        area_purple, area_non_purple, binary_mask = calculate_area(rotated_image, threshold=0)
        
        total_area = area_purple ######not right !!!! but
        
        purple_percentage = (area_purple / total_area) *100
        
        biofilm_percentages.append(purple_percentage)

        print("% biofilm = ", f"{purple_percentage:.2f}%")
        
        #%%
        
        rotated_image_grey = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        
        trimmed_image =  rotated_image_grey[50:-50, 50:-50]
        
        mask = trimmed_image != 0
        
        hist = cv2.calcHist([trimmed_image[mask]], [0], None, [256], [0, 256])

        x_values = 1- np.arange(256) / 255.0

        # Plot histogram
        plt.figure(figsize=(1.5, 2))
        plt.plot(x_values, hist, color='black')
        plt.xlabel('Normalised pixel intensity')
        plt.ylabel('Frequency')
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.show()
        
        # Calculate standard deviation
        # Calculate the standard deviation while ignoring zeros
        std_deviation = np.std(trimmed_image[mask]) / 255

        print("Standard Deviation (ignoring zeros):", std_deviation)
       
        
        std.append(std_deviation)
    
        # middle_x = rotated_image_grey.shape[1] // 2
        # middle_y = rotated_image_grey.shape[0] // 2
  
        
        # x__intensity = rotated_image_grey[:, middle_x]
        # x__intensity = x__intensity[50:-50]
            
        # # Calculate the  pixel intensity along the y-axis (vertical)
        # y__intensity = rotated_image_grey[middle_y, :]
        # y__intensity = y__intensity[50:-50]

        # # Plotting the mean intensity along the x and y cross-sections
        # plt.figure(figsize=(2.5,3))
        # plt.plot(range(len(x__intensity)), x__intensity/255, c="blue", label="y axis")
        # plt.plot(range(len(y__intensity)), y__intensity/255, c="red", label="x axis")
        # plt.legend()
        # sns.despine(top=True, right=True, left=False, bottom=False)
        # # plt.xlim(0,600)
        # plt.ylim(0.1,0.9)
        # plt.show()

        # print(np.std(x__intensity/255))
        # print(np.std(y__intensity/255))
    
    
        # x_std_intensities.append(np.std(x__intensity/255))
        # y_std_intensities.append(np.std(y__intensity/255))
    
    
    #%%

        plot_intensity_colormap(rotated_image_grey)
      
        
        # height, width = rotated_image_grey.shape[:2]
        
        # # cv2.line(rotated_image_grey, (width // 2, 0), (width // 2, height), (0, 0, 255), 9)

        # # Draw a line down the center of the x-axis (blue line)
        # # cv2.line(rotated_image_grey, (0, height // 2), (width, height // 2), (255, 0, 0), 9)
        
        # plt.imshow(rotated_image_grey)
        # plt.title('Rotated Image with Red and Blue Lines')
        # plt.show()
                
#%%
        result_image = cv2.bitwise_and(rotated_image, rotated_image, mask=binary_mask)
        
        plt.imshow(binary_mask, cmap="gray")
        plt.title(f'biofilm{i+1}')
        plt.show()
        
        plt.imshow(result_image)
        plt.title(f'biofilm{i+1}')
        plt.show()
        
        result_image_grey = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

        plot_intensity_colormap(result_image_grey)
        
        # Calculate mean intensity
        non_zero_values = result_image[result_image != 0]
        mean_biofilm_intensity = np.mean(non_zero_values)
        # mean_biofilm_intensity = np.mean(result_image)
        mean_biofilm_intensity = (mean_biofilm_intensity / 255) *100

        biofilm_intensities.append(mean_biofilm_intensity)
        
        print(f'Mean biofilm intensity{i+1}: {mean_biofilm_intensity} %')

#%%

# Plot the standard deviations against distances from the center
mean_percentage = np.mean(biofilm_percentages)
std_percentage = np.std(biofilm_percentages)

mean_biofilm_intensity = np.mean(biofilm_intensities)
std_biofilm_intensity = np.std(biofilm_intensities)

# zipped_data = zip(x_std_intensities, y_std_intensities)
# std_intensity_means = [np.mean(pair) for pair in zipped_data]

# std_intensity = np.mean([std_intensity_means])
# std_std_intensity = np.std([std_intensity_means])




print(f"Mean Biofilm Percentages: {np.round(mean_percentage, 0)}", f"{np.round(std_percentage, 0)}")

print(f"Mean Biofilm Intensities: {np.round(mean_biofilm_intensity, 0)}", f"{np.round(std_biofilm_intensity, 0)}")

# print(f"Mean Intensity std: {np.round(std_intensity, 0)}", f"{np.round(std_std_intensity, 0)}")

results = pd.DataFrame(list(zip(biofilm_percentages, biofilm_intensities, std)),
                        columns=['biofilm_percentages', 'biofilm_intensities', 'std_intensity_means'])

print(results)

if "PS" in file:
    results.to_excel(file[7:17] + ".xlsx")
if "POM" in file:
    results.to_excel(file[7:16] + ".xlsx")
if "PTFE" in file:
    results.to_excel(file[7:19] + ".xlsx")





