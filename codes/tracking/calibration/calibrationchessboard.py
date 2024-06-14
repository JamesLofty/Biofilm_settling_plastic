"""
Created on Fri Sep 29 14:18:32 2023

@author: jameslofty
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2

# Define the chessboard pattern size (number of internal corners)
chessboard_size = (9, 6)  # Change this to match your chessboard

# Create lists to store object points and image points from all the calibration images
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Generate a set of object points for the chessboard
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Specify the directory containing your calibration images
# calibration_images_dir = 'pictures_cam1/'
calibration_images_dir = 'pictures_cam2/'


#%%
# Loop through the calibration images
for i in range(1, 18):  # Change the range as needed
    # Load the calibration image
    # image_path = calibration_images_dir + f'picture_cam1_{i}.jpg'
    image_path = calibration_images_dir + f'picture_cam2_{i}.jpg'

    img = cv2.imread(image_path)
    plt.imshow(img)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    print(f'picture_cam_{i}.png', ret)
    
    # If corners are found, add object points and image points
    if ret:
        obj_points.append(objp)
        img_points.append(corners)
        
        # Draw and display the corners (optional)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)  # Display for 500 ms per image

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# Print the calibration results
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", distortion_coeffs)


# Save the calibration results (optional)
np.save('camera_matrix_'+calibration_images_dir[9:-1], camera_matrix)
np.save('distortion_coeffs_'+calibration_images_dir[9:-1], distortion_coeffs)

# Close any open windows (if any)
cv2.destroyAllWindows()

plt.imshow(img)

undistorted_image = cv2.undistort(img, camera_matrix, distortion_coeffs)
plt.imshow(undistorted_image)
