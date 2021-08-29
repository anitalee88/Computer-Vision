#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)
        
#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""

# Write your code here
image_num = len(images)
corner_num = corner_x * corner_y
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
imgpoints = imgpoints.reshape(10, 49, 2)

# H
def find_homography(obj_points, img_points):
    A = []
    for i in range(obj_points.shape[0]):
        u1, v1, _ = obj_points[i]
        u2, v2 = img_points[i]
        A.append([u1, v1, 1, 0, 0, 0, -u1 * u2, -u2 * v1, -u2])
        A.append([0, 0, 0, u1, v1, 1, -u1 * v2, -v2 * v1, -v2])   
    A = np.array(A)
    _, _, vt = np.linalg.svd(A)

    x = vt[-1]
    x = x / x[-1]
    x = x.reshape(3,3)
    return x

# V
def find_V(hi, hj):
    A = []
    A.append([hi[0] * hj[0], 
              hi[0] * hj[1] + hi[1] * hj[0], 
              hi[1] * hj[1], 
              hi[2] * hj[0] + hi[0] * hj[2], 
              hi[2] * hj[1] + hi[1] * hj[2], 
              hi[2] * hj[2]])
    A.append([(hi[0] * hi[0]) - (hj[0] * hj[0]),
              (hi[0] * hi[1] + hi[1] * hi[0]) - (hj[0] * hj[1] + hj[1] * hj[0]), 
              (hi[1] * hi[1]) - (hj[1] * hj[1]), 
              (hi[2] * hi[0] + hi[0] * hi[2]) - (hj[2] * hj[0] + hj[0] * hj[2]), 
              (hi[2] * hi[1] + hi[1] * hi[2]) - (hj[2] * hj[1] + hj[1] * hj[2]), 
              (hi[2] * hi[2]) - (hj[2] * hj[2])])
    A = np.array(A)
    return A

# B
def find_B(V_list):
    V_list = V_list.reshape(-1, 6)
    _, _, vt = np.linalg.svd(V_list)
    b = vt[-1]
    # B is positive definite
    if b[0] < 0:
        b = -b
    B = np.array([[b[0], b[1], b[3]], 
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]], np.float32)
    return B

# K
def find_K(B_matrix):
    K_inv_t = np.linalg.cholesky(B_matrix)
    K_inv = K_inv_t.T
    K = np.linalg.inv(K_inv)
    K = K / K[-1, -1]
    return K

# extrinsics
def find_extrinsics(K_matrix, h1, h2, h3):
    K_matrix_inv = np.linalg.inv(K_matrix)
    lambda_value = 1 / np.linalg.norm(np.dot(K_matrix_inv, h1))
    r1 = lambda_value * np.dot(K_matrix_inv, h1)
    r2 = lambda_value * np.dot(K_matrix_inv, h2)
    r3 = np.cross(r1, r2)
    t = lambda_value * np.dot(K_matrix_inv, h3)
    return np.array([r1, r2, r3, t]).T



H = np.zeros((image_num, 3, 3), np.float32)
for i in range(image_num):
    H[i] = find_homography(objpoints[i], imgpoints[i]).T
    
V = np.zeros((image_num, 2, 6), np.float32)
for i in range(image_num):
    V[i] = find_V(H[i][0], H[i][1])

B = find_B(V)

K = find_K(B)

extrinsics = np.zeros((image_num, 3, 4), np.float32)
for i in range(image_num):
    extrinsics[i] = find_extrinsics(K, H[i, 0], H[i, 1], H[i, 2])
    
    
    
    
# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K   #mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()


#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""

