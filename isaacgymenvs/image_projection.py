import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #not used
import mplot3d_dragger # not used
from mpl_toolkits.mplot3d import Axes3D # This bitch might not work. Uninstall matplotlib, install --||--, and upgrade --||-- should work
from skimage import transform
from skimage.io import imread, imshow

###############################################################
# Image transformation

x_rot = 119 #placed in 61 degrees than 29+90 =119
y_rot = 0 
z_rot = 0

omega = math.radians(x_rot)
theta = math.radians(y_rot)
kappa = math.radians(z_rot)

tx = 0 # unsure of the unit
ty = 150 # mm
tz = 100 # mm

# load point cloud from exomy.py
img = imread('/home/gymuser/isaac_rover_privatRep/envs/tasks/color_cam.png')

#img = np.array([ [-1, -1, 1, 0], [-1, -1, 3, 0], [-1, 1, 1, 0], [-1, 1, 3, 0], [1, -1, 1, 0], [1, -1, 3, 0], [1, 1, 1, 0], [1, 1, 3, 0] ]) # test

###############################################################
# Rotation and translation matrices

rotMat_x = np.array([[1, 0, 0, 0],
                    [0, math.cos(omega), math.sin(omega), 0],
                    [0, -(math.sin(omega)), math.cos(omega), 0],
                    [0, 0, 0, 1]])
rotMat_y = np.array([ [math.cos(theta), 0, -(math.sin(theta)), 0],
                    [0, 1, 0, 0],
                    [math.sin(theta), 0, math.cos(theta), 0],
                    [0, 0, 0, 1]])
rotMat_z = np.array([[math.cos(kappa), math.sin(kappa), 0, 0],
                    [-(math.sin(kappa)), math.cos(kappa), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

trans_mat = np.array([  [1, 0, 0, tx],
                        [0, 1, 0, ty],
                        [0, 0, 1, tz],
                        [0, 0, 0, 1]])

###############################################################
# Compute homography matrix and apply to input
projection_mat = np.matmul(np.matmul(np.matmul(rotMat_x, rotMat_y), rotMat_z), trans_mat)
proj_trans = transform.AffineTransform(matrix=projection_mat, dimensionality=3)
tf_img = img.dot(proj_trans)  

###############################################################
# Plot camera view and top/transformed view
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.scatter(tf_img[:, 0], tf_img[:,1], tf_img[:,2])
ax1.set_title('Birds view')
ax1.set_xlabel("x axis")
ax1.set_ylabel("y axis")
ax1.set_zlabel("z axis")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(img[:, 0], img[:,1], img[:,2])
ax.set_title('Camera Plane')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

plt.show() 