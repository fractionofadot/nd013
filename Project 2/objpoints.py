# objpoints_and_imgpoints
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('')
plt.imshow(img)

images = glob.glob('../calibration/calibration*.jpg')

objpoints = []
imgpoints = []

nx,ny = (8,6)

# initialize with zeros, three columns (x,y,z)
# (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((ny*nx,3), np.float32)

# Use ngrid to generate coordinates
# reshape back into two columns
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for fname in images:
	img = mpimg.imread(fname)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	# If found, draw corners
	if ret == True:
		imtpoints.append(corners)
		objpoints.append(objp)

		img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
		plt.imshow(img)