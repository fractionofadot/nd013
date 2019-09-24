gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Note: If you are reading in an image using mpimg.imread() 
# this will read in an RGB image and you should convert to grayscale 
# using cv2.COLOR_RGB2GRAY, but if you are using cv2.imread() or 
# the glob API, as happens in this video example, this will read in 
# a BGR image and you should convert to grayscale using cv2.COLOR_BGR2GRAY. 
# We'll learn more about color conversions later on in this lesson, 
# but please keep this in mind as you write your own code and look at code examples.

# Finding chessboard corners (for an 8x6 board):
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

# Drawing detected corners on an image:
img = cv2.drawChessboardCorners(img, (8,6), corners, ret)

# Camera calibration, given object points, image points, and the shape of the grayscale image:
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistorting a test image:
dst = cv2.undistort(img, mtx, dist, None, mtx)