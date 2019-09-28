import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('../road_images_video/stopsign.jpg')

plt.imshow(img)
plt.plot(850,320,'.') # top right
plt.plot(865,450,'.') # bottom right
plt.plot(533,350,'.') # bottom left 
plt.plot(535,210,'.') # top left

def warp(img):
	img_size = (img.shape[1], img.shape[0])
	src = np.float32(
		[[850,320],
		[865,450],
		[533,350],
		[535,210]])
	dst = np.float32(
		[[870,240],
		[870,370],
		[520,370],
		[520,240]])

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

	return warped

warped_im = warped(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Source Image')
ax1.imshow(img)
ax2.set_title('Warped Image')
ax2.imshow(warped_im)