# Examples of Useful Code

# You need to pass a single color channel to the cv2.Sobel() function, 
# so first convert it to grayscale:
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# Note: Make sure you use the correct grayscale conversion depending on 
# how you've read in your images. Use cv2.COLOR_RGB2GRAY if you've read 
# in an image using mpimg.imread(). Use cv2.COLOR_BGR2GRAY if you've 
# read in an image using cv2.imread().

# Calculate the derivative in the xx direction 
# (the 1, 0 at the end denotes xx direction):
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

# Calculate the derivative in the yy direction 
# (the 0, 1 at the end denotes yy direction):
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# Calculate the absolute value of the xx derivative:
abs_sobelx = np.absolute(sobelx)

# Convert the absolute value image to 8-bit:
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
# Note: It's not entirely necessary to convert to 8-bit (range from 0 to 255) 
# but in practice, it can be useful in the event that you've written 
# a function to apply a particular threshold, and you want it to work 
# the same on input images of different scales, like jpg vs. png. 
# You could just as well choose a different standard range of values, 
# like 0 to 1 etc.

# Create a binary threshold to select pixels based on gradient strength:

thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')