#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    slopes = {"left": {"avg" : 0, "all" : []}, "right" : {"avg" : 0, "all" : []}}
    intercepts = {"left": {"avg" : 0, "all" : []}, "right" : {"avg" : 0, "all" : []}}
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2-y1)/(x2-x1)
            b = y1 - m*x1
            
            if 0.2 < m < 0.8:
                slopes["left"]["all"].append(m)
                intercepts["left"]["all"].append(b)
            elif -0.8 < m < -0.2:
                slopes["right"]["all"].append(m)
                intercepts["right"]["all"].append(b)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    slopes["left"]["avg"] = np.mean(slopes["left"]["all"])
    slopes["right"]["avg"] = np.mean(slopes["right"]["all"])
    
    intercepts["left"]["avg"] = np.mean(intercepts["left"]["all"])
    intercepts["right"]["avg"] = np.mean(intercepts["right"]["all"])
    
    # y = mx + b, x = y-b / m
    min_y = 315
    max_y = img.shape[0]
    
    l_x1 = int((max_y - intercepts["left"]["avg"]) / slopes["left"]["avg"])
    l_y1 = int(max_y)
    l_x2 = int((min_y - intercepts["left"]["avg"]) / slopes["left"]["avg"])
    l_y2 = int(min_y)
    
    r_x1 = int((max_y - intercepts["right"]["avg"]) / slopes["right"]["avg"])
    r_y1 = int(max_y)
    r_x2 = int((min_y - intercepts["right"]["avg"]) / slopes["right"]["avg"])
    r_y2 = int(min_y)
    
    print((l_x1, l_y1), (l_x2, l_y2), slopes["left"]["avg"], slopes["right"]["avg"])

    
    cv2.line(img, (l_x1, l_y1), (l_x2, l_y2), [0,0,255], thickness)
    cv2.line(img, (r_x1, r_y1), (r_x2, r_y2), [0,0,255], thickness)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

import os

os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
test_images = os.listdir("test_images/")
filename = test_images[4]
image = mpimg.imread("test_images/" + filename)

def image_pipeline(image):
    gray = grayscale(image)
    imshape = gray.shape
    vertices = np.array([[(0,imshape[0]),(490, 315), (490, 315), (imshape[1],imshape[0])]], dtype=np.int32)
    gauss = gaussian_blur(gray, 3)
    cannyi = canny(gauss,50,125)
    rho = 1 
    theta = np.pi/180
    threshold = 5
    min_line_len = 4
    max_line_gap = 10
    hough = hough_lines(cannyi,rho, theta, threshold, min_line_len, max_line_gap)
    roi = region_of_interest(hough, vertices)
    wi = weighted_img(roi, image)
    mpimg.imsave(filename, wi)
    return wi

plt.imshow(image_pipeline(image))

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)
    imshape = gray.shape
    vertices = np.array([[(0,imshape[0]),(490, 315), (490, 315), (imshape[1],imshape[0])]], dtype=np.int32)
    gauss = gaussian_blur(gray, 3)
    cannyi = canny(gauss,50,125)
    rho = 1 
    theta = np.pi/180
    threshold = 5
    min_line_len = 4
    max_line_gap = 10
    hough = hough_lines(cannyi,rho, theta, threshold, min_line_len, max_line_gap)
    roi = region_of_interest(hough, vertices)
    wi = weighted_img(roi, image)
    return wi

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,3)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
