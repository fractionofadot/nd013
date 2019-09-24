# **Finding Lane Lines on the Road** 

## Robert Parker

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on my work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of several steps. First, I converted the images to grayscale, then I used a gaussian blur to further reduce any potential noise before applying Canny edge detection. I then created a polygonal/triangular region of interest to mask out the rest of the image and focus on the lane. 

Next, I applied the Hough Transform to convert all of the individual points of the outlined lanes into multiple lines.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function using numpy polyfit to help calculate and find the average slope and intercept for each lane and extrapolate the line. To determine which points belonged to which lane, I set a boundary where `0.4 < m < 0.7` (a positive slope) belonged to the right line, and `-0.7 < m < -0.4` (a negative slope) went to the left line. 

I then returned a weighted image with the drawn lines over the original.

![solidYellowLeft.jpg](solidYellowLeft.jpg)

### 2. Identify potential shortcomings with your current pipeline


Potential shortcomings would be:
- what would happen when a car crosses the lane? 
- what about when it rains, or if there is low visibility?
- what if the contrast between the line and the road is very low?
- what if there is a faded line in the middle of the lane?

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to add a curved line extrapolation. It seems numpy can help with that as well. 

It may be worth adding some color distinction. We assume that lanes must be yellow or white, but that is not always the case. It may also be reasonable to assume the road is gray or black (or, at least not a color), even if it is not always so. So that may help isolate the lane markings even further. If I just tried to eliminate grayscale colors, where all three bands are relatively even. 
