# Jenn Werth
# Homework 4

import cv2
import numpy as np
import matplotlib.pyplot as plt

# import the flower image 
flower = cv2.imread("flower_HW4.jpg")
gray_flower = cv2.cvtColor(flower,cv2.COLOR_BGR2GRAY) # convert to grayscale
plt.imshow(gray_flower, cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")
plt.show()


# Part 1 
# Run an in-built edge detector and a corner detector
# produce two output images: 
#   (i) image with edges
#   ii) image with corners

# Image with Edges
# using the Canny Method
edged_flower = cv2.Canny(gray_flower, threshold1=30, threshold2=100)
plt.imshow(edged_flower,cmap="gray")
plt.axis("off")
plt.title("Original Edged Flower Image (Canny Edge Detection)")
plt.savefig('q1_edges.jpg')
plt.show()

# Image with Corners
# using openCV cornerHarris() function 
flower = cv2.cvtColor(flower,cv2.COLOR_BGR2RGB)

# modify the data type setting to 32-bit floating point before applying function
gray = np.float32(gray_flower)
flower_corners = cv2.cornerHarris(gray, 4, 11, 0.07)

# dilate to mark corners
flower_corners = cv2.dilate(flower_corners, None)

# Thresholding
flower[flower_corners > 0.01 * flower_corners.max()]=[0, 0, 255]
plt.imshow(flower)
plt.axis("off")
plt.title("Original Flower with Corners")
plt.savefig('q1_corners.jpg')
plt.show()


