import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

# Part 3
# Scale the image by 1.6 in both the x and y-directions and perform (1)

flower = cv2.imread('flower_HW4.jpg')
flower = cv2.cvtColor(flower,cv2.COLOR_BGR2RGB) 

plt.axis('off')
plt.imshow(flower)
plt.show()
#cv2.imshow("Original", flower)
#cv2.waitKey(0)  # press any key to delete the pop up window

rows, cols, dim = flower.shape

# scaling matrix 
M = np.float32([[1.6, 0  , 0],
            	[0,   1.6, 0],
            	[0,   0,   1]])

# NOTE: the scaling makes the image not fit in the window
scaled_flower = cv2.warpPerspective(flower,M,(cols,rows))

plt.axis('off')
plt.title("Scaled Flower")

plt.imshow(scaled_flower)
plt.show()


# edge detection / corner detection 
# Image with Edges
# using the Canny Method
gray_flower = cv2.cvtColor(scaled_flower,cv2.COLOR_BGR2GRAY) # convert to grayscale

edged_flower = cv2.Canny(gray_flower, threshold1=30, threshold2=100)
plt.imshow(edged_flower,cmap="gray")
plt.axis("off")
plt.title("Edged Scaled Flower Image (Canny Edge Detection)")
plt.savefig('q3_scaled_edges.jpg')
plt.show()

# Image with Corners
# using openCV cornerHarris() function 

scaled_flower = cv2.cvtColor(scaled_flower,cv2.COLOR_BGR2RGB) 

# modify the data type setting to 32-bit floating point before applying function
gray = np.float32(gray_flower)
flower_corners = cv2.cornerHarris(gray, 8, 11, 0.07)

# dilate to mark corners
flower_corners = cv2.dilate(flower_corners, None)

# Thresholding
scaled_flower[flower_corners > 0.01 * flower_corners.max()]=[0, 0, 255]
scaled_flower = cv2.cvtColor(scaled_flower,cv2.COLOR_BGR2RGB)
plt.imshow(scaled_flower)
plt.axis("off")
plt.title("Scaled Flower with Corners")
plt.savefig('q3_scaled_corners.jpg')
plt.show()