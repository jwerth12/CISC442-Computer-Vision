import cv2
import numpy as np
import matplotlib.pyplot as plt

# Part 5
# Shear the original image in the y-direction by 1.4 and perform (1)

flower = cv2.imread("flower_HW4.jpg")
flower = cv2.cvtColor(flower,cv2.COLOR_BGR2RGB)

plt.axis('off')

rows, cols, dim = flower.shape
# create transformation matrix for shearing by 1.4 in y direction
M = np.float32([[1,0,0], 
                [1.4,1,0], 
                [0,0,1]])  
sheared_flower = cv2.warpPerspective(flower,M,(cols, rows))

plt.title("Sheared Flower - Y direction")
plt.imshow(sheared_flower)
plt.show()


# Image with Edges
# using the Canny Method
gray_flower = cv2.cvtColor(sheared_flower,cv2.COLOR_BGR2GRAY) # convert to grayscale
edged_flower = cv2.Canny(gray_flower, threshold1=30, threshold2=100)
plt.imshow(edged_flower,cmap="gray")
plt.axis("off")
plt.title("Edged Sheared Flower - Y-direction")
plt.savefig('q5_shearedy_edges.jpg')
plt.show()


# Image with Corners
# using openCV cornerHarris() function 

sheared_flower = cv2.cvtColor(sheared_flower,cv2.COLOR_BGR2RGB) 

# modify the data type setting to 32-bit floating point before applying function
gray = np.float32(gray_flower)
flower_corners = cv2.cornerHarris(gray, 8, 11, 0.07)

# dilate to mark corners
flower_corners = cv2.dilate(flower_corners, None)

# Thresholding
sheared_flower[flower_corners > 0.01 * flower_corners.max()]=[0, 0, 255]

sheared_flower = cv2.cvtColor(sheared_flower,cv2.COLOR_BGR2RGB)
plt.imshow(sheared_flower)
plt.axis("off")
plt.title("Sheared Flower with Corners")
plt.savefig('q5_shearedy_corners.jpg')
plt.show()