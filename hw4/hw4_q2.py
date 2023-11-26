import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

# Part 2
# Rotate the original image by 60-degrees and perform (1)

# read the flower image in again (we modified it with the corners)
flower = cv2.imread("flower_HW4.jpg")
flower = cv2.cvtColor(flower,cv2.COLOR_BGR2RGB)

flower_rotated = imutils.rotate(flower, angle=60)

plt.imshow(flower_rotated)
plt.axis("off")
plt.title("Rotated Flower")
plt.show()

# Performing edge detection / corners again 
gray_rotated = cv2.cvtColor(flower_rotated,cv2.COLOR_BGR2GRAY) # convert to grayscale
edged_rotated = cv2.Canny(gray_rotated, threshold1=30, threshold2=100)
plt.imshow(edged_rotated,cmap="gray")
plt.axis("off")
plt.title("Rotated Edged Flower Image (Canny Edge Detection)")
plt.savefig('q2_rotated_edges.jpg')
plt.show()

# modify the data type setting to 32-bit floating point before applying function
gray = np.float32(gray_rotated)
rotated_corners = cv2.cornerHarris(gray, 6, 11, 0.07)

# dilate to mark corners
rotated_corners = cv2.dilate(rotated_corners, None)

# Thresholding
flower_rotated[rotated_corners > 0.01 * rotated_corners.max()]=[0, 0, 255]

plt.imshow(flower_rotated)
plt.axis("off")
plt.title("Rotated Flower with Corners")
plt.savefig('q2_rotated_corners.jpg')
plt.show()