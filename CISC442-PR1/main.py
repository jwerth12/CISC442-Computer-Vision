import cv2
import numpy as np
import pr1_functions as wcv 

# importing declared functions as personal wcv (werth cv)

# NOTE: the initial window takes a second to pop up
"""
######### TESTING PART 1 #############
kernel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

image = cv2.imread("./images/lena.png", cv2.IMREAD_COLOR)

# Testing against built in filter2D function
# Value -1 represents that the resulting image will have same depth as the source image
convolve_test = cv2.filter2D(image, -1, kernel)

# Code without using filter2D
convolved = wcv.convolve(image, kernel)

cv2.imshow('Original Image', image)
cv2.imshow('Convolved Image', convolved)
cv2.imshow('Correct Convolved with filter2D', convolve_test)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('p1_convolved.jpg', convolved)

########## TESTING PART 2 #############
reduced = wcv.reduce(image)     # reduce using the built function
cv2.imwrite('p2_reduced.jpg', reduced)

# compare result with correct results  
blurred = cv2.GaussianBlur(image, (3, 3), 0)
reduced_correct = cv2.resize(
    blurred, (image.shape[1] // 2, image.shape[0] // 2))

cv2.imshow('Original Image', image) # for reference
cv2.imshow('Reduced Image', reduced)
cv2.imshow('Correct Reduced Image', reduced_correct)
cv2.waitKey()
cv2.destroyAllWindows()

########## TESTING PART 3 #############
expanded = wcv.expand(image)
expanded_correct = cv2.pyrUp(image)

cv2.imshow('Original Image', image) # for reference
cv2.imshow('Correct Expanded Image', expanded_correct)
cv2.imshow('Expanded Image', expanded)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('p3_expanded.jpg', expanded)

########## TESTING PART 4 ##############
gaussian = wcv.gaussianPyramid(image, 4) 
cv2.imshow('Guassian Pyramid 1', gaussian[0])
cv2.imshow('Guassian Pyramid 2', gaussian[1])
cv2.imshow('Guassian Pyramid 3', gaussian[2])
cv2.imshow('Guassian Pyramid 4', gaussian[3])

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('p4_gaussian_1.jpg', gaussian[0])
cv2.imwrite('p4_gaussian_2.jpg', gaussian[1])
cv2.imwrite('p4_gaussian_3.jpg', gaussian[2])
cv2.imwrite('p4_gaussian_4.jpg', gaussian[3])


########## TESTING PART 5 ##############
laplacian = wcv.laplacianPyramid(image, 4)
cv2.imshow('Laplacian Pyramid 1', laplacian[0])
cv2.imshow('Laplacian Pyramid 2', laplacian[1])
cv2.imshow('Laplacian Pyramid 3', laplacian[2])
cv2.imshow('Laplacian Pyramid 4', laplacian[3])

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('p5_laplacian_1.jpg', laplacian[0])
cv2.imwrite('p5_laplacian_2.jpg', laplacian[1])
cv2.imwrite('p5_laplacian_3.jpg', laplacian[2])
cv2.imwrite('p5_laplacian_4.jpg', laplacian[3])

########## TESTING PART 6 ##############
reconstructed = wcv.reconstruct(laplacian,4)
cv2.imshow('Reconstructed Image', reconstructed)
cv2.imshow('Original Image', image)

# report the error using image difference 
error = wcv.getError(reconstructed, image)
print("MSE = ", error)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('p6_reconstructed.jpg', reconstructed)
"""
########## TESTING PART 7 ##############
left_1 = cv2.imread("./images/Test_A1.png", cv2.IMREAD_COLOR)
right_1 = cv2.imread("./images/Test_A2.png", cv2.IMREAD_COLOR)

# NOTE: You MUST input a coordinate for each image as a blend boundary 
blend1 = wcv.blend(left_1, right_1, 6)
cv2.imwrite('p7_blend1.jpg', blend1)

left_2 = cv2.imread("./images/Test_B1.png", cv2.IMREAD_COLOR)
right_2 = cv2.imread("./images/Test_B2.png", cv2.IMREAD_COLOR)

blend2 = wcv.blend(left_2, right_2, 6)
cv2.imwrite('p7_blend2.jpg', blend2)

left_3 = cv2.imread("./images/Test_C1.png", cv2.IMREAD_COLOR)
right_3 = cv2.imread("./images/Test_C2.png", cv2.IMREAD_COLOR)

blend3 = wcv.blend(left_3, right_3, 6)
cv2.imwrite('p7_blend3.jpg', blend3)

left_4 = cv2.imread("./images/Test_D1.png", cv2.IMREAD_COLOR)
right_4 = cv2.imread("./images/Test_D2.png", cv2.IMREAD_COLOR)

blend4 = wcv.blend(left_4, right_4, 6)
cv2.imwrite('p7_blend4.jpg', blend4)