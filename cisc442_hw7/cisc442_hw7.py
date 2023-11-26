# Jenn Werth
# homework 7

import cv2
import numpy as np
import matplotlib.pyplot as plt

################# Part 1 #################
# Generate and show four levels of multi-resolution. 
# Use a Gaussian kernel of your choice. 

einstein = cv2.imread("./images/Einstein.jpg")

# Function to create multi-resolution images at 4 levels
#   NOTE: this is a Gaussian pyramid
# returns the layers of the formed pyramid 
def mres(input):
    gaussian = []  # set the base as the original image 
    curr = input

    for i in range(0,4):
        # show the image 
        cv2.imshow('Multi Resolution Level', curr)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # save it for submission 
        cv2.imwrite(f"./images/Q1_MultiRes_Level{i+1}.jpg", curr)

        # iterate through 
        gaussian.append(curr)
        curr = cv2.pyrDown(curr)    # this uses a 5x5 gaussian kernel
    
    return gaussian

# Use the above function to produce the images 
multi_res = mres(einstein)


################## Part 2 #################
# Function to generate and show four levels of multi-scale. 
# Use the same Gaussian kernel as above.
def mscale(input):
    scales = []  # add original to our list
    curr = input

    for i in range(0,4):
        # show the image 
        cv2.imshow('Multi Scale Level', curr)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # save it for submission
        cv2.imwrite(f"./images/Q2_MultiScale_Level{i+1}.jpg", curr)

        # iterate
        scales.append(curr)
        curr = cv2.GaussianBlur(curr, (5, 5), 0)

    return scales   # return all different scales of image

# use the function to generate images 
multi_scale = mscale(einstein)


################## Part 3 #################
# Generate Laplacian planes using a Laplacian kernel of your choice

# Function to create pyramid 
#   inpute: input image, n: number of layers
# NOTE: for compatibility purposes this will generate 4 planes
def laplacian(input):

    # first create a gaussian pyramid 
    gaussian = mres(input)              # note this calls mres so it will pop up here 

    laplacian = []
    for i in range(0, 3):
        curr = cv2.Laplacian(gaussian[i], ddepth=-1)
        laplacian.append(curr)

        # show the image
        cv2.imshow('Laplacian Level', curr)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
        # save it for submission
        cv2.imwrite(f"./images/Q3_Laplacian_Plane{i+1}.jpg", curr)
    
    laplacian.append(gaussian[3])   # we take the tip of the guassian pyramid as tip of laplacian
    cv2.imwrite(f"./images/Q3_Laplacian_Plane4.jpg", laplacian[3])
    
    return laplacian

laplacian_planes = laplacian(einstein)


############ Part 4 #################
# Generate an approximation to Laplacian using the difference of Gaussian planes from (1). 
#   This uses multi resolution
# NOTE: you need to do 'Expand' on images before taking the difference.
# Function to produce the pyramid 
def approx_lap_1(input):
    gaussian = mres(input)  # note this calls mres so images will pop up 
    approx = []

    for i in range(0,3):
        # get dimensions for the current layer 
        height, width, channels = gaussian[i].shape

        # resize the layer up 
        resized = cv2.resize(gaussian[i+1], (width, height))

        # get the difference of guassian planes from 1
        curr = cv2.subtract(gaussian[i], resized)
        approx.append(curr)

        # show the image
        cv2.imshow('Laplacian Approximated Level - muit res', curr)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # save for submission 
        cv2.imwrite(f"./images/Q4_Laplacian_Approx{i+1}.jpg", curr)
    
    # remember to add final level 
    approx.append(gaussian[3])
    cv2.imwrite(f"./images/Q4_Laplacian_Approx4.jpg", approx[3])
    return approx

# run function to produce images
approx = approx_lap_1(einstein)

############ Part 5 #################
# Generate an approximation to Laplacian using the difference of Gaussian planes from (2)
#   This uses multi-scale
# Function to produce the pyramid 
def approx_lap_2(input):
    gaussian = mscale(input)    # using multi scale this time, note the images will pop up again from call
    approx = []

    for i in range(0,3):

        # get the difference of guassian planes from 2
        curr = cv2.subtract(gaussian[i], gaussian[i+1])
        approx.append(curr)

        # show the image
        cv2.imshow('Laplacian Approximated Level - multi scale', curr)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # save for submission 
        cv2.imwrite(f"./images/Q5_Laplacian_Approx{i+1}.jpg", curr)
    
    # remember to add final level 
    approx.append(gaussian[3])
    cv2.imwrite(f"./images/Q5_Laplacian_Approx4.jpg", approx[3])
    return approx

# run function to produce images 
approx2 = approx_lap_2(einstein)
