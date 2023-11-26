import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.
# Convolution is taking the dot product 

def convolve(I, H):
    # assertion checks
    assert I is not None, "Image could not be read, check with os.path.exists()"
    assert H is not None, "Invalid kernel, could not be read" 


    # Determine dimensions of image and inputted kernel
    image_height, image_width, num_channels = I.shape
    kernel_height, kernel_width = H.shape

    # Calculate the dimensions of the output
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize an empty output with all zeros
    output = np.zeros((output_height, output_width, num_channels), dtype=np.uint8)

    # Perform convolution for each color channel (since we know this is in RGB)
    for c in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):

                # Extract the region of interest from the input image channel
                # trying to preserve computational efficiency, increase accuracy
                roi = I[i:i + kernel_height, j:j + kernel_width, c]
                
                # Sum and multiply the region of interest with kernel
                output_pixel = np.sum(roi * H)
                
                #output[i,j,c] = output_pixel
                # Ensure the output value is in range (0-255)
                output[i, j, c] = np.clip(output_pixel, 0, 255).astype(np.uint8)

    return output



#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it; 
# use separable 1D Gaussian kernels.

def reduce(I):
    # assertion checks
    assert I is not None, "Image could not be read, check with os.path.exists()"

    # Height, width, RGB channel
    height, width, channels = I.shape

    # Gaussian blurring using opencv -- does this count as using separable 1D kernels?
    blurred = cv2.GaussianBlur(I, (3, 3), 0)

    # Downsample -- interpolation algorithm
    # Resize the filtered image to half its width and height
    reduced = cv2.resize(blurred, (width // 2, height // 2))

    return reduced

#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded, 
# twice the width and height of the input.

def expand(I):
    # assertion checks
    assert I is not None, "Image could not be read, check with os.path.exists()"

    # Height, width, RGB channel
    height, width, channels = I.shape

    # Resize the image to twice the width and height of input
    expanded = cv2.resize(I, (width * 2, height * 2))

    return expanded

#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.

def gaussianPyramid(I, n):

    # assertion checks
    assert I is not None, "Image could not be read, check with os.path.exists()"
    assert n, "Must input a positive integer for the number of layers in the pyramid"

    G = I.copy()
    layers = [G]     # array to hold the layers of the pyramid 
    curr = G

    for i in range(n):
        curr = reduce(curr)
        layers.append(curr)

    return layers   # the layers form the pyramid

#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.

def laplacianPyramid(I, n):

    # assertion checks
    assert I is not None, "Image could not be read, check with os.path.exists()"
    assert n, "Must input a positive integer for the number of layers in the pyramid"


    gaussian = gaussianPyramid(I, n)    # first create the Gaussian pyramid 
    laplacian = [gaussian[n-1]]         # first layer in laplacian = last in gaussian 

    # construct pyramid by taking difference of gaussian levels 
    for i in range(n - 1, 0, -1):
        next = expand(gaussian[i])
        
        # make sure we have same size images for operations 
        # NOTE: this will only be a problem if our original image has odd dimensions
        height, width, channels = gaussian[i-1].shape
        next = cv2.resize(next, (width,height))
        diff = cv2.subtract(gaussian[i-1], next)

        laplacian.append(diff)

    laplacian.append(gaussian[n - 1])   # tip of pyramid 

    return laplacian 

#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels 
# to generate the original image. Report the error in reconstruction using image difference.

def reconstruct(LI, n):
    # assertion checks
    assert LI, "Laplacian Pyramid could not be read, check with os.path.exists() or make sure not an empty list"
    assert n, "Must input a positive integer for the number of layers in the pyramid"


    # reverse the pyramid so that we are starting at the tip and set that as our starting image
    # pyramid = list(reversed(LI))
    reconstructed = LI[0] 

    for i in range(1,n):
        upsampled = expand(reconstructed)       # upsample the reconstructed image
        reconstructed = cv2.add(upsampled, LI[i])  # add the upsampled image to our current layer 

    return reconstructed

#################################################################

# PART 7: Mosaicking 

# blendEven 
#   evenly blends two images together down the middle 
# Parameters:
#   left, right: images
#   n: the number of layers we want int our pyramid
#   NOTE: this is just to understand blending, blending with specified coordinates can be found below
def blendEven(left, right, n):

    # assertion checks
    assert left is not None, "Left image could not be read, check with os.path.exists()"
    assert right is not None, "Right image could not be read, check with os.path.exists()"
    assert n, "Must input a positive integer for the number of layers in the pyramid"

    resized_left = left.copy()
    height, width, channels = resized_left.shape
    right = cv2.resize(right, (width, height))

    lp_left = laplacianPyramid(resized_left,n)
    lp_right = laplacianPyramid(right,n)

    # Now add left and right halves of images in each level
    Layers = []
    for la,lb in zip(lp_left,lp_right):
        rows,cols,dpt = la.shape
        layer = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        Layers.append(layer)
    
    # now reconstruct
    curr = Layers[0]
    for i in range(1,6):
        height, width, channels = Layers[i].shape
        curr = cv2.resize(curr, (width, height))
        curr = cv2.add(curr, Layers[i])
    
    cv2.imshow('Pyramid_blending',curr)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return curr

# getBoundaries(left, right)
#   Helper function to get the user clicked coordinates on the two images 
#   These coordinates are the blend boundaries -- Blend boundaries are where the two images meet 
# Parameters: 
#   left: left image
#   right: right image
# User Instruction:
#   The function will take ONE coordinate from each photo as the boundary 
# Returns:
#   Returns the coordinates of the blend boundaries for each image
def getBoundaries(image1, image2):

    # assertion checks
    assert image1 is not None, "First image could not be read, check with os.path.exists()"
    assert image2 is not None, "Second image could not be read, check with os.path.exists()"
  
    boundaries = []             # store the coordinates, there will be a set for each image

    images = [image1.copy(), image2.copy()]

    print("Please mark the boundary for the each image")

    for image in images:
        cv2.namedWindow('Image Viewer')
        
        # Create a list to store the points
        # coordinates = []

        # function to handle mouse events 
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                boundaries.append((x,y))
                #coordinates.append((x, y))
                cv2.circle(image,(x,y),10,(255,0,0),-1)
                cv2.imshow('Image Viewer', image)
                print(f"Boundary chosen at ({x}, {y}), press any key to continue")
        
        # Set the mouse callback
        cv2.setMouseCallback('Image Viewer', mouse_callback)

        # Find boundary coordinates for each image
        cv2.imshow('Image Viewer', image)
        # Wait for a key 
        cv2.waitKey(0)
        # Destroy the window
        cv2.destroyWindow('Image Viewer')

    return boundaries    # return the coordinates for the bitmask boundaries 

# getDimensions
#   helper function to get the dimensions for the blended image based on boundaries 
#   we need to have same size images to perform blending 
#   returns the images with padding so they are positioned accordingly for blending
def getDimensions(left, right, boundaries):

    # assertion checks
    assert left is not None, "Left image could not be read, check with os.path.exists()"
    assert right is not None, "Right image could not be read, check with os.path.exists()"
    assert boundaries, "No boundaries were given"

    height1, width1, channels1 = left.shape
    height2, width2, channels2 = right.shape

    scaler = height1 / height2

    right = cv2.resize(right, (width2, int(scaler*height2))) 

    # Determine both image dimensions
    left_height, left_width, l_channels = left.shape
    right_height, right_width, r_channels = right.shape

    # Determine how much of image is offset for blending (sub x coord from width)

    # Determine what portion of each photo will be cut out from the blend boundaries  
    overlap1 = left_width - boundaries[0][0]
    overlap2 = boundaries[1][0]
    
    # width of final image based on overlap 
    output_width = left_width + right_width - overlap1 - overlap2

    pad_width1 = max(0, output_width - left_width)
    pad_width2 = max(0, output_width - right_width)

    # pad the images to resize so that we can blend them together
    resized_left = cv2.copyMakeBorder(left, 0, 0, 0, pad_width1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    resized_right = cv2.copyMakeBorder(right, 0, 0, pad_width2, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    cv2.imshow('Resized Left', resized_left)
    cv2.imshow('Resized Right', resized_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized_left, resized_right

# Blend function
#   Blends two images based on given blend boundaries inputted by the user
# Parameters:
#   Left and Right: the two images to be blended
#   n: the number of layers you want to use in laplacian pyramid
# Returns:
#   Blended image of the left and right image using laplacian pyramids 
def blend(left, right, n):

    # Check for value errors
    assert left is not None, "Left image could not be read, check with os.path.exists()"
    assert right is not None, "Right image could not be read, check with os.path.exists()"
    
    # call helper functions to get boundaries and resize for blended image 
    boundaries = getBoundaries(left, right)
    resized_left, resized_right = getDimensions(left, right, boundaries)

    lp_left = laplacianPyramid(resized_left,n)
    lp_right = laplacianPyramid(resized_right,n)

    # Now add left and right halves of images in each level
    Layers = []
    for l,r in zip(lp_left,lp_right):
        rows,cols,dpt = l.shape
        curr = np.hstack((l[:,0:cols//2], r[:,cols//2:]))
        Layers.append(curr)
    
    # now reconstruct
    reconstructed = Layers[0]
    for i in range(1,n):
        height, width, channels = Layers[i].shape
        reconstructed = cv2.resize(reconstructed, (width, height))
        reconstructed = cv2.add(reconstructed, Layers[i])
    
    cv2.imshow('Pyramid_blending',reconstructed)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return reconstructed


# getError
#   Helper function to calculate the mean squared error between two images 
# Parameters:
#       Takes in two images, image1 and image2
# Returns:
#       The mean squared error for the differences in the images
def getError(image1,image2):

    # Convert the images to grayscale.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Separating out calculations for MSE 
    diff = gray1 - gray2
    squared = diff ** 2
    mse = np.mean(squared)

    return mse