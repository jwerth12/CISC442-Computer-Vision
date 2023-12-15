# Jenn Werth
# program for a stereo analysis system involving
#   1) Region-based
#   2) Feature-based matching 

import numpy as np
import cv2

# 1a) Region Based Analysis Functions
#       Creates Matching Scores for
#           1. Sum of absolute differences 
#           2. Sum of squared differences 
#           3. Normalized cross correlation 


# Sum of Absolute Differences
#   Parameters: template, window 
#       takes in a template and a current window to compare
#   Returns the sum of absolute differences matching score for that position 
def sad(template, window):
    score = np.sum(np.abs(template - window))
    return score

# Sum of Squared Differences
#   Parameters: template, window
#       takes in a template and a current window to compare
#   Returns the sum of squared differences matching score for that position 
def ssd(template, window):
    score = np.sum((template - window)**2)
    return score

# Normalized Cross Correlation 
#   Parameters: template, window
#       takes in a template and a current window to compare
#   Returns the normalized cross correlation matching score for that position 
def ncc(template, window):

    # check denominator so that we do not divide by 0 (since we are dealing with small numbers may get a warning)
    denominator = np.sqrt(np.sum(template**2)) * np.sqrt(np.sum(window**2))
    if np.isclose(denominator, 0.0):
        score = 0.0  # Assign a default value or handle this case appropriately
   
    # otherwise proceed normally 
    else:
        score = np.sum(template * window) / denominator

    # score = np.sum(template * window) / (np.sqrt(np.sum(template**2)) * np.sqrt(np.sum(window**2)))
    return score

# Function to perform actual matching with these helper functions - based on given template 

# Parameters:
    # - left_image: The left stereo image.
    # - right_image: The right stereo image.
    # - DISTANCE: The distance metric used for matching (e.g., 'ssd', 'sad', 'ncc').
    # - SEARCH_RANGE: The maximum allowable horizontal search range for disparity.
    # - TEMPLATE_SIZE_X: The width of the template (region) used for matching.
    # - TEMPLATE_SIZE_Y: The height of the template (region) used for matching.
    # - disparity: ???

    # Returns:
    # - disparity_map: the resulting map with optimized scores (disparity represents the pixel difference between images)

def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
   
    # Convert to grayscale for matching
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions for template / image
    height, width = left_gray.shape

    half_template_x = TEMPLATE_SIZE_X // 2
    half_template_y = TEMPLATE_SIZE_Y // 2

    # initiate empty disparity map 
    disparity_map = np.zeros_like(left_gray, dtype=np.float32)

    # Iterate over the image to generate the template
    for y in range(half_template_y, height - half_template_y):
        for x in range(half_template_x, width - half_template_x):

            # create template based on dimensions 
            template = left_gray[y - half_template_y:y + half_template_y + 1,
                                 x - half_template_x:x + half_template_x + 1]


            # store our best desparity / minimum score for each 
            min_score = 10000    # set high to ensure we will get a smaller number
            best_disparity = 0

            # Define search range limits based on input
            search_start = max(0, x - SEARCH_RANGE)
            search_end = min(width - 1, x + SEARCH_RANGE)

            # Find best match in the right image based on template
            for d in range(search_start, search_end + 1):

                # generate window to compare
                window = right_gray[y - half_template_y:y + half_template_y + 1,
                                    d - half_template_x:d + half_template_x + 1]

                # check that window has same shape as template for computation
                if window.shape == template.shape:

                    # compute score and compare to our minimum score already found
                    score = DISTANCE(template, window)

                    if score < min_score:
                        min_score = score
                        best_disparity = abs(x - d)

            # keep best disparity for our disparity map
            disparity_map[y, x] = best_disparity

    # return our map 
    return disparity_map


# 1b) Feature-based analysis 
#   Use Harris corners detector for feature extraction, 
#   Then use the descriptor value for matching (i.e., Harris corner response measure)
#   Give the user the option to choose different matching scores: SAD, SSD, NCC
#       NOTE: this is done in main 

# feature_based function
#   all same parameters as region based above
# Returns disparity map with best scores 
def feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
	
    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Use built in Harris corner detector to find keypoints
    #   Using values from OpenCV documentation 
    #   size of neighborhood = 2, sobel kernel size = 3, harris detector free parameter = 0.04 
    corners_left = cv2.cornerHarris(left_gray, 2, 3, 0.04) 
    corners_right = cv2.cornerHarris(right_gray, 2, 3, 0.04)
    
    # extract key points for iteration
    keypoints_left = np.argwhere(corners_left > 0.001 * corners_right.max())
    keypoints_right = np.argwhere(corners_right > 0.001 * corners_right.max())

    # Create empty disparity map to hold scores
    disparity_map = np.zeros_like(left_gray, dtype=np.float32)

    # Iterate through image to get scores 
    for left_pt in keypoints_left:
        x_left, y_left = left_pt[1], left_pt[0]

        # create the template
        template = left_gray[y_left - TEMPLATE_SIZE_Y//2:y_left + TEMPLATE_SIZE_Y//2 + 1,
                             x_left - TEMPLATE_SIZE_X//2:x_left + TEMPLATE_SIZE_X//2 + 1]


        # track the best score we get for each point 
        min_score = 10000
        best_disparity = 0

        # Define search limits from input 
        search_start = max(0, x_left - SEARCH_RANGE)
        search_end = min(right_gray.shape[1] - 1, x_left + SEARCH_RANGE)

        # Search for the best between matching window and the right image 
        for d in range(search_start, search_end + 1):

            # define our window 
            window = right_gray[y_left - TEMPLATE_SIZE_Y//2:y_left + TEMPLATE_SIZE_Y//2 + 1,
                                d - TEMPLATE_SIZE_X//2:d + TEMPLATE_SIZE_X//2 + 1]

            # check shape of window for valid calculations 
            if window.shape == template.shape:

                # Calculate Sum of Squared Differences (SSD)
                score = DISTANCE(template, window)

                if score < min_score:
                    min_score = score
                    best_disparity = abs(x_left - d)

        # Assign the best disparity to the disparity map
        disparity_map[y_left, x_left] = best_disparity

    return disparity_map


# 2) Validity Checks
# If the left-to-right match does not correspond to right-to-left match,
#   a 'zero'(gap) is placed at that location in the disparity. 
# Implementation: switch the left image with right image and see if the correspondences match
#   If correspondences are same in both directions, 
#   then consider that the correspondence are valid, otherwise invalid.


# valid_feature_based: performs validity check and modifies the disparity map based on matches
    # Parameters same as above but note
    # - disparity: The original disparity map 
def valid_feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
    
    # Perform feature-based stereo matching in both directions (right-to-left and left-to-right)
    disparity_left_to_right = feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity)
    disparity_right_to_left = feature_based(right_image, left_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity)

    cv2.imshow('Original', disparity)
    cv2.imshow('Left-to-Right', disparity_left_to_right)
    cv2.imshow('Right-to-Left', disparity_right_to_left)

    for i in range(disparity.shape[0]): 
        for j in range(disparity.shape[1]):
            
            # Check if the left-to-right match corresponds to the right-to-left match
            # use the descriptor to do this (within radius) 
            if disparity_left_to_right[i, j] != disparity_right_to_left[i, j]:
                #print(disparity_left_to_right[i, j], disparity_right_to_left[i, j])
                disparity[i,j] = 0


    cv2.imshow('Valid Map', disparity)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    return disparity
    
# Region-based stereo matching: perform averaging in the neighbourhood to fill these gaps (zeroes)
#   The neighbourhood window can be as big as needed 
#   Some may use quadratic fits in this neighbourhood to get the interpolated values in the gaps

def average_neighborhood(disparity):
    height, width = disparity.shape

    # Define the neighborhood window size - this will be a square
    window_size = 3

    # new disparity map for averages
    avgs = np.zeros_like(disparity, dtype=np.float32)

    # Iterate through original disparity map 
    for y in range(height):
        for x in range(width):

            # Get the window for the neighborhood around each pixel
            y_min, y_max = max(0, y - window_size // 2), min(height, y + window_size // 2 + 1)
            x_min, x_max = max(0, x - window_size // 2), min(width, x + window_size // 2 + 1)

            # Get the average disparity in the neighborhood
            neighborhood = disparity[y_min:y_max, x_min:x_max]
            avgs[y, x] = np.mean(neighborhood)

    # returning an averaged disparity map 
    return avgs