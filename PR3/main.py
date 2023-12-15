# Jenn Werth

# program for a stereo analysis system involving
#   1) Region-based
#   2) Feature-based matching 

import functions as wcv

import numpy as np
import cv2

def main():

    # load images 
    left_image = cv2.imread('./images/barn1.jpg')
    right_image = cv2.imread('./images/barn2.jpg')

    # get user inputted parameters
    # Testing with search_range = 30, template_x = 9, template_y = 9
    distance = input('Enter distance [SAD, SSD, NCC] (in upper case please): ')
    method = input('Enter method [region, feature]: ')

    search_range = int(input('Enter search range (need to be integer): '))
    
    template_x = int(input('Enter template_x size (need to be odd integer): '))
    template_y = int(input('Enter template_y size (need to be odd integer): '))

    # initialize empty disparity map
    disparity_map = np.zeros_like(left_image, dtype=np.float32)

    ###### TESTING REGION BASED #########

    # NOTE: This takes a couple minutes to generate 

    if method == "region":
        if distance == "SSD":
            disparity_map = wcv.region_based(left_image, right_image, wcv.ssd, search_range, template_x, template_y, disparity_map) / template_x

            # save resulting disparity map 
            # multipy by 255 to deal with small values in saving 
            cv2.imwrite('./disparity-maps/region-based-ssd.jpeg', 255*disparity_map)

        elif distance == "SAD":
            disparity_map = wcv.region_based(left_image, right_image, wcv.sad, search_range, template_x, template_y, disparity_map) / template_x
        
            # save resulting disparity map 
            cv2.imwrite('./disparity-maps/region-based-sad.jpeg', 255*disparity_map)

        elif distance == "NCC":
            disparity_map = wcv.region_based(left_image, right_image, wcv.ncc, search_range, template_x, template_y, disparity_map) / template_x

            # save resulting disparity map 
            cv2.imwrite('./disparity-maps/region-based-ncc.jpeg', 255*disparity_map)

        # show the resulting disparity map 
        cv2.imshow('Disparity Map', disparity_map)

        # Validity checking -- average neighborhoods 
        averaged = wcv.average_neighborhood(disparity_map)
        cv2.imshow('Valid - Averaged Map', averaged)
        cv2.imwrite('./valid-maps/region-based'+distance+'.jpeg', 255*averaged)
        
    # Feature Based Matching 
    if method == "feature":
        if distance == "SSD":
            disparity_map = wcv.feature_based(left_image, right_image, wcv.ssd, search_range, template_x, template_y, disparity_map) / template_x
            
            # save resulting disparity map 
            # multipy by 255 to deal with small values in saving 
            cv2.imwrite('./disparity-maps/feature-based-ssd.jpeg', disparity_map)

            valid_disparity_map = wcv.valid_feature_based(left_image, right_image, wcv.ssd, search_range, template_x, template_y, disparity_map)


        elif distance == "SAD":
            disparity_map = wcv.feature_based(left_image, right_image, wcv.sad, search_range, template_x, template_y, disparity_map) / template_x
        
            # save resulting disparity map 
            cv2.imwrite('./disparity-maps/feature-based-sad.jpeg', disparity_map)

            valid_disparity_map = wcv.valid_feature_based(left_image, right_image, wcv.sad, search_range, template_x, template_y, disparity_map)


        elif distance == "NCC":
            disparity_map = wcv.feature_based(left_image, right_image, wcv.ncc, search_range, template_x, template_y, disparity_map) / template_x

            # save resulting disparity map 
            cv2.imwrite('./disparity-maps/feature-based-ncc.jpeg', disparity_map)

            valid_disparity_map = wcv.valid_feature_based(left_image, right_image, wcv.ncc, search_range, template_x, template_y, disparity_map)

        # show the resulting map 
        cv2.imshow('Disparity Map', disparity_map)

        # testing validity check 
        cv2.imshow('Valid Map Image', valid_disparity_map)
        cv2.imwrite('./valid-maps/feature-based-'+distance+'.jpeg', 255*valid_disparity_map)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()