
import cv2
import numpy as np

def normalize_depth(depth):
    return (depth-depth.min())/(depth.max()-depth.min())

def clean_background(rgba_image, rgb_image):
    
   

    #rgba_image = cv2.imread(rgba_image)
    rgba_image = cv2.imread(rgba_image, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.imread(rgb_image)

    height, width, _ = rgba_image.shape
    #rgb_image_corrected = (rgb_image * 255).astype(np.uint8)

    #cv2.imwrite("./rgba_image_OGI.png", rgba_image_corrected)
    #cv2.imwrite("./rgb_image_pre_DMM.png", rgb_image_corrected)

    # Iterate over all pixels
    for y in range(height):
        for x in range(width):
            r, g, b, a = rgba_image[y, x]  # Extract RGBA values
            # Set RGB to black if alpha is 0
            if a == 0:
                rgb_image[y, x] = 0 # Set RGB pixel to black
    rgb_image_normalized = normalize_depth(rgb_image)
    rgb_image_normalized = (rgb_image_normalized * 255).astype(np.uint8)


    cv2.imwrite("./epth_teddy.png", rgb_image_normalized)
    return rgb_image    

clean_background("./clean.png","./teddy_bear_d.png")