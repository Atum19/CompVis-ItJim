#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 04:42:08 2021

@author: autorun
"""
#%%
import os
import cv2
import time
import pylab
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


#%%
def demo_blur_kernels():
    # Make empty (black) image
    image = np.zeros([480, 720, 3], dtype=np.uint8)
    # Add white fields and dots
    image[:240, :360, :] = 255
    image[240:, 360:, :] = 255
    image[118:122, 538:542, :] = 255
    image[350:370, 170:190, :] = 255
    image[118:122, 178:182, :] = 0
    image[350:370, 530:550, :] = 0

    # Show the initial image
    tmp = np.hstack((image, image))
    cv2.imshow('', tmp)
    cv2.waitKey(0)

    for i in range(6):
        kernel = (3 + i * 4, 3 + i * 4)
        image_filtered_box = cv2.boxFilter(image, 0, kernel, borderType=cv2.BORDER_REPLICATE)
        image_filtered_box = cv2.putText(image_filtered_box, 'Box filter ' + str(kernel[0]) + 'x' + str(kernel[1]),
                                         (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        image_filtered_gauss = cv2.GaussianBlur(image, kernel, i + 1, borderType=cv2.BORDER_REPLICATE)
        image_filtered_gauss = cv2.putText(image_filtered_gauss, 'Gaussian filter ' + str(kernel[0]) + 'x' +
                                           str(kernel[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                           0.6, (255, 0, 0), 1, cv2.LINE_AA)

        image_filtered = np.hstack((image_filtered_box, image_filtered_gauss))

        cv2.imshow('', image_filtered)
        cv2.waitKey(0)

    return 0

#%%
def demo_sobel_vs_laplacian():

    real_picture = True  # Use 'False' first then 'True'

    if real_picture:
        scale_factor = 1
        image_filename = '../images_hm_task1/hearts 3.png'  # or leaves.JPG
        image = cv2.imread(image_filename)
        image = cv2.resize(image, (image.shape[1] // scale_factor, image.shape[0] // scale_factor))

        # Add text and show the initial image
        im_to_show = image.copy()
        im_to_show = cv2.putText(im_to_show, 'Initial image', (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('', im_to_show)
        cv2.waitKey(0)

    else:

        # Make empty (black) image
        image = np.zeros([480, 720, 3], dtype=np.uint8)

        # Add white fields
        image[:240, :360, :] = 255
        image[240:, 360:, :] = 255
        image[118:122, 538:542, :] = 255
        image[350:370, 170:190, :] = 255

    # Conversion of the image into another color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Show the initial image
    tmp = np.hstack((image, image))
    cv2.imshow('', tmp)
    cv2.waitKey(0)

    # Calculate Sobel gradients along both axes
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # Join both result images to display simultaneously
    stack_image = np.hstack((sobel_x, sobel_y))

    # Take the normalized absolute value to show both positive and negative gradients
    stack_image = np.absolute(255 * stack_image / np.max(stack_image))
    stack_image = np.uint8(stack_image)

    # Show the Sobel gradient image
    cv2.imshow('', stack_image)
    cv2.waitKey(0)

    # Calculate magnitude of Sobel gradients along both axes
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # and do all the transformations to show the result nicely
    sobel_mag = np.absolute(sobel_mag)
    sobel_mag = sobel_mag / np.max(sobel_mag) * 255
    sobel_mag = np.uint8(sobel_mag)
    sobel_mag = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)
    sobel_mag = cv2.putText(sobel_mag, 'Sobel Magnitude', (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Calculate 2D Laplacian for the same image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # and do all the transformations to show the result nicely
    laplacian = laplacian / np.max(laplacian) * 255
    laplacian = np.uint8(laplacian)
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    laplacian = cv2.putText(laplacian, 'Laplacian', (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Join Sobel magnitude and 2D Laplacian to display simultaneously
    stack_image = np.hstack((sobel_mag, laplacian))
    cv2.imshow('', stack_image)
    cv2.waitKey(0)

    # Calculate Sobel angles of gradients
    sobel_arg = cv2.phase(np.array(sobel_x, np.float32), np.array(sobel_y, dtype=np.float32), angleInDegrees=True)

    # and do all the transformations to show the result nicely
    sobel_arg = sobel_arg / np.max(sobel_arg) * 255
    sobel_arg = np.array(sobel_arg, dtype=np.uint8)
    sobel_arg = cv2.cvtColor(sobel_arg, cv2.COLOR_GRAY2BGR)
    sobel_arg = cv2.putText(sobel_arg, 'Sobel gradient angles', (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Join Sobel magnitude and Sobel angles to display simultaneously
    stack_image = np.hstack((sobel_mag, sobel_arg))
    cv2.imshow('', stack_image)
    cv2.waitKey(0)


#%%
def demo_binarization():

    # image_filename = '../workshops_data/contours_example.jpg'
    image_filename = '../images_hm_task1/hearts 3.png'
    img_bgr = cv2.imread(image_filename)
    img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // 2, img_bgr.shape[0] // 2))

    # Add text and show the initial image
    im_to_show = img_bgr.copy()
    im_to_show = cv2.putText(im_to_show, 'Initial image', (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('', im_to_show)
    cv2.waitKey(0)

    # Conversion of the image into another color space
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply simple threshold with various levels
    levels = [64, 128, 192]
    for level in levels:
        ret, img_thresh = cv2.threshold(img_gray, level, 255, cv2.THRESH_BINARY)

        # Add text and show the threshold image
        img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
        img_thresh = cv2.putText(img_thresh, 'Simple threshold level ' + str(level), (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('', img_thresh)
        cv2.waitKey(0)

    # Apply adaptive mean threshold
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Add text and show the threshold image
    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    img_thresh = cv2.putText(img_thresh, 'Adaptive mean threshold', (50, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)

    # Apply adaptive Gaussian threshold
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    # Add text and show the threshold image
    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    img_thresh = cv2.putText(img_thresh, 'Adaptive Gaussian threshold', (50, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)


#%%
def demo_morphology_vs_median_filter():
    # image_filename = '../workshops_data/Antenna_Handbook.png'
    image_filename = '../images_hm_task1/hearts 6.png'
    img_bgr = cv2.imread(image_filename)

    # Conversion of the image into another color space
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply adaptive Gaussian threshold
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)
    cv2.imshow('', img_bin)
    cv2.waitKey(0)

    # Dilation
    img = cv2.dilate(img_bin, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imshow('', img)
    cv2.waitKey(0)

    # Erosion 1
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imshow('', img)
    cv2.waitKey(0)

    # Erosion 2
    img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, 'Morphology result', (70, 70), 0, 1, [0, 0, 255], 3)
    cv2.imshow('', img)
    cv2.waitKey(0)

    # Median filter of grayscale image
    img_median = cv2.medianBlur(img_bin, 3)
    img_median = cv2.cvtColor(img_median, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_median, 'Median filter result', (70, 70), 0, 1, [0, 0, 255], 3)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    # Comparison
    img_compare = np.hstack((img, img_median))
    cv2.imshow('', img_compare)
    cv2.waitKey(0)


#%%
def demo_morphology_line_detection():

    # image_filename = '../workshops_data/Electrical_specification.png'
    image_filename = '../images_hm_task1/hearts 6.png'
    img_bgr = cv2.imread(image_filename)

    # Conversion of the image into another color space
    img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    im_to_show = cv2.resize(img_bgr, (img_bgr.shape[1] // 5, img_bgr.shape[0] // 5))
    cv2.imshow('', im_to_show)
    cv2.waitKey(0)

    # Dilation
    kernel_length = 100
    img_v = cv2.dilate(img_g, np.ones((kernel_length, 1), np.uint8), iterations=1)
    img_v = cv2.erode(img_v, np.ones((kernel_length, 1), np.uint8), iterations=1)

    img_h = cv2.dilate(img_g, np.ones((1, kernel_length), np.uint8), iterations=1)
    img_h = cv2.erode(img_h, np.ones((1, kernel_length), np.uint8), iterations=1)

    img = cv2.bitwise_and(img_h, img_v)

    cv2.imwrite('../workshops_data/Electrical_specification_template.png', img)
    im_to_show = cv2.resize(img, (img_bgr.shape[1] // 5, img_bgr.shape[0] // 5))
    cv2.imshow('', im_to_show)
    cv2.waitKey(0)


# *******************************************************************************
#                           M A I N    P R O G R A M                            *
# *******************************************************************************


#%%
if __name__ == '__main__':

    # demo_blur_kernels()
    demo_sobel_vs_laplacian()
    # demo_binarization()
    # demo_morphology_vs_median_filter()
    # demo_morphology_line_detection()

