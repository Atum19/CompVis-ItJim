import os
import cv2
import time
import pylab
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


image_filename = '../images_hm_task1/hearts 1.png'  # or leaves.JPG
image = cv2.imread(image_filename)
# image = cv2.resize(image, (image.shape[1] // scale_factor, image.shape[0] // scale_factor))

# Add text and show the initial image
im_to_show = image.copy()
# im_to_show = cv2.putText(im_to_show, 'Initial image', (50, 50),
#                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('', im_to_show)
cv2.waitKey(0)
# cv2.destroyWindow('')