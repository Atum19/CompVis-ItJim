'''
USAGE:
python hog_image_recognition.py --path person_car_cup
python hog_image_recognition.py --path flowers
'''

import os
import cv2
import argparse

from sklearn.svm import LinearSVC
from skimage import feature

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='what folder to use for HOG description', 
					choices=['flowers', 'person_car_cup'])
args = vars(parser.parse_args())

images = []
labels = []
# get all the image folder paths
image_paths = os.listdir(f"input/{args['path']}")
for path in image_paths:
	# get all the image names
    all_images = os.listdir(f"input/{args['path']}/{path}")

	# iterate over the image names, get the label
    for image in all_images:
        image_path = f"input/{args['path']}/{path}/{image}"
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 256))

        # get the HOG descriptor for the image
        hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    
        # update the data and labels
        images.append(hog_desc)
        labels.append(path)


# train Linear SVC 
print('Training on train images...')
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)

# predict on the test images
print('Evaluating on test images...')
# loop over the test dataset
for (i, imagePath) in enumerate(os.listdir(f"test_images/{args['path']}/")):

	image = cv2.imread(f"test_images/{args['path']}/{imagePath}")
	resized_image = cv2.resize(image, (128, 256))

	# get the HOG descriptor for the test image
	(hog_desc, hog_image) = feature.hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
	# prediction
	pred = svm_model.predict(hog_desc.reshape(1, -1))[0]

	# convert the HOG image to appropriate data type. We do...
	# ... this instead of rescaling the pixels from 0. to 255.
	hog_image = hog_image.astype('float64')
	# show thw HOG image
	cv2.imshow('HOG Image', hog_image)

	# put the predicted text on the test image
	cv2.putText(image, pred.title(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 2)
	cv2.imshow('Test Image', image)
	cv2.imwrite(f"outputs/{args['path']}_hog_{i}.jpg", hog_image*255.) # multiply by 255. to bring to OpenCV pixel range
	cv2.imwrite(f"outputs/{args['path']}_pred_{i}.jpg", image)
	cv2.waitKey(0)

