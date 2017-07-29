from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from time import time
from scipy.ndimage import imread
from scipy import misc
from sklearn.preprocessing import LabelBinarizer

from model import Model

# Read in train and test image
TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test/'
PRED_DIR = 'data/pred/'
SAVE_PATH='checkpoints/convnet_face/face-convnet-2'

def load_data(img_dir, labels=False, one_hot=False, binary_class=False):
	img_data = []
	data_labels = []
	if labels:
		# get folder names aka class labels
		class_labels = os.listdir(img_dir)
		for label in class_labels:
			images = os.listdir(img_dir + label)
			# gather image array and its class
			for img in images:
				img_arr = imread(img_dir + label + "/" + img)
				img_data.append(img_arr)
				data_labels.append(label)

	if one_hot:
		data_labels_enc = []
		# convert categorical labels to one-hot array
		enc = LabelBinarizer()
		data_labels = enc.fit_transform(data_labels)

		if binary_class:
			# quick and dirty way of converting binary labels to sparse matrix
			for i in data_labels:
				if i == 1:
					i = [1, 0]
					data_labels_enc.append(i)
				else:
					i = [0, 1]
					data_labels_enc.append(i)

				return np.array(img_data), np.array(data_labels_enc)

		else:
			# multi class encoding
			return np.array(img_data), np.array(data_labels)

def random_flip(img):
	if np.random.choice([True, False]):
		img = np.fliplr(img)
	return img

def random_rotate(img):
	angle = np.random.uniform(low=-10.0, high=10.0)
	return misc.imrotate(img, angle, interp='bicubic')

def augment_data(img_data, img_labels, random_flip_img=True, random_rotate_img=True):
	aug_data = []
	# add augmentation images equal to len of img_data
	for img in img_data:
		if random_flip_img:
			img = random_flip(img)
		if random_rotate_img:
			random_rotate(img)
		aug_data.append(img)

	aug_data = np.array(aug_data)
	img_data = np.vstack((img_data, aug_data))

	# concat augmented labels to orignal labels
	img_labels = np.vstack((img_labels, img_labels))

	# random shuffle data and labels
	s = np.arange(img_data.shape[0])
	np.random.shuffle(s)
	img_data = img_data[s]
	img_labels = img_labels[s]

	return img_data, img_labels


print("Loading train data...")
train_data, train_labels = load_data(TRAIN_DIR, labels=True, one_hot=True)

print("Loading test data...")
test_data, test_labels = load_data(TEST_DIR, labels=True, one_hot=True)

print(train_data.shape)
print(train_labels.shape)

print(test_data.shape)
print(test_labels.shape)

# print(train_data[:2])
# print(train_labels[:2])

# print(test_data[:2])
# print(test_labels[:2])

print("Augmenting train data...")
aug_data, aug_labels = augment_data(train_data, train_labels)

print(aug_data.shape)
# print(len(aug_data))
# print(aug_data[:2])
print(aug_labels.shape)
# print(len(aug_labels))
# print(aug_labels[:2])

# from PIL import Image
# imgtest = Image.fromarray(train_data[0])
# imgtest.show()

face_rcog = Model()

# Train model
print("Training model...")
face_rcog.train(aug_data, aug_labels)

# Test model
print("Evaluating model...")
face_rcog.test(test_data, test_labels, save_path=SAVE_PATH)

# Making predictions
print("Predicting...")

# Load new data for pred
new_data = []
for img in os.listdir(PRED_DIR):
	img_pred = imread(PRED_DIR + '/' + img)
	new_data.append(img_pred)


# print(new_data)

t0 = time()
preds = face_rcog.predict(new_data, save_path=SAVE_PATH)
tt = time() - t0
print(preds)
print('Time taken (s): ', round(tt, 3))
