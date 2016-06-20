# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from scipy.misc import imread, imresize

import os
import numpy as np
import tensorflow as tf

# Data loading and preprocessing
cwd = os.getcwd()
loadpath = cwd + "/processedData.npz"
l = np.load(loadpath)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', None, 'Checkpoint file')
flags.DEFINE_boolean('eval', False, 'Whether to train or evaluate')
flags.DEFINE_string('image', None, 'Image file for evaluation')

# Parse data
X = l['trainimg']
Y = l['trainlabel']
X_test = l['testimg']
Y_test = l['testlabel']

nclass = 36
letters = 6

print(Y.shape)
Y = np.reshape(Y, [-1, nclass * letters])
Y_test = np.reshape(Y_test, [-1, nclass * letters])

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()


# Convolutional network building
network = input_data(shape=[None, 125, 30, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 48, [5, 5], activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 48, [5, 5], activation='relu')
network = max_pool_2d(network, [1, 2])
network = conv_2d(network, 64, [5, 5], activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 2048, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, letters * nclass, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

def processImage():
    if FLAGS.image != None:
        currimg  = imread(FLAGS.image)
        vec = imresize(currimg, [36, 36])/255.
        return vec

if FLAGS.eval == False:
    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=10, run_id='cifar10_cnn')
else:
    if FLAGS.checkpoint != None:
        model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="checkpoints/")
        model.load(FLAGS.checkpoint)
        img = processImage()
        img = np.reshape(img, [1, 36, 36, 3])
        predictions = model.predict(img)
        print(np.argmax(predictions))
    else:
        print("No checkpoint file specified, aborting")
