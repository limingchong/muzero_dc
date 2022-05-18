#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:30:35 2017

@author: artur
"""


def data_readIn_and_subdivision_XoXo(path):
    import numpy as np
    import os
    from scipy import misc
    import imageio

    def data_subdivision(ALLimages, ALLimage_labels):
        import numpy
        [images, image_labels] = shuffle_data(ALLimages, ALLimage_labels)
        nr_images = images.shape[0]
        images_train = images[0:numpy.int16(nr_images * 0.7), :, :, :]
        image_labels_train = image_labels[0:numpy.int16(nr_images * 0.7), :]
        images_test = images[numpy.int16(nr_images * 0.7):numpy.int16(nr_images * 0.85), :, :, :]
        image_labels_test = image_labels[numpy.int16(nr_images * 0.7):numpy.int16(nr_images * 0.85), :]
        images_valid = images[numpy.int16(nr_images * 0.85):, :, :, :]
        image_labels_valid = image_labels[numpy.int16(nr_images * 0.85):]
        return images_train, image_labels_train, images_test, image_labels_test, images_valid, image_labels_valid

    images = np.empty([2000, 116, 116], dtype=np.float32)
    labels = np.zeros([2000, 2], dtype=np.float32)
    for i in range(1000):
        os.chdir(path + 'training_data_sm/circles_sm/')
        images[i, :, :] = np.float32(imageio.imread('ci' + str(i + 1) + '.bmp') / 255)
        labels[i, 0] = 1
        os.chdir(path + 'training_data_sm/crosses_sm/')
        images[i + 1000, :, :] = np.float32(imageio.imread('cr' + str(i + 1) + '.bmp') / 255)
        labels[i + 1000, 1] = 1
    [images, labels] = shuffle_data(images, labels)
    images = np.expand_dims(images, axis=3)  # add depth dimension of 1 since grayscale

    return data_subdivision(images, labels)


def shuffle_data(images, image_labels):
    import numpy
    random_seed = numpy.int64(numpy.round(numpy.random.rand(1) * 1000))
    numpy.random.seed(random_seed)
    numpy.random.shuffle(images)
    numpy.random.seed(random_seed)
    numpy.random.shuffle(image_labels)
    return images, image_labels
