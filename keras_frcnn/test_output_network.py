# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:09:51 2018

@author: kunl
"""

import keras
from keras.applications.resnet50 import ResNet50

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)

model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(600, 1000, 3), pooling=None, classes=1000)

model.summary()

print('output : ', get_img_output_length(1000, 600))