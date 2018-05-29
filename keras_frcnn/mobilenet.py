# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:30:03 2018

@author: shiro
"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.roi_pooling_conv import RoiPoolingConv



def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        return 'model\\mobilenet_weights.h5'

def copy_weights(oldmodel, newmodel):
    dic_w = {}
    for layer in oldmodel.layers:
        dic_w[layer.name] = layer.get_weights()
    
    for i, layer in enumerate(newmodel.layers):
        if layer.name in dic_w and layer.name != 'softmax' and layer.name != 'input':
            #print(newmodel.layers[i].get_weights()[0].shape)
            #print(newmodel.layers[i].get_weights()[0][:,:,0,0])
            newmodel.layers[i].set_weights(dic_w[layer.name])
            print(layer.name)
            #print(newmodel.layers[i].get_weights()[0][:,:,0,0])
    return newmodel

def depthwise_compute(input, stride=1, padding=1, padding_filter=0, filter=3):
    #zero padding
    input = input + 2*padding
    return (input -filter + 2*padding_filter)//stride +1
    
def get_img_output_length(width, height):
    def get_output_length(input_length):
        res = input_length + 2
        res = (res + 2*0 - 3)//2 + 1 
        # depthwise
        res = depthwise_compute(res)
        res = depthwise_compute(res, stride=2)
        res = depthwise_compute(res)
        res = depthwise_compute(res, stride = 2)
        res = depthwise_compute(res)
        res = depthwise_compute(res, stride = 2)
        res = depthwise_compute(res)
        res = depthwise_compute(res)
        res = depthwise_compute(res)
        res = depthwise_compute(res)
        res = depthwise_compute(res)
        res = depthwise_compute(res, stride=2)
        res = depthwise_compute(res)
        return res
    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = MobileNet(input_shape=input_shape, input_tensor=img_input, alpha=1.0, include_top=False, weights=None)
    print(x.layers[-1].name)
    return x.layers[-1].output


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    pooling_regions = 7
#    num_features = 3
#    if K.backend() == 'tensorflow':
#        pooling_regions = 7
#        input_shape = (num_rois, 7, 7, num_features)
#    elif K.backend() == 'theano':
#        pooling_regions = 7
#        input_shape = (num_rois, num_features, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'), name='TimeDistributed_flatten')(out_roi_pool)
    #out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    #out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
