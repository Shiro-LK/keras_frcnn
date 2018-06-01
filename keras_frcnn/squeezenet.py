# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:30:03 2018

@author: shiro
"""

from __future__ import print_function
from __future__ import absolute_import

import keras.backend as K
import warnings
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Activation
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.roi_pooling_conv import RoiPoolingConv
import math

from keras.layers import Convolution2D,AveragePooling2D
from keras.layers import BatchNormalization

from keras.applications.imagenet_utils import _obtain_input_shape

WEIGHTS_PATH = 'https://github.com/wohlert/keras-squeezenet/releases/download/v0.1/squeezenet_weights.h5'
def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        weights_path = get_file('squeezenet_weights.h5', WEIGHTS_PATH, cache_subdir='models')
        return weights_path

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

def _fire(x, filters, name="fire", BN=True, trainable=False):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', trainable=trainable, padding='same', name=name + "_squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', trainable=trainable, padding='same', name=name + "_expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', trainable=trainable, padding='same', name=name + "_expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    if BN == True:
        x = BatchNormalization(name=name+'_bn')(x)
    return x

def _fire_td(x, filters, name="fire_td", BN=True, trainable=False):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = TimeDistributed(Convolution2D(sq_filters, (1, 1), activation='relu', trainable=trainable, padding='same', name=name + "_squeeze1x1"), name='TimeDistributed_' + name + "_squeeze1x1")(x)
    expand1 = TimeDistributed(Convolution2D(ex1_filters, (1, 1), activation='relu', trainable=trainable, padding='same', name=name + "_expand1x1"), name='TimeDistributed_' + name + "_expand1x1")(squeeze)
    expand2 = TimeDistributed(Convolution2D(ex2_filters, (3, 3), activation='relu', trainable=trainable, padding='same', name=name + "_expand3x3"), name='TimeDistributed_' + name + "_expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    if BN == True:
        x = TimeDistributed(BatchNormalization(name=name+'_bn'), name='TimeDistributed_' + name + '_bn')(x)
    return x

def SqueezeNet(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, channels=3, trainable=False):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if channels != 3:
        x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", trainable=trainable, activation="relu", name='conv1_channels_'+str(channels))(img_input)
    else:
        x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", trainable=trainable, activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="valid")(x)
    x = BatchNormalization(name= 'maxpool1_bn')(x)
    
    x = _fire(x, (16, 64, 64), name="fire2", trainable=trainable)
    x = _fire(x, (16, 64, 64), name="fire3", BN = False, trainable=trainable)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="valid", trainable=trainable)(x)
    x = BatchNormalization(name= 'maxpool3_bn')(x)
    
    x = _fire(x, (32, 128, 128), name="fire4", trainable=trainable)
    x = _fire(x, (32, 128, 128), name="fire5", BN=False, trainable=trainable)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="valid")(x)
    x = BatchNormalization(name= 'maxpool5_bn')(x)
    
    x = _fire(x, (48, 192, 192), name="fire6", trainable=trainable)
    x = _fire(x, (48, 192, 192), name="fire7", trainable=trainable)



    if include_top:
        x = Dropout(0.5, name='dropout9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = AveragePooling2D(pool_size=(13, 13), name='avgpool10')(x)
        x = Flatten(name='flatten10')(x)
        x = Activation("softmax", name='softmax')(x)
#    else:
#        if pooling == "avg":
#            x = GlobalAveragePooling2D(name="avgpool10")(x)
#        else:
#            x = GlobalMaxPooling2D(name="maxpool10")(x)

#    model = Model(img_input, x, name="squeezenet")

#    if weights == 'imagenet':
#        weights_path = get_file('squeezenet_weights.h5',
#                                WEIGHTS_PATH,
#                                cache_subdir='models')
#
#        model.load_weights(weights_path, by_name=True)

    return x



def divide(x):
    x = math.ceil(x/2)
    #print(x)
    x = math.floor(x/2)+x%2 -1
    #print(x)
    x = math.floor(x/2)+x%2 -1
    #print(x)
    x = math.floor(x/2)+x%2 -1
    #print(x)
    return int(x)

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return divide(input_length)

    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False, channels=3):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (channels, None, None)
    else:
        input_shape = (None, None, channels)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = SqueezeNet(input_tensor=img_input, include_top=False, weights=None, channels=channels, trainable=trainable)
    #x.summary()
    return x


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier_layers(x_, trainable=False):
    x = _fire_td(x_, (64, 256, 256), name="fire8_td", trainable=trainable)
    x = _fire_td(x, (64, 256, 256), name="fire_td9", trainable=trainable)
    
    return x
def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 7
        input_shape = (num_rois, 7, 7, 512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 512, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, trainable=trainable)
    
    out = TimeDistributed(AveragePooling2D(name='Global_average_Pooling_classifier_layer'), name='TimeDistributed_AVG')(out)
    out = TimeDistributed(Flatten(name='flatten'), name='TimeDistributed_flatten')(out)
    #out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    #out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero', name='dense_class'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero', name='dense_regr'),
                               name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

