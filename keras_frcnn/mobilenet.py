# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:30:03 2018

@author: shiro
"""

"""MobileNet v1 models for Keras.
MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.
MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.
The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).
The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------
The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------
The weights for all 16 models are obtained and translated
from TensorFlow checkpoints found at
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, DepthwiseConv2D, Conv2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed, ZeroPadding2D, Reshape, Dropout, Activation
from keras.engine.topology import get_source_inputs
from keras.utils import layer_util, conv_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.roi_pooling_conv import RoiPoolingConv
from keras import initializers, regularizers, constraints 

BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'

def relu6(x):
    return K.relu(x, max_value=6)

def get_weight_path(alpha=1.0, rows=224):
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
        weigh_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weigh_path, cache_subdir='models')   
            
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
        #res = depthwise_compute(res, stride=2)
        #res = depthwise_compute(res)
        return res
    return get_output_length(width), get_output_length(height)

def get_img_std_class(width, height):
    def get_output_length(input_length):

        res = depthwise_compute(input_length, stride=2)
        res = depthwise_compute(res)
        return res
    return get_output_length(width), get_output_length(height)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), trainable=False, channels=3):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    if channels == 3:
        
        x = Conv2D(filters, kernel,
                   padding='valid',
                   use_bias=False,
                   strides=strides,
                   trainable=trainable, 
                   name='conv1')(x)
    else:
        x = Conv2D(filters, kernel,
                   padding='valid',
                   use_bias=False,
                   strides=strides,
                   trainable=trainable,
                   name='conv1_channels_'+str(channels))(x)
        
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, trainable=False):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        trainable=trainable,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               trainable=trainable,
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def _depthwise_conv_block_td(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, trainable=False):
    ## depthwise conv block time distributed
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = TimeDistributed(ZeroPadding2D(padding=(1, 1), name='conv_pad_%d_td' % block_id))(inputs)
    x = TimeDistributed(DepthwiseConv2D((3, 3),
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        trainable=trainable,
                        name='conv_dw_%d_td' % block_id))(x)
    x = TimeDistributed(BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn_td' % block_id))(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = TimeDistributed(Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               trainable=trainable,
               name='conv_pw_%d_td' % block_id))(x)
    x = TimeDistributed(BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn_td' % block_id))(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def nn_base(input_tensor=None, trainable=False, channels=3, alpha=1.0, depth_multiplier=1):
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

    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=trainable, channels=channels)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, trainable=trainable, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), trainable=trainable, block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, trainable=trainable, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), trainable=trainable, block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, trainable=trainable, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), trainable=trainable, block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, trainable=trainable, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, trainable=trainable, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, trainable=trainable, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, trainable=trainable, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, trainable=trainable, block_id=11)
    
    return x


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier_layers(x_, trainable=False, alpha=1.0, depth_multiplier=1):
    x = _depthwise_conv_block_td(x_, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12, trainable=trainable,)
    x = _depthwise_conv_block_td(x, 1024, alpha, depth_multiplier, block_id=13, trainable=trainable)
    
    return x
def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False, alpha=1.0, depth_mult=1):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 1024, 7, 7)


    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    
    out = classifier_layers(out_roi_pool, trainable=True, alpha=1.0, depth_multiplier=1)   
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
