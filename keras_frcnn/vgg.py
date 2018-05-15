# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

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
		return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
	'''
	A function calculates the shape of feature map from the shape of input image
	'''
	def get_output_length(input_length):
		return input_length//16

	return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False):
	'''
	Create a base net starting from input to the feature map
	
	# Return
		| a cnn network starting from input to the feature map

	'''
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

	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	return x


def rpn(base_layers, num_anchors):
	'''
	Construct a region pooling net (RPN) from base layers
	
	# Args
		| base_layers: base net containing the layers from input to feature map
		| num_anchors: anchor_scales　* anchor_aspects
	
	# Yield
		| x_class: the output layer of rpn classification, a placeholder with output shape of (samples, H',W',num_anchor) in tf backend
		| x_regr: the output layer of rpn regression, a placeholder with output shape of (samples, H',W', 4 * num_anchor) in tf backend
		| base_layers: input base net
	
	'''
	x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
		base_layers)

	x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
	x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

	return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
	'''
	Construct a ROI classifier from base layers and input roi. Only ROI classifier uses
	the TimeDistributed Layer. 
	
	# Args
		| base_layers: base layers
		| input_rois: a placeholder for input rois  
		| num_rois: the number of ROI to be processed each time (? to be verified)
		| num_classes: number of classes including background
		
	# Return
		| out_class: (num_rois, nb_classes) softmax probailities
		| out_regr: (num_rois, 4*(nb_classes-1)) regression result
	'''
	# compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

	# The pooling_region defines the fixed feature map used in ROI pooling layer

	if K.backend() == 'tensorflow':
		pooling_regions = 7
		input_shape = (num_rois, 7, 7, 512)
	elif K.backend() == 'theano':
		pooling_regions = 7
		input_shape = (num_rois, 512, 7, 7)

	out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

	out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
	out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
	out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
	# https://keras.io/layers/core/#dense
	# Input nD tensor with shape: (batch_size, ..., input_dim)
	# Output has shape (batch_size, ..., units).
	out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
								name='dense_class_{}'.format(nb_classes))(out)
	# note: no regression target for bg class
	out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
							   name='dense_regress_{}'.format(nb_classes))(out)

	return [out_class, out_regr]
