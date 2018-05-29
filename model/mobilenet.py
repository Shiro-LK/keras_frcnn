# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:18:03 2018

@author: shiro
"""
import keras

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
def get_output_size(input):
	# W = (W- F + 2x P)/S  +1 avec P padding, F filtre, S stride
	# conv block zero padding + conv + pooling
	res = input + 2
	print(res)
	res = (res + 2*0 - 3)//2 + 1 
	print(res)
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
	
	
#oldmodel = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, include_top=False, weights='imagenet')
#oldmodel.summary()

#newmodel = keras.applications.mobilenet.MobileNet(input_shape=(None, None,3), alpha=1.0, include_top=False, weights=None)
#newmodel.summary()
#newmodel = copy_weights(oldmodel, newmodel)
#newmodel.save_weights('mobilenet_weights.h5')

n1 = 250
n2 = 400
newmodel = keras.applications.mobilenet.MobileNet(input_shape=(n1, n2,3), alpha=1.0, include_top=False, weights=None)
newmodel.summary()
print(get_output_size(n1))
print(get_output_size(n2))