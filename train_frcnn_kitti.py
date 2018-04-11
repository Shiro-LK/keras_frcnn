"""
this code will train on kitti data set
"""
from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses_fn
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
import os
from keras_frcnn import resnet as nn
# from keras_frcnn import vgg as nn
from keras_frcnn.simple_parser import get_data

'''
Outline of the training process

The faster RCNN network implemented here consists of two classifers with shared inputs: model_rpn + model_classifier.
The model_classifier is a ROI classifier. 

During the training process, for each input image with its ground truth bboxes, 
a set of anchors are generated first. The output of RPN is used to train the ROI classifier. 
The two models are trained separately in each iteration.

'''

def train_kitti():
	# config for data argument
	cfg = config.Config()

	cfg.use_horizontal_flips = True
	cfg.use_vertical_flips = True
	cfg.rot_90 = True
	cfg.num_rois = 32
	cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path())

	# TODO: the only file should to be change for other data to train
	cfg.model_path = './model/kitti_frcnn_last.hdf5'
	cfg.simple_label_file = 'kitti_simple_label.txt'

	# classes_count: a dict storing the number of images in each class (class name as key)
	# classes_mapping: a dict storing the class idx for each class, (class name as key) 
	all_images, classes_count, class_mapping = get_data(cfg.simple_label_file)

	# class_count now contains background category
	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)

	cfg.class_mapping = class_mapping
	with open(cfg.config_save_file, 'wb') as config_f:
		pickle.dump(cfg, config_f)
		print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
			cfg.config_save_file))

	inv_map = {v: k for k, v in class_mapping.items()}

	print('Training images per class:')
	pprint.pprint(classes_count)
	print('Num classes (including bg) = {}'.format(len(classes_count)))
	random.shuffle(all_images)
	num_imgs = len(all_images)
	train_imgs = [s for s in all_images if s['imageset'] == 'trainval']
	val_imgs = [s for s in all_images if s['imageset'] == 'test']

	print('Num train samples {}'.format(len(train_imgs)))
	print('Num val samples {}'.format(len(val_imgs)))

	# data_gen_train is a list [x_img, [y_rpn_cls, y_rpn_reg], img_dict]
	# x_img: [1,H,W,3], raw input image
	# y_rpn_cls: [1,H',W',2*num_anchors], anchor validality (0/1) + anchor class(0/1). H',W' is feature map shape
	# y_rpn_regr: [1,H',W',8*num_anchors], anchor class (4 duplicate) + regression (x1,y1,w,h)
	# img_data_aug: a dict obj storing bbox and image width, height
	
	data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
												   K.image_dim_ordering(), mode='train')
	data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,
												 K.image_dim_ordering(), mode='val')

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
	else:
		input_shape_img = (None, None, 3)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(None, 4))

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
	
	# Create rpn net
	# rpn is a list [x_class,x_regr,shared_layers]
	# x_class: the output layer of rpn classification, a placeholder (H',W',num_anchor)
	# x_regr: the output layer of rpn regression, a placeholder (H',W',4*num_anchor)
	# shared_layers: input base layers
	
	rpn = nn.rpn(shared_layers, num_anchors)

	# ROI classifier
	classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True)
	
	# Create the RPN with input of img_input and output of x_class and x_regr
	model_rpn = Model(img_input, rpn[:2])
	
	# Create the ROI classifier with input of img_input and roi_input and output of classifier
	model_classifier = Model([img_input, roi_input], classifier)

	# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
	model_all = Model([img_input, roi_input], rpn[:2] + classifier)

	try:
		print('loading weights from {}'.format(cfg.base_net_weights))
		model_rpn.load_weights(cfg.model_path, by_name=True)
		model_classifier.load_weights(cfg.model_path, by_name=True)
	except Exception as e:
		print(e)
		print('Could not load pretrained model weights. Weights can be found in the keras application folder '
			  'https://github.com/fchollet/keras/tree/master/keras/applications')

	optimizer = Adam(lr=1e-5)
	optimizer_classifier = Adam(lr=1e-5)
	# Two loss functions are used to compute y_cls and y_regr respectively
	model_rpn.compile(optimizer=optimizer,
					  loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
	# Two loss functions are used to compute y_cls and y_regr respectively
	# Note that the ROI regression loss excludes background category
	model_classifier.compile(optimizer=optimizer_classifier,
							 loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
							 metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
	model_all.compile(optimizer='sgd', loss='mae')

	epoch_length = 1000
	num_epochs = int(cfg.num_epochs)
	iter_num = 0

	losses = np.zeros((epoch_length, 5))
	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []
	start_time = time.time()

	best_loss = np.Inf

	class_mapping_inv = {v: k for k, v in class_mapping.items()}
	print('Starting training')

	vis = True

	for epoch_num in range(num_epochs):

		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

		while True:
			try:

				if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []
					print(
						'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
							mean_overlapping_bboxes, epoch_length))
					if mean_overlapping_bboxes == 0:
						print('RPN is not producing bounding boxes that overlap'
							  ' the ground truth boxes. Check RPN settings or keep training.')

				# data_gen_train yields a list [x_img, [y_rpn_cls, y_rpn_reg], img_dict]
				
				# X: [1,H,W,3], raw input image
				# Y: [y_rpn_cls, y_rpn_reg]
				#     y_rpn_cls: [1,H',W',2*num_anchors], anchor validality (0/1) + anchor class(0/1). H',W' is feature map shape
				#     y_rpn_regr: [1,H',W',8*num_anchors], anchor class (4 duplicate) + regression (x1,y1,w,h)
				# img_data: a dict obj storing bbox and image width, height

				# Y is now the ground truth, X Y are 4D tensors
				X, Y, img_data = next(data_gen_train)

				loss_rpn = model_rpn.train_on_batch(X, Y)

				# The ouput of model_rpn is [x_class,x_regr]
				# P_rpn[0] = x_class: (1, H',W', num_anchor) the softmax probability for objectness classification
				# P_rpn[1] = x_regr: (1, H', W',4*num_anchor) the bbox regression result [x1,y1,w,h]

				P_rpn = model_rpn.predict_on_batch(X)
				
				# result: (n,4) Each row stores (x1,y1,x2,y2), n is the number of boxes returned after non maximum suppression
				
				result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
												overlap_thresh=0.7,
												max_boxes=300)
				
				# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
				
				# X2: (1, N', 4), cast (x1,y1,x2,y2) in result to (x1,y1,w,h), where N' is the number of bobx returned from non
				# maximum suppression. 
				# Y1: (1 , N' , K), each row is a binary class vector (only 0/1). K excludes background
				# Y2: (1 , N' , 8(K-1)), each row stores [4 labels, 4 regression values (tx,ty,tw,th)]
				# The labels are (1 1 1 1) for not background class and (0 0 0 0) for background class
				# The regression values are (tx,ty,tw,th)
				# IoUs: 2D matrix (N',1) best iou value (classifier_min_overlap,1) for each bbox in R                 
				
				X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)

				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue
				
				# The last axis is class label and the last one represent background
				# neg_samples are these classified as background
				# pos_samples are these classified as non-background
				
				neg_samples = np.where(Y1[0, :, -1] == 1)
				pos_samples = np.where(Y1[0, :, -1] == 0)
				
				# cast (1, N, K) to (N, K)
				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []
				# cast (1, N, K) to (N, K)
				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []

				rpn_accuracy_rpn_monitor.append(len(pos_samples))
				rpn_accuracy_for_epoch.append((len(pos_samples)))

				# Important: Keep balance between the number of positive and negative samples
				# sel_samples is randomly generated indices
				
				if cfg.num_rois > 1:
					if len(pos_samples) < cfg.num_rois // 2:
						selected_pos_samples = pos_samples.tolist()
					else:
						selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
					try:
						selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
																replace=False).tolist()
					except:
						selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
																replace=True).tolist()

					sel_samples = selected_pos_samples + selected_neg_samples
				else:
					# in the extreme case where num_rois = 1, we pick a random pos or neg sample
					selected_pos_samples = pos_samples.tolist()
					selected_neg_samples = neg_samples.tolist()
					if np.random.randint(0, 2):
						sel_samples = random.choice(neg_samples)
					else:
						sel_samples = random.choice(pos_samples)
				
				# Train the output of rpn using the roi classifier 
				# X is raw image: image input to the ROI classifier
				# X2: (1, N, 4) is the (x1,y1,w,h): input to the ROI classifier
				# Y1: (1 , N' , K), true class including background
				# Y2: (1 , N' , 8(K-1)), true regression [4 labels, 4 regression values (tx,ty,tw,th)], 
				# 
				# In the model_classifier
				# The pred_cls shape is (1, N',K). 
				# The pred_reg shape is (1, N', 4*(K-1)) which exclude background classes
				
				loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
															 [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

				losses[iter_num, 0] = loss_rpn[1]
				losses[iter_num, 1] = loss_rpn[2]

				losses[iter_num, 2] = loss_class[1]
				losses[iter_num, 3] = loss_class[2]
				losses[iter_num, 4] = loss_class[3]

				iter_num += 1

				progbar.update(iter_num,
							   [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
								('detector_cls', np.mean(losses[:iter_num, 2])),
								('detector_regr', np.mean(losses[:iter_num, 3]))])

				if iter_num == epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
					rpn_accuracy_for_epoch = []

					if cfg.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
							mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))
						print('Elapsed time: {}'.format(time.time() - start_time))

					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					iter_num = 0
					start_time = time.time()

					if curr_loss < best_loss:
						if cfg.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
						best_loss = curr_loss
						model_all.save_weights(cfg.model_path)

					break

			except Exception as e:
				print('Exception: {}'.format(e))
				# save model
				model_all.save_weights(cfg.model_path)
				continue
	print('Training complete, exiting.')


if __name__ == '__main__':
	train_kitti()
