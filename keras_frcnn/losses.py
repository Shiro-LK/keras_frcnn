from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	'''
	Compute rpn objectness classification loss. The loss is log loss normalized by 
	the number of positive and negative anchors. Details refer to Faster RCNN paper.
	
	# Args
		| num_anchors: number of anchors (scales * aspects)
	'''
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		'''
		# Args
			| y_true: (1, H, W, 2 * num_anchors), bbox label calculated from rpn, [valid, label]	
			valid is 1 for positive or negative anchors and 0 for neural anchors. 
			Label is 1 for positive and 0 for negative anchors (non target anchor). 
			| y_pred: (1, H, W, num_anchors), bbox softmax probability predicted by the network
		
		# Return
			| normalized log loss (only positive and negative anchor considered)
		'''
		if K.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	'''
	Calculate ROI regression loss
	
	# Args:
		| num_classes: K-1, the number of classes excluding background
	'''
	def class_loss_regr_fixed_num(y_true, y_pred):
		'''
		Calculate ROI regression loss
		
		# Args
			| y_pred: (1, N, 4(K-1)) predicted regression, background does not have a regression output
			| y_true (1, N, 8(K-1)), each row stores [4 labels, 4 regression values (tx,ty,tw,th)]
			The 4 labels are 1111 for specific class
		
		# Return
			| Loss computed based on the paper faster RCNN
		
		'''		
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	'''
	Cross entropy loss in ROI classifier
	
	# Input
		| y_true: (1, N, K), binary class vector including background
		| y_pred: (1, N, K), binary class vector including background
	'''
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
