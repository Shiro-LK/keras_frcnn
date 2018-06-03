from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import pandas as pd
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn.get_validation_loss import get_validation_loss, get_validation_lossv2
import tensorflow as tf
from functions import createSummaryTensorboard, TensorboardWrite, PrintException
seed = 10
random.seed(seed)
np.random.seed(seed)

print('current path : ', dir_path)



#####################################
######### PARSER ####################
#####################################
    
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="path", help="Path to training data (txt file).")
parser.add_option("--pi", "--path_image", dest="path_image", help="Path to training image.", default='../')
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                default="simple")

parser.add_option("--sf", "--size_fixed", dest="size_fixed", help="indicate if the fixed size", action="store_true", default=False)
parser.add_option("-s", "--size", dest="size", help="indicate if the fixed size", default=600)
parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg, vgg_lite, or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to store all the metadata related to the training (to be used when testing).",
                default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--use_validation", dest="use_validation", help="Determines if we evaluate against the validation set loss.", action="store_true", default=False)
parser.add_option("--tensorboard_images", dest="tensorboard_images", help="Print boxes of n first images in tensorboard during validation.", default=0)
parser.add_option("--tensorboard_path", dest="tensorboard_path", help="Path to save tensorboard logs.", default='tmp/')
parser.add_option("--logs_path", dest="logs_path", help="Where logs for the losses should be saved.", default='./logs.csv')
parser.add_option("--remove_mean", dest="remove_mean", help="remove mean value in RGB image (default=False)", action="store_true", default=False)
parser.add_option("--channels", dest="channels", help="Number of channels in the image (RGB = 3)", default=3)
parser.add_option("-b", "--bbox_threshold", dest="bbox_threshold", help="bbox_threshold", default=0.8)
parser.add_option("-r", "--overlap_threshold_rpn", dest="overlap_threshold_rpn", help="overlap_threshold_rpn", default=0.7)
parser.add_option("-c", "--overlap_threshold_classifier", dest="overlap_threshold_classifier", help="overlap_thresh_classifier", default=0.5)

(options, args) = parser.parse_args()

if not options.path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
elif options.parser == 'simple_multichan':
    from keras_frcnn.simple_parser_multichan import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.channels = int(options.channels)
C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)
C.remove_mean = bool(options.remove_mean)
C.use_validation = bool(options.use_validation)
C.logs_path = options.logs_path
C.tensorboard_images = int(options.tensorboard_images)
C.tensorboard_path = options.tensorboard_path
C.threshold = [float(options.bbox_threshold), float(options.overlap_threshold_rpn), float(options.overlap_threshold_classifier)]
if options.size is not None:
    C.im_size = int(options.size)
    
if options.size_fixed == False:
    C.fixed_size = False
    print('Input size min image : ', C.im_size)
else:
	C.fixed_size = True
	print('Input size image fixed : ', C.im_size, 'x', C.im_size)
print('Compute Loss Validation : ', C.use_validation)
print('Remove mean from image : ', C.remove_mean)
print('number of images in tensorboard :', C.tensorboard_images)
print('flips horizontal, vertical and rotation : ', C.use_horizontal_flips, C.use_vertical_flips, C.rot_90)
#######################
### Tensorboard #######
#######################
if C.use_validation == True:
    writer_test = tf.summary.FileWriter(C.tensorboard_path+'test')

writer_train = tf.summary.FileWriter(C.tensorboard_path+'train')
########################
#### Choose model ######
########################
num_features = 512
if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'vgg_lite':
    C.network = 'vgg_lite'
    from keras_frcnn import vgg_lite as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
    num_features = 1024
elif options.network == 'mobilenet':
    from keras_frcnn import mobilenet as nn
    C.network = 'mobilenet'

elif options.network == 'squeezenet':
    from keras_frcnn import squeezenet as nn
    C.network = 'squeezenet'
    num_features = 384
else:
    print('Not a valid model')
    raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()


##########################
### Load data and labels #
##########################
if options.parser == 'simple_multichan':   
    all_imgs, classes_count, class_mapping, classes_count_train, classes_count_test = get_data(options.path, path=options.path_image, channels=C.channels)
else:
    all_imgs, classes_count, class_mapping, classes_count_train, classes_count_test = get_data(options.path, path=options.path_image)

print('##### Count train and test data ####', classes_count_train, classes_count_test)
      
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

print('Training and testing images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

C.rpn_stride = C.im_size//nn.get_img_output_length(C.im_size, C.im_size)[0]
print('RPN stride : ', C.rpn_stride)
config_output_filename = options.config_filename

#############################
### Save config file ########
#############################

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'training']
val_imgs = [s for s in all_imgs if s['imageset'] == 'testing']

print('Num train samples {}'.format(len(train_imgs)))
print('Num test samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')


########################
#### Load model ########
########################
if K.image_dim_ordering() == 'th':
    input_shape_img = (C.channels, None, None)
    input_shape_features = (num_features, None, None)
else:# tensorflow backend
    input_shape_img = (None, None, C.channels)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))
feature_map_input = Input(shape=input_shape_features)# used for prediction validation

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True, channels=C.channels)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
classifier_only = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn)

model_classifier_only = Model([feature_map_input, roi_input], classifier_only)
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)


try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True, skip_mismatch = True)
    model_classifier.load_weights(C.base_net_weights, by_name=True, skip_mismatch = True)
    model_classifier_only.load_weights(C.base_net_weights, by_name=True, skip_mismatch = True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)

model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), None])
model_classifier_only.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

model_all.summary()

################
## Parameters ##
################
epoch_length = len(train_imgs)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()
loss_log = list()

best_loss = np.Inf

if C.use_validation:
    val_best_loss = np.Inf


class_mapping_inv = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping_inv[v]: np.random.randint(0, 255, 3) for v in class_mapping_inv}
##############
## Training ##
##############
print('Starting training')


for epoch_num in range(0,num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num+1 , num_epochs))
    
    while True:
		# continue until we reach the number of iteration necessary for one epoch.
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            # data_gen_train yields a list [x_img, [y_rpn_cls, y_rpn_reg], img_dict]	
            # X: [1,H,W,3], raw input image
            # Y: [y_rpn_cls, y_rpn_reg]
            #     y_rpn_cls: [1,H',W',2*num_anchors], anchor validality (0/1) + anchor class(0/1). H',W' is feature map shape
            #     y_rpn_regr: [1,H',W',8*num_anchors], anchor class (4 duplicate) + regression (x1,y1,w,h)
            # img_data: a dict obj storing bbox and image width, height
             
            # Y is now anchors, X Y are 4D tensors
            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)
            #loss_rpn = loss_rpn[0:2]
            
            # The ouput of model_rpn is [x_class,x_regr]
            # P_rpn[0] = x_class: (1, H',W', num_anchor) the softmax probability for objectness classification
            # P_rpn[1] = x_regr: (1, H', W',4*num_anchor) the bbox regression result [x1,y1,w,h]

            P_rpn = model_rpn.predict_on_batch(X)
            
            # R : (n,4) Each row stores (x1,y1,x2,y2), n is the number of boxes returned after non maximum suppression
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=C.threshold[1], max_boxes=300)
            
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format

            # X2: (1, N', 4), cast (x1,y1,x2,y2) in result to (x1,y1,w,h), where N' is the number of bobx returned from non
            # maximum suppression. 
            # Y1: (1 , N' , K), each row is a binary class vector (only 0/1). K excludes background
            # Y2: (1 , N' , 8(K-1)), each row stores [4 labels, 4 regression values (tx,ty,tw,th)]
            # The labels are (1 1 1 1) for not background class and (0 0 0 0) for background class
            # The regression values are (tx,ty,tw,th)
            # IoUs: 2D matrix (N',1) best iou value (classifier_min_overlap,1) for each bbox in R     
                
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

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
            
            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

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
            # In the model_classifier
            # The pred_cls shape is (1, N',K). 
            # The pred_reg shape is (1, N', 4*(K-1)) which exclude background classes
                
                
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                      ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                
                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []
                
                if C.use_validation:
                    val_losses = get_validation_lossv2(data_gen_val, len(val_imgs),
                                                     model_rpn, model_classifier, model_classifier_only, C, class_mapping_inv, class_to_color, writer_tensorboard = writer_test, num_epoch=epoch_num)

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Total losss :{}'.format(curr_loss))
                    
                    train_summary = createSummaryTensorboard(mean_overlapping_bboxes, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss)
                    TensorboardWrite(writer_train, train_summary, step=epoch_num+1)

                    if C.use_validation:
                        print(('Validation mean number of bounding boxes from RPN overlapping ground truth boxes:' + 
                               '{}'.format(val_losses['mean_overlapping_bboxes'])))
                        print('Validation classifier accuracy for bounding boxes from RPN: {}'.format(val_losses['class_acc']))
                        print('Validation loss RPN classifier: {}'.format(val_losses['loss_rpn_cls']))
                        print('Validation loss RPN regression: {}'.format(val_losses['loss_rpn_regr']))
                        print('Validation loss Detector classifier: {}'.format(val_losses['loss_class_cls']))
                        print('Validation loss Detector regression: {}'.format(val_losses['loss_class_regr']))
                        print('Validation total loss :{}'.format(val_losses['curr_loss'])) 
                        
                        test_summary = createSummaryTensorboard(val_losses['mean_overlapping_bboxes'], val_losses['class_acc'], val_losses['loss_rpn_cls'], val_losses['loss_rpn_regr'],
                                                                val_losses['loss_class_cls'], val_losses['loss_class_regr'], val_losses['curr_loss'])
                        TensorboardWrite(writer_test, test_summary, step=epoch_num+1)                         
                    print('Elapsed time: {}'.format(time.time() - start_time))
                iter_num = 0
                start_time = time.time()
                
                if not C.use_validation:
                    loss_log.append({'epoch': epoch_num + 1, 'mean_overlapping_bboxes, ': mean_overlapping_bboxes, 'class_acc': class_acc, 
                                    'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_regr': loss_rpn_regr, 'loss_class_cls': loss_class_cls, 
                                    'loss_class_regr': loss_class_regr, 'curr_loss': curr_loss})
                if C.use_validation:
                    loss_log.append({'epoch': epoch_num + 1, 'mean_overlapping_bboxes, ': mean_overlapping_bboxes, 'class_acc': class_acc, 
                                    'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_regr': loss_rpn_regr, 'loss_class_cls': loss_class_cls, 
                                    'loss_class_regr': loss_class_regr, 'curr_loss': curr_loss, 
                                    'val_mean_overlapping_bboxes, ': val_losses['mean_overlapping_bboxes'], 
                                    'val_class_acc': val_losses['class_acc'], 'val_loss_rpn_cls': val_losses['loss_rpn_cls'], 
                                    'val_loss_rpn_regr': val_losses['loss_rpn_regr'], 'val_loss_class_cls': val_losses['loss_class_cls'], 
                                    'val_loss_class_regr': val_losses['loss_class_regr'], 'val_curr_loss': val_losses['curr_loss']})

                if curr_loss < best_loss:
                    if C.verbose:
                        if not C.use_validation:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                            model_all.save_weights(C.model_path)
                        else:
                            print('Total loss decreased from {} to {}'.format(best_loss,curr_loss))
                    best_loss = curr_loss

                if C.use_validation:
                    if val_losses['curr_loss'] < val_best_loss:
                        if C.verbose:
                            print(('Validation total loss decreased from {} to {}'.format(val_best_loss,val_losses['curr_loss']) +
                                   ', saving weights'))
                        val_best_loss = val_losses['curr_loss']
                        model_all.save_weights(C.model_path)
                    else:
                        if C.verbose:
                            print('Validation total loss did not decrease.')
                
                print('Saving logs')
                pd.DataFrame(loss_log).to_csv(C.logs_path)
                break
                

        except Exception as e:
            print('Exception train: {}'.format(e))
            PrintException()
            continue

print('Saving logs')
pd.DataFrame(loss_log).to_csv(C.logs_path)
print('Training complete, exiting.')
