##########################
## FUNCTIONS TRAIN/TEST ##
##########################
import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import numpy as np
import tensorflow as tf
import sys 
import linecache

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def load_model_weights(model1, model2):
    '''
        model2 load the weight of the moodel1
    '''    
    #exp = ['flatten', 'input']
    for i, layers in enumerate(model2.layers):
      if layers.name.find('flatten') == -1 and layers.name.find('input') == -1 and layers.name.find('roi') == -1 and layers.name.find('activation') == -1 and layers.name.find('add') == -1:
          model2.layers[i].set_weights(model1.get_layer(model2.layers[i].name).get_weights())
    
    
def createSummaryTensorboard(mean_bbox, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss):
    '''
        Create summary so as to plot the loss and the accuracy of the 
        faster rcnn detector during the training and validation
    '''
    summary = tf.Summary()
    summary.value.add(tag="Mean number of bounding boxes from RPN overlapping ground truth boxes", simple_value=mean_bbox)
    summary.value.add(tag="Classifier accuracy bbox rpn", simple_value=class_acc)
    summary.value.add(tag="Loss RPN classifier", simple_value=loss_rpn_cls)
    summary.value.add(tag="Loss RPN regression", simple_value=loss_rpn_regr)
    summary.value.add(tag="Loss Detector classifier", simple_value=loss_class_cls)
    summary.value.add(tag="Loss Detector regression", simple_value=loss_class_regr)
    summary.value.add(tag="Total losss", simple_value=curr_loss)
    return summary

def TensorboardWrite(writer, var_summary, step):
    '''
        write summary on tensorboard
        
    '''
    writer.add_summary(var_summary, step)
    writer.flush()
    
def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
    if C.fixed_size:
        new_width = int(img_min_side)
        new_height = int(img_min_side)
        ratio_width = img_min_side/width
        ratio_height = img_min_side/height
    else:
    
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
            ratio_width = ratio
            ratio_height = ratio
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
            ratio_width = ratio
            ratio_height = ratio
            
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, [ratio_width, ratio_height]

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    if C.remove_mean:
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_original_img_channels(image, C):
    """ formats the image in its original channels based on config """
    #img = img[:, :, (2, 1, 0)]
    img = np.copy(image).astype(np.float32)
    img *= C.img_scaling_factor
    if C.remove_mean:
        img[:, :, 0] += C.img_channel_mean[0]
        img[:, :, 1] += C.img_channel_mean[1]
        img[:, :, 2] += C.img_channel_mean[2]
    
    #img = np.transpose(img, (2, 0, 1))
    #img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

def format_img_tensorboard(img, C):
    """ formats an image for model prediction based on config so as to print in tensorboard """
    img, ratio = format_img_size(img[0,:,:,:], C)
    img = format_original_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 / ratio[0]))
    real_y1 = int(round(y1 / ratio[1]))
    real_x2 = int(round(x2 / ratio[0]))
    real_y2 = int(round(y2 / ratio[1]))

    return (real_x1, real_y1, real_x2 ,real_y2)
    
def load_models_for_test(C, class_mapping):
    if C.network == 'vgg':
        from keras_frcnn import vgg as nn
    elif C.network == 'vgg_lite':
        from keras_frcnn import vgg_lite as nn
    elif C.network == 'mobilenet':
        from keras_frcnn import mobilenet as nn
    elif C.network == 'squeezenet':
        from keras_frcnn import squeezenet as nn
    else:
        import keras_frcnn.resnet as nn
        
    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg' or C.network == 'vgg_lite':
        num_features = 512
    elif C.network == 'squeezenet' or C.network == 'mobilenet':
        num_features = 512

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)


    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier_only.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier_only.compile(optimizer='sgd', loss='mse')
    
    return model_rpn, model_classifier_only
    
def predict_on_image(img, model_rpn, model_classifier_only, C, class_mapping_inv, class_to_color, bbox_threshold = 0.8, overlap_thresh_rpn = 0.7, overlap_thresh_classifier = 0.5, tensorboard=False):
    '''
        predict on only one image
        if tensorboard parmeters is true, we need to do get the original image so remove the preprocessing (during training pass)
        tensorboard is False if we are doing testing pass
    '''
    try:
        # compute image with preprocessing (remove mean) and ratio with/height (array of two ratio)
        if tensorboard == False:
            X, ratio = format_img(img, C)
            if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))
        else:
            X = np.copy(img)
            img, ratio = format_img_tensorboard(img, C)
            #print('ratio with tensorboard :', ratio)
        

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)
        

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=overlap_thresh_rpn)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping_inv[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            # remove double boxes depending of a threshold
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=overlap_thresh_classifier)
            print('number of boxes :', new_boxes.shape[0])
            for jk in range(new_boxes.shape[0]):
                # if there is no boxes, jump loop
                (x1, y1, x2, y2) = new_boxes[jk,:]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                all_dets.append((key,100*new_probs[jk]))
                img = drawBoxOnImage(img, x1=real_x1, y1=real_y1, x2=real_x2, y2=real_y2, key=key, prob=new_probs[jk], class_to_color=class_to_color)
                

        return img, all_dets
    except Exception as e:
            print('Exception predict on image: {}'.format(e))
            PrintException()
    
def drawBoxOnImage(img, x1, y1, x2, y2, key, prob, class_to_color):
    '''
        draw box on image
        input : real coordinate, image, class and its probability and also the color of the class (dic)
        return img with drawing boxes
    '''
    
    cv2.rectangle(img,(x1, y1), (x2, y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
    textLabel = '{}: {}'.format(key,int(100*prob))
    
    (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
    textOrg = (x1, y1-0)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
    cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return img        
    #if len(bboxes.keys()) > 0:
    #    cv2.imwrite('./results_imgs/{}.png'.format(img_name.split()[0]),img)