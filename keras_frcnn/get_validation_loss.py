import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras import backend as K
import numpy as np

import random
from functions import predict_on_image
import tensorflow as tf
from PIL import Image
import io
from functions import PrintException, load_model_weights

#def PrintException():
#    exc_type, exc_obj, tb = sys.exc_info()
#    f = tb.tb_frame
#    lineno = tb.tb_lineno
#    filename = f.f_code.co_filename
#    linecache.checkcache(filename)
#    line = linecache.getline(filename, lineno, f.f_globals)
#    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def TensorboardImage(writer, image, number):
    '''
        writer : 
        images : image OPENCV format. (value between 0 and 255 because of format_img_tensorboard function)
    '''
    

    height, width, channel = image.shape
    output = io.BytesIO()
        
    # temp = images[i,:,:,:]*255
    temp = np.copy(image)
    if channel == 1:
        temp = np.repeat(temp, 3, axis=2)
    temp = temp.astype('uint8')
        
    temp = Image.fromarray(temp)
    temp.save(output, format='JPEG')
    image_string = output.getvalue()
    output.close()
    img = tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)
        #img = sess.run(tf.summary.image(name='image'+str(i), tensor=np.expand_dims(images[i,:,:,:], axis=0), max_outputs=1))

    summary =  tf.Summary(value=[tf.Summary.Value(tag='image_'+str(number), image=img )])
    writer.add_summary(summary)

def get_validation_lossv2(data_gen_val, epoch_length, model_rpn, model_classifier, model_classifier_only, C, class_mapping_inv, class_to_color, writer_tensorboard=None, threshold=[0.8, 0.7, 0.5]):
    '''
        compute loss on validation data. Can also print images on tensorboard with the boxes predicted.
        threshold : bbox, rpn, classifier
    '''
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    class_mapping = C.class_mapping
    
    iter_num = 0
    
    progbar = generic_utils.Progbar(epoch_length)
    print('Validating')
    
    for epoch_num in range(0, epoch_length):

        try:
            #print('begin try, iter num : {} , epoch num {}'.format(iter_num, epoch_num))
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations (validation)'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_val)

            loss_rpn = model_rpn.evaluate(X, Y, verbose=0)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=threshold[1], max_boxes=300)
            
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                iter_num +=1
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

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

            loss_class = model_classifier.evaluate([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]], verbose=0)

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, 
                           [('val_rpn_cls', np.mean(losses[:iter_num, 0])), ('val_rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('val_detector_cls', np.mean(losses[:iter_num, 2])), ('val_detector_regr', np.mean(losses[:iter_num, 3]))]
                          )
            #print('end try before if, iter num : {} , epoch num {}'.format(iter_num, epoch_num))
            if iter_num == epoch_length or iter_num == epoch_length-1:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                
            if iter_num - 1 < C.tensorboard_images:
                try:
                    load_model_weights(model_classifier, model_classifier_only)
                except Exception as e:
                    print('Exception Validation: iter num {}, {}'.format(iter_num, e))
                    PrintException()
                    exit()
                
                img, all_dets = predict_on_image(np.copy(X), model_rpn, model_classifier_only, C, class_mapping_inv, class_to_color, bbox_threshold = threshold[0], overlap_thresh_rpn = threshold[1], overlap_thresh_classifier = threshold[2], tensorboard=True)
                print('writer tensorboard')
                TensorboardImage(writer_tensorboard, img, iter_num - 1)
            
            #print('end try end if, iter num : {} , epoch num {}'.format(iter_num, epoch_num))
        except Exception as e:
            print('Exception Validation: iter num {}, {}'.format(iter_num, e))
            PrintException()
            continue
    print('end validation loss loop, iter num' + str(iter_num))
    return {'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_regr': loss_rpn_regr, 'loss_class_cls': loss_class_cls, 
            'loss_class_regr': loss_class_regr, 'class_acc': class_acc, 'curr_loss': curr_loss, 
            'mean_overlapping_bboxes': mean_overlapping_bboxes}


def get_validation_loss(data_gen_val, epoch_length, model_rpn, model_classifier, C, writer_tensorboard=None):
    
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    class_mapping = C.class_mapping
    
    iter_num = 0
    
    progbar = generic_utils.Progbar(epoch_length)
    print('Validating')
    
    for epoch_num in range(0, epoch_length):

        try:
            #print('begin try, iter num : {} , epoch num {}'.format(iter_num, epoch_num))
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations (validation)'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_val)

            loss_rpn = model_rpn.evaluate(X, Y, verbose=0)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                iter_num +=1
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

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

            loss_class = model_classifier.evaluate([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]], verbose=0)

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, 
                           [('val_rpn_cls', np.mean(losses[:iter_num, 0])), ('val_rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('val_detector_cls', np.mean(losses[:iter_num, 2])), ('val_detector_regr', np.mean(losses[:iter_num, 3]))]
                          )
            #print('end try before if, iter num : {} , epoch num {}'.format(iter_num, epoch_num))
            if iter_num == epoch_length or iter_num == epoch_length-1:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)     
            #print('end try end if, iter num : {} , epoch num {}'.format(iter_num, epoch_num))
        except Exception as e:
            print('Exception: iter num {}, {}'.format(iter_num, e))
            PrintException()
            continue
    print('end validation loss loop, iter num' + str(iter_num))
    return {'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_regr': loss_rpn_regr, 'loss_class_cls': loss_class_cls, 
            'loss_class_regr': loss_class_regr, 'class_acc': class_acc, 'curr_loss': curr_loss, 
            'mean_overlapping_bboxes': mean_overlapping_bboxes}
