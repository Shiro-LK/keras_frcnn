from __future__ import division
import os

import numpy as np
import sys
import pickle
from optparse import OptionParser
from keras_frcnn import config
from functions import load_models_for_test, predict_on_image
import time
import cv2

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default=None)
parser.add_option("-s", "--save", dest="save", help="save", default=False)
parser.add_option("-i", "--image", dest="test_image", help="Path to test one image.", default=None)
parser.add_option("-n", "--num_rois", dest="num_rois",
                help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config.pickle")
parser.add_option("-b", "--bbox_threshold", dest="bbox_threshold", help="bbox_threshold", default=0.8)
parser.add_option("-r", "--overlap_threshold_rpn", dest="overlap_threshold_rpn", help="overlap_threshold_rpn", default=0.7)
parser.add_option("-c", "--overlap_thresh_classifier", dest="overlap_thresh_classifier", help="overlap_thresh_classifier", default=0.5)
(options, args) = parser.parse_args()

if not options.test_path and not options.test_image:   # if filename is not given or image is not given
    parser.error('Error: path to test data must be specified. Pass --path OR --image to command line')

with open(options.config_filename, 'rb') as f_in:
    C = pickle.load(f_in)


saved = bool(options.save)
# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False



class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print(class_mapping_inv)

# Choose color of the boxes depending of the class
class_to_color = {class_mapping_inv[v]: np.random.randint(0, 255, 3) for v in class_mapping_inv}

C.num_rois = int(options.num_rois)

# load models rpn and classifier
model_rpn, model_classifier_only  = load_models_for_test(C, class_mapping)

bbox_threshold = float(options.bbox_threshold)
overlap_threshold_rpn = float(options.overlap_thresh_rpn)
overlap_threshold_class = float(options.overlap_thresh_classifier)

if options.test_path is not None:
    img_path = options.test_path
    print("###  Display images in the path ###")
    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join(img_path,img_name)

        img = cv2.imread(filepath)
        img, all_dets = predict_on_image(img, model_rpn, model_classifier_only, C, class_mapping_inv, class_to_color, bbox_threshold=bbox_threshold, overlap_thresh_rpn=overlap_threshold_rpn, overlap_thresh_classifier=overlap_threshold_class)
        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        cv2.imshow('img', img)
        cv2.waitKey(0)

if options.test_image is not None:
    print("###  Display image ###")
    img_name = options.test_image
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        print('Cannot open the file :', img_name)
        
    print(img_name)
    st = time.time()
    img = cv2.imread(img_name)
    img, all_dets = predict_on_image(img, model_rpn, model_classifier_only, C, class_mapping_inv, class_to_color, bbox_threshold=bbox_threshold, overlap_thresh_rpn=overlap_threshold_rpn, overlap_thresh_classifier=overlap_threshold_class)
    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    
    if saved == False:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    else:
        cv2.imwrite('test.jpg',img)