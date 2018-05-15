### Code structure

Documentation have been added to the following codes:

- data_generator.py: generating anchors with overlap and regression parameters from ground truth bbox (rpn net function)
- roi_helper.py: converting rpn output to roi, non-maximum suppression
- data_augment.py: augment input image by vertical and horizontal flip
- losses.py: losses function used in faster rcnn
- roi_pooling_conv.py: ROI pooling convolution layer
- vgg.py: faster rcnn implementation of vgg net