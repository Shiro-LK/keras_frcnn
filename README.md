# Keras Faster-RCNN

> this is a very userful implementation of faster-rcnn based on tensorflow and keras, the model is very clear and just saved in .h5 file, out of box to use, and easy to train on other data set with full support. if you have any question, feel free to ask me via wechat: jintianiloveu

## Requirements
Basically, this code supports both python2.7 and python3.5, the following package should installed:
* tensorflow
* keras
* scipy
* cv2

## Out of box model to predict

I have trained a model to predict on PASCAL VOC. Pretrained model in this link : https://drive.google.com/open?id=1lnacvFOAEM4S6U71O3PBRIqHTY0JhEw2
You can use transfer learning with this model (resnet50 backbone).

## Train New Dataset

to train a new dataset is also very simple and straight forward. Simply convert your detection label file whatever format into this format:

```
/path/training/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian,training
/path/training/image_2/000001.png,599.41,156.40,629.75,189.25,Truck,testing
```
Which is `/path/to/img.png,x1,y1,x2,y2,class_name,imageset`, with this simple file, we don't need class map file, our training program will statistic this automatically.

```
python train_frcnn_v2.py
```

## For Predict

If you want see how good your trained model is, simply run:
```
python test_frcnn_v2.py
```
you can also using `-p` to specific single image to predict, or send a path contains many images, our program will automatically recognise that.

**That's all, help you enjoy!**
