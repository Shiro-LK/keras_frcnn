# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:26:32 2018

@author: shiro
"""
import xml.etree.ElementTree as ET
import sys
import glob
import cv2
import numpy as np

def read_pascal_voc(f, filename, path_folder, pathtosave= '', mode='train'):

    tree = ET.parse(filename)
    root = tree.getroot()
    plot = False
    prints = False

    for child in root:
        if child.tag == 'filename':
            img = path_folder+'JPEGImages/'+child.text
            img_save=pathtosave+child.text
            cvimg = cv2.imread(img)
            #cvimg2 = cvimg.copy()
            try:
                (height, width) = cvimg.shape[:2]
            except:
                print(img)
        elif child.tag == 'object':
            classe = child.find('name').text
            coordinate = np.zeros(4)
            for box in child.find('bndbox'):
                if box.tag == 'xmin':
                    coordinate[0] = float(box.text)
                elif box.tag == 'ymin':
                    coordinate[1] = float(box.text)
                elif box.tag == 'xmax':
                    coordinate[2] = float(box.text)
                elif box.tag == 'ymax':
                    coordinate[3] = float(box.text)
            x1 = int(round(float(coordinate[0])))
            y1 = int(round(float(coordinate[1])))
            x2 = int(round(float(coordinate[2])))
            y2 = int(round(float(coordinate[3])))
#            x = int(coordinate[0])+int(coordinate[2])
#            x = int(x/2)
#            
#            y = int(coordinate[1])+int(coordinate[3])
#            y = int(y/2)
#            
#            w = int(coordinate[2])-int(coordinate[0])
#            w = int(w/2)
#            
#            h = int(coordinate[3])-int(coordinate[1])
#            h = int(h/2)
#            
#            if x-w>=0 and x+w<width:
#                pass
#            else:
#                if prints:
#                    print('image {}, class : {},  width : {}, height {}, x : {}, y : {}, w : {}, h :{} '.format(img, classe, width, height, x, y, w, h))
#                w = w-1
#                if plot:
#                    cv2.rectangle(cvimg, (x-w, y-h), (x+w, y+h),(0,255,0),3)
#                    cv2.imshow('wind', cvimg)
#                    cv2.rectangle(cvimg2, (int(coordinate[0]), int(coordinate[1])), (int(coordinate[2]), int(coordinate[3])),(0,255,0),3)
#                    cv2.imshow('wind2', cvimg2)
#                    cv2.waitKey(0)
#                    
#                
#            if y-h>=0 and y+h<height:
#                pass
#            else:
#                if prints:
#                    print('image {}, class : {},width : {}, height {}, x : {}, y : {}, w : {}, h :{} '.format(img, classe, width, height, x, y, w, h))
#                h = h-1
#                if plot:
#                    cv2.rectangle(cvimg, (x-w, y-h), (x+w, y+h),(0,255,0),3)
#                    cv2.imshow('wind', cvimg)
#                    cv2.rectangle(cvimg2, (int(coordinate[0]), int(coordinate[1])), (int(coordinate[2]), int(coordinate[3])),(0,255,0),3) 
#                    cv2.imshow('wind2', cvimg2)
#                    cv2.waitKey(0)
            if mode =='train':
                f.write(img_save+','+str(width)+','+ str(height)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classe+',training'+'\n') #x,y,w,h
            else:
                f.write(img_save+','+str(width)+','+ str(height)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classe+',testing'+'\n') #x,y,w,h

def save_pascal_voc(path_folder='../../dataset/VOCdevkit/VOC2007train/Annotations/', tosave='VOC2007train.txt', path_img='/DeepLearning/dataset/VOCdevkit/VOC2007test/JPEGImages/', mode = 'train'):
    with open(tosave, 'w') as f:
        full_path = [i for i in glob.glob(path_folder+'Annotations/'+'*.xml')]
        for p in full_path:
            read_pascal_voc(f, p , path_folder = path_folder, pathtosave=path_img, mode= mode)
            
def combine_files(listFile, filename):
    final_data = []
    for file in listFile:
        with open(file, 'r') as f:
            for data in f:
                final_data.append(data)
    with open(filename, 'w') as f:
        for data in final_data:
            f.write(data)
            
          
save_pascal_voc(path_folder='../dataset/VOCdevkit/VOC2007train/', tosave='VOC2007train.txt', path_img='../dataset/VOCdevkit/VOC2007train/JPEGImages/', mode='train')       
save_pascal_voc(path_folder='../dataset/VOCdevkit/VOC2007test/', tosave='VOC2007test.txt', path_img='../dataset/VOCdevkit/VOC2007test/JPEGImages/', mode='test')
combine_files(['VOC2007train.txt', 'VOC2007test.txt'], 'VOC2007.txt')
            
save_pascal_voc(path_folder='../dataset/VOCdevkit/VOC2012train/', tosave='VOC2012train.txt', path_img='dataset/VOCdevkit/VOC2012train/JPEGImages/', mode='train')       
#save_pascal_voc(path_folder='../dataset/VOCdevkit/VOC2012test/', tosave='VOC2012test.txt', path_img='dataset/VOCdevkit/VOC2012test/JPEGImages/', mode='test')
#combine_files(['VOC2012train.txt', 'VOC2012test.txt'], 'VOC2012.txt')
combine_files(['VOC2012train.txt', 'VOC2007.txt'], 'VOC++.txt')