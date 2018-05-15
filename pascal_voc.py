# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:26:32 2018

@author: shiro
"""
import xml.etree.ElementTree as ET
import sys
import glob
import cv2
def read_pascal_voc(f, filename, path, mode='train'):
    tree = ET.parse(filename)
    root = tree.getroot()
    plot = False
    prints = False
    for child in root:
        if child.tag == 'filename':
            img = path+child.text
            cvimg = cv2.imread(img)
            #cvimg2 = cvimg.copy()
            try:
                (height, width) = cvimg.shape[:2]
            except:
                print(img)
        elif child.tag == 'object':
            classe = child.find('name').text
            coordinate = []
            for box in child.find('bndbox'):
                coordinate.append(box.text)
            x1 = int(coordinate[0])
            y1 = int(coordinate[1])
            x2 = int(coordinate[2])
            y2 = int(coordinate[3])
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
                f.write(img+','+str(width)+','+ str(height)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classe+',training'+'\n') #x,y,w,h
            else:
                f.write(img+','+str(width)+','+ str(height)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classe+',testing'+'\n') #x,y,w,h

def save_pascal_voc(path_folder='../../dataset/VOCdevkit/VOC2007train/Annotations/', tosave='VOC2007train.txt', path_img='/DeepLearning/dataset/VOCdevkit/VOC2007test/JPEGImages/', mode = 'train'):
    with open(tosave, 'w') as f:
        full_path = [i.replace('\\', '/') for i in glob.glob(path_folder+'*.xml')]
        for p in full_path:
            read_pascal_voc(f, p , path = path_img, mode= mode)
            
def combine_files(listFile, filename):
    final_data = []
    for file in listFile:
        with open(file, 'r') as f:
            for data in f:
                final_data.append(data)
    with open(filename, 'w') as f:
        for data in final_data:
            f.write(data)
            
            
save_pascal_voc(path_folder='../dataset/VOCdevkit/VOC2007train/Annotations/', tosave='VOC2007train.txt', path_img='../dataset/VOCdevkit/VOC2007train/JPEGImages/', mode='train')       
save_pascal_voc(path_folder='../dataset/VOCdevkit/VOC2007test/Annotations/', tosave='VOC2007test.txt', path_img='../dataset/VOCdevkit/VOC2007test/JPEGImages/', mode='test')
combine_files(['VOC2007train.txt', 'VOC2007test.txt'], 'VOC2007.txt')