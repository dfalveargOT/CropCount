#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:39:02 2019

@author: DavidFelipe
"""

import cv2
import tensorflow as tf
import os
import numpy as np
import skimage
import progressbar
#path = "./Training/training/"
#classes = ["Acce/","Fail/"]

class DataAugmentation:
    
    def __init__(self,path, path_class, flip=1, rotate=1, noise=1, bil_filter=1,
                 R1_angle=45, R2_angle=70, tools=True):
        
        if tools == True:
            self.path = path
            self.path_class = path_class
            self.lista_elments = os.listdir(path + path_class)
            self.name_p = "A_"
            self.name_f = "FP"
            self.name_r = "RT"
            self.name_n = "NO"
            self.name_b = "BF"
            self.ext = ".png"
            self.flags = [flip, rotate, noise, bil_filter]
            for flag in self.flags:
                if flag == 0:
                    print("Function deactivated")
            print(self.path + self.path_class + self.name_p)
        self.R1_Angle = R1_angle
        self.R2_Angle = R2_angle
        
    def generate(self):
        val = len(self.lista_elments)
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=val)
        bar.start()
        counter = 0
        for name_image in self.lista_elments:
            checkpoint = self.check_data(name_image)
            if checkpoint == False:
                continue
            image_load = cv2.imread(self.path + self.path_class + name_image)
            #FLIP
            if self.flags[0] == 1:
                name_save = name_image
                processed_image, name_save = self.flip(image_load, name_save)
                self.save_data(processed_image, name_save)
#                print(name_save)
            #ROTATE
            if self.flags[1]  == 1:
                name_save = name_image
                processed_image, name_save = self.rotate(image_load, name_save)
                self.save_data(processed_image, name_save)
#                print(name_save)
            #NOISE
            if self.flags[2]  == 1:
                name_save = name_image
                processed_image, name_save = self.noise(image_load, name_save)
                self.save_data(processed_image, name_save)
#                print(name_save)
            #BILFILTER
            if self.flags[3]  == 1:
                name_save = name_image
                processed_image, name_save = self.bilateral_filter(image_load, name_save)
                self.save_data(processed_image, name_save)
#                print(name_save)
            counter += 1
            bar.update(counter)
            name_save = ""
#            if counter == 10:
#                break
        print("Total processed : " + str(counter))
            
    def save_data(self, list_img, list_name):
        quant = len(list_img)
        for item in range (0, quant):
            cv2.imwrite(list_name[item], list_img[item].astype('uint8'))
    
    def flip(self, img, img_name, full=True):
        flip = [np.fliplr(img)]
        name_save = -1
        if full == True:
            ide = [1]
            img_name = img_name[:img_name.find(".")]
            name_save = [self.path + self.path_class + self.name_p + img_name + self.name_f + str(ide[0]) + self.ext]
        return flip, name_save
    
    def rotate(self, img, img_name, full=True):

#        rot1 = ndimage.rotate(img, self.R1_Angle)
#        rot2 = ndimage.rotate(img, self.R2_Angle)
        rot1 = skimage.transform.rotate(img, angle=self.R1_Angle, mode='reflect', preserve_range=True).astype(np.uint8)
        rot2 = skimage.transform.rotate(img, angle=self.R2_Angle, mode='reflect', preserve_range=True).astype(np.uint8)
        rot = [rot1, rot2]
        name_save = -1
        if full == True:
            ide = [1,2]
            img_name = img_name[:img_name.find(".")]
            name_save1 = self.path + self.path_class + self.name_p + img_name + self.name_r + str(ide[0]) + self.ext
            name_save2 = self.path + self.path_class + self.name_p + img_name + self.name_r + str(ide[1]) + self.ext
            name_save = [name_save1, name_save2]
        return rot, name_save
    
    def noise(self, img, img_name, full=True):
        shape_img = img.shape
        shape = [shape_img[0], shape_img[1], shape_img[2]]
        image_hold = tf.placeholder(dtype = tf.float32, shape = shape)
        noise = tf.random_normal(shape=tf.shape(image_hold), mean=0.0, stddev=1.0,dtype=tf.float32)
        output = tf.add(image_hold, noise)
        with tf.Session() as session:
            result = session.run(output, feed_dict={image_hold: img})
            noise_image = [np.copy(result).astype("uint8")]
        name_save = -1
        if full == True:
            img_name = img_name[:img_name.find(".")]
            ide = [0]
            name_save = [self.path + self.path_class + self.name_p + img_name + self.name_n + str(ide[0]) + self.ext]
            
        return noise_image, name_save
    
    def bilateral_filter(self, img, img_name, full=True):
        bil_image = [cv2.bilateralFilter(img,10,100,100)]
        name_save = -1
        if full == True:
            ide = [1]
            img_name = img_name[:img_name.find(".")]
            name_save = [self.path + self.path_class + self.name_p + img_name + self.name_b + str(ide[0]) + self.ext]
        return bil_image, name_save
    
    def check_data(self, string):
        png = jpg = -1
        png = string.find(".png")
        jpg = string.find(".jpg")
        if(png > 0 or jpg > 0):
            flag = True
        else:
            flag = False
        return flag